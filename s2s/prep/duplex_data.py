"""Data for the full-duplex experiment: two-channel hotel calls with frame-aligned streams.

    # 1. call scripts (text, tool calls, backchannels, interruptions)
    python -m s2s.prep.duplex_data script --train 8000 --eval 200
    # 2. speech: guest = a random Kokoro voice per call, agent = one fixed voice; both channels mixed on a timeline
    python -m s2s.prep.duplex_data audio --split train --gpus 2
    # 3. features: Nemotron streaming vectors of the guest channel (with noise / phone line), Mimi codes of the agent
    python -m s2s.prep.duplex_data features --split train --gpus 2 --augment-prob 0.5 --phone-prob 0.4

Every call gets data/duplex/<split>/<id>/{guest.wav, agent.wav, timeline.json, guest.npy, agent.npy}; the
manifest data/manifests/duplex_<split>.jsonl lists calls whose features exist.

Timeline (seconds; the frame grid is 80 ms, Mimi's and the listener's rate):
  * the agent "decides" at a frame (its text starts there) and its audio starts LEAD frames later
  * after a guest turn the agent answers 0.25-0.6 s after the guest stops; a tool call happens at that
    decision frame, and the answer follows after the tool result
  * a guest turn with a preface pauses 0.5-0.9 s after it; the agent may say "Mhm." in the pause, and must
    NOT answer yet (the turn is not finished)
  * a guest backchannel ("Mhm.") lands inside agent speech; the agent keeps talking
  * an interrupted agent reply is cut 0.25-0.45 s after the guest starts talking
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np

from s2s.audio import load_audio, resample, save_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.dist import resolve_num_workers, run_sharded, strip_gpu_args
from s2s.utils import read_jsonl, write_jsonl

SR = 24000          # Kokoro and Mimi
FRAME = 0.08        # 12.5 Hz
LEAD = 2            # agent text leads its audio by 2 frames (160 ms)
AGENT_VOICE = "af_heart"
GUEST_VOICES = ["af_bella", "af_sarah", "af_sky", "af_nova", "af_river", "af_jessica", "af_kore", "am_adam",
                "am_michael", "am_eric", "am_liam", "am_onyx", "am_echo", "bf_alice", "bf_isabella", "bf_lily",
                "bm_daniel", "bm_lewis", "bm_fable", "af_nicole", "am_puck", "bf_emma", "bm_george"]


def call_dir(cfg, split: str, cid: str) -> str:
    return os.path.join(cfg.paths.data_dir, "duplex", split, cid)


def calls_path(cfg, split: str) -> str:
    return os.path.join(cfg.paths.data_dir, "manifests", f"duplex_{split}_calls.jsonl")


def manifest_path(cfg, split: str) -> str:
    return os.path.join(cfg.paths.data_dir, "manifests", f"duplex_{split}.jsonl")


# --------------------------------------------------------------------------- timeline
def build_timeline(events: list[dict], durations: dict[int, dict], rng: random.Random) -> dict:
    """Place every event on one clock. durations[i] = {"main": s, "preface": s, "bc": s} for event i."""
    def frame_at(sec: float) -> int:
        return int(math.ceil(sec / FRAME - 1e-9))

    agent, guest, tools = [], [], []
    t = rng.uniform(0.2, 0.5)            # time the next event may start
    decide_at: float | None = t           # when the agent should start its next "say"
    next_guest_start: float | None = None
    for i, ev in enumerate(events):
        d = durations.get(i, {})
        if ev["who"] == "agent" and ev["kind"] == "tool":
            f = frame_at(decide_at if decide_at is not None else t)
            tools.append({"frame": f, "call": ev["call"], "result": ev["result"]})
            decide_at = (f + 1) * FRAME + rng.uniform(0.1, 0.3)
            continue
        if ev["who"] == "agent":
            f = frame_at(decide_at if decide_at is not None else t)
            a0 = (f + LEAD) * FRAME
            seg = {"text": ev["text"], "frame": f, "audio_start": a0, "audio_end": a0 + d["main"], "full": d["main"],
                   "event": i}
            if ev.get("interrupted") and d["main"] > 1.6:
                g0 = a0 + rng.uniform(0.8, max(0.9, d["main"] - 0.6))
                seg["audio_end"] = g0 + rng.uniform(0.25, 0.45)
                seg["cut"] = True
                next_guest_start = g0
            agent.append(seg)
            t = seg["audio_end"]
            decide_at = None
            continue
        if ev["kind"] == "backchannel":   # guest "mhm" inside the last agent segment
            last = agent[-1]
            start = last["audio_start"] + rng.uniform(0.35, 0.7) * (last["audio_end"] - last["audio_start"])
            guest.append({"text": ev["text"], "start": start, "end": start + d["main"], "kind": "backchannel",
                          "event": i})
            continue
        # guest turn
        start = next_guest_start if next_guest_start is not None else t + rng.uniform(0.3, 0.9)
        next_guest_start = None
        if ev.get("preface"):
            pre_end = start + d["preface"]
            guest.append({"text": ev["preface"], "start": start, "end": pre_end, "kind": "preface", "event": i})
            pause = rng.uniform(0.5, 0.9)
            if ev.get("agent_backchannel"):
                f = frame_at(pre_end + 0.05)
                a0 = (f + LEAD) * FRAME
                agent.append({"text": ev["agent_backchannel"], "frame": f, "audio_start": a0,
                              "audio_end": a0 + d["bc"], "full": d["bc"], "event": i, "backchannel": True})
                pause = max(pause, a0 + d["bc"] + 0.1 - pre_end)
            start = pre_end + pause
        end = start + d["main"]
        guest.append({"text": ev["text"], "start": start, "end": end, "kind": "interrupt" if ev.get("interrupt") else "turn",
                      "event": i})
        t = end
        decide_at = end + rng.uniform(0.25, 0.6)
    last = max([s["audio_end"] for s in agent] + [g["end"] for g in guest])
    return {"agent": agent, "guest": guest, "tools": tools, "duration": last + rng.uniform(0.5, 1.0)}


def mix_channels(timeline: dict, clips: dict[tuple, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = int(math.ceil(timeline["duration"] * SR))
    guest = np.zeros(n, np.float32)
    agent = np.zeros(n, np.float32)
    fade = int(0.03 * SR)
    for seg in timeline["agent"]:
        wav = clips[("agent", seg["event"], "bc" if seg.get("backchannel") else "main")]
        s = int(round(seg["audio_start"] * SR))
        keep = min(len(wav), int(round((seg["audio_end"] - seg["audio_start"]) * SR)))
        w = wav[:keep].copy()
        if seg.get("cut") and keep > fade:
            w[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        agent[s: s + len(w)] += w[: max(0, n - s)]
    for seg in timeline["guest"]:
        wav = clips[("guest", seg["event"], "preface" if seg["kind"] == "preface" else "main")]
        s = int(round(seg["start"] * SR))
        guest[s: s + len(wav)] += wav[: max(0, n - s)]
    return np.clip(guest, -1, 1), np.clip(agent, -1, 1)


# --------------------------------------------------------------------------- stages
def stage_script(cfg, args) -> None:
    from s2s.data.duplex_calls import generate_calls

    for split, n, seed in (("train", args.train, 1), ("eval", args.eval, 2)):
        if n:
            print(write_jsonl(calls_path(cfg, split), generate_calls(n, seed=seed)), f"{split} calls")


class _Kokoro:
    def __init__(self):
        from kokoro import KPipeline

        self.KPipeline, self.pipes = KPipeline, {}

    def __call__(self, text: str, voice: str, speed: float) -> np.ndarray:
        lang = voice[0]
        if lang not in self.pipes:
            self.pipes[lang] = self.KPipeline(lang_code=lang)
        chunks = []
        for result in self.pipes[lang](text, voice=voice, speed=speed):
            audio = result[2] if isinstance(result, tuple) else getattr(result, "audio", None)
            if audio is not None:
                chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(int(0.3 * SR), np.float32)
        nz = np.nonzero(np.abs(wav) > 1e-3)[0]  # trim Kokoro's leading / trailing silence
        return wav[max(0, nz[0] - 240): nz[-1] + 480] if len(nz) else wav


def stage_audio(cfg, args) -> None:
    calls = read_jsonl(calls_path(cfg, args.split))
    tts = _Kokoro()
    done = 0
    for k, c in enumerate(calls):
        if args.shard is not None and k % args.num_shards != args.shard:
            continue
        d = call_dir(cfg, args.split, c["id"])
        if os.path.exists(os.path.join(d, "timeline.json")):
            continue
        rng = random.Random(f"{args.split}/{c['id']}")
        voice = rng.choice(GUEST_VOICES)
        speed = rng.uniform(0.9, 1.1)
        clips, durations = {}, {}
        for i, ev in enumerate(c["events"]):
            if ev["kind"] == "tool":
                continue
            who = ev["who"]
            v, sp = (AGENT_VOICE, 1.0) if who == "agent" else (voice, speed)
            clips[(who, i, "main")] = tts(ev["text"], v, sp)
            dur = {"main": len(clips[(who, i, "main")]) / SR}
            if ev.get("preface"):
                clips[("guest", i, "preface")] = tts(ev["preface"], voice, speed)
                dur["preface"] = len(clips[("guest", i, "preface")]) / SR
            if ev.get("agent_backchannel"):
                clips[("agent", i, "bc")] = tts(ev["agent_backchannel"], AGENT_VOICE, 1.0)
                dur["bc"] = len(clips[("agent", i, "bc")]) / SR
            durations[i] = dur
        timeline = build_timeline(c["events"], durations, rng)
        timeline["guest_voice"] = voice
        g, a = mix_channels(timeline, clips)
        os.makedirs(d, exist_ok=True)
        save_audio(os.path.join(d, "guest.wav"), g, SR)
        save_audio(os.path.join(d, "agent.wav"), a, SR)
        with open(os.path.join(d, "timeline.json"), "w") as f:  # written last: marks the call as done
            json.dump(timeline, f)
        done += 1
        if done % 200 == 0:
            print(f"[audio{'' if args.shard is None else f' shard {args.shard}'}] {done} calls", flush=True)
    print(f"audio: {done} new calls")


def stage_features(cfg, args) -> None:
    import torch

    from s2s.augment import augment
    from s2s.models.codec import MimiCodec
    from s2s.models.encoders import build_encoder
    from s2s.utils import resolve_device

    calls = read_jsonl(calls_path(cfg, args.split))
    device = resolve_device(cfg.device)
    todo = []
    for k, c in enumerate(calls):
        if args.shard is not None and k % args.num_shards != args.shard:
            continue
        d = call_dir(cfg, args.split, c["id"])
        if os.path.exists(os.path.join(d, "guest.wav")) and not os.path.exists(os.path.join(d, "agent.npy")):
            todo.append(c)
    if todo:
        enc = build_encoder(args.encoder, device)
        codec = MimiCodec(cfg.codec.model, device, num_codebooks=int(cfg.codec.num_codebooks))
    for b in range(0, len(todo), args.batch_size):
        batch = todo[b: b + args.batch_size]
        guests, agents = [], []
        for c in batch:
            d = call_dir(cfg, args.split, c["id"])
            g = load_audio(os.path.join(d, "guest.wav"), SR)
            if args.split == "train" and random.Random(c["id"]).random() < args.augment_prob:
                g = augment(g, SR, f"duplex/{c['id']}", p_phone=args.phone_prob)
            guests.append(resample(g, SR, 16000))
            agents.append(load_audio(os.path.join(d, "agent.wav"), SR))
        feats = enc.encode(guests)
        codes = codec.encode_codes(agents)
        for c, f, cd in zip(batch, feats, codes):
            d = call_dir(cfg, args.split, c["id"])
            t = min(f.shape[0], cd.shape[1])
            np.save(os.path.join(d, "guest.npy"), f[:t].numpy().astype(np.float16))
            np.save(os.path.join(d, "agent.npy"), cd[:, :t].cpu().numpy().astype(np.int16))
            if args.drop_audio:  # the features are all training needs; saves ~3 MB per second of call
                for w in ("guest.wav", "agent.wav"):
                    os.remove(os.path.join(d, w))
        if (b // args.batch_size) % 20 == 0:
            print(f"[features{'' if args.shard is None else f' shard {args.shard}'}] {b + len(batch)}/{len(todo)}",
                  flush=True)
        del feats, codes
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.shard is None:
        write_manifest(cfg, args.split, calls, args.encoder)


def write_manifest(cfg, split: str, calls: list[dict], encoder: str) -> None:
    rows = []
    for c in calls:
        d = call_dir(cfg, split, c["id"])
        if not os.path.exists(os.path.join(d, "agent.npy")):
            continue
        rel = os.path.relpath(d, os.path.join(cfg.paths.data_dir, "manifests"))
        rows.append({**c, "dir": rel, "encoder": encoder})
    print(write_jsonl(manifest_path(cfg, split), rows), f"calls -> {manifest_path(cfg, split)}")


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("stage", choices=["script", "audio", "features"])
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--eval", type=int, default=200)
    p.add_argument("--split", default="train")
    p.add_argument("--encoder", default="nemotron:nvidia/nemotron-speech-streaming-en-0.6b@1")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--augment-prob", type=float, default=0.5)
    p.add_argument("--phone-prob", type=float, default=0.4)
    p.add_argument("--drop-audio", action="store_true", help="features: delete the wavs once a call is done")
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args()
    cfg = config_from_args(args)
    if args.stage == "script":
        return stage_script(cfg, args)
    n = resolve_num_workers(args.gpus)
    if args.shard is None and n > 1 and cfg.device != "cpu":
        run_sharded("s2s.prep.duplex_data", strip_gpu_args(sys.argv[1:]), n)
        if args.stage == "features":
            write_manifest(cfg, args.split, read_jsonl(calls_path(cfg, args.split)), args.encoder)
        return
    (stage_audio if args.stage == "audio" else stage_features)(cfg, args)


if __name__ == "__main__":
    main()
