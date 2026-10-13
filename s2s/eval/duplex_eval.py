"""Evaluate the full-duplex model by letting it run on held-out calls, frame by frame, in real time order.

    python -m s2s.eval.duplex_eval --ckpt checkpoints/duplex --manifest data/manifests/duplex_eval.jsonl \
        --n 50 --out results/duplex
    # your own recording as the guest (mono wav, any rate); the agent's reply is mixed in the right channel
    python -m s2s.eval.duplex_eval --ckpt checkpoints/duplex --wav me.wav --out results/duplex_me

The guest channel is fixed (pre-recorded), so the agent is measured on what it controls:
  * tools: right tool and arguments (precision / recall against the call script)
  * answers: did the agent start speaking within 2 s after each guest turn, and how fast (response gap)
  * mid-turn pauses: did it wrongly answer during a guest pause after a preface
  * talking over: share of guest speech time the agent spoke through (its short backchannels excluded)
  * interruptions: when the guest starts a real turn while the agent talks, how long until it stops;
    guest "mhm"s while the agent talks: did it (correctly) keep going
  * intelligibility: Whisper transcript of the agent's audio vs the agent's own inner text (WER)
  * content: the agent's whole text vs the scripted agent lines (rough WER), and the real-time factor
Writes <out>/summary.json, summary.md and a stereo wav per call (left = guest, right = agent).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from s2s.audio import load_audio, resample, save_audio
from s2s.data.datasets import load_manifest
from s2s.data.hotel_v3 import GenericBackend
from s2s.eval.spoken_wer import spoken_wer
from s2s.runtime.duplex import DuplexSession, load_duplex, run_offline
from s2s.utils import resolve_device

FRAME = 0.08
SR = 24000


def speech_frames(wav: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    hop = int(FRAME * SR)
    n = int(np.ceil(len(wav) / hop))
    out = np.zeros(n, bool)
    for i in range(n):
        x = wav[i * hop:(i + 1) * hop]
        out[i] = len(x) and 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) > threshold_db
    # bridge gaps shorter than 3 frames (pauses between words)
    idx = np.nonzero(out)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 3:
            out[a:b] = True
    return out


def regions(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def score_call(timeline: dict, agent_speech: np.ndarray, texts: list[int], tok, pad_id: int, tools_made: list,
               tools_ref: list) -> dict:
    n = len(agent_speech)
    guest = timeline["guest"]
    res = {"turns": 0, "answered": 0, "gaps": [], "pause_answers": 0, "pauses": 0, "over": 0, "guest_frames": 0,
           "stops": [], "stop_events": 0, "bc_kept": 0, "bc_events": 0}
    for g in guest:
        s, e = int(g["start"] / FRAME), int(np.ceil(g["end"] / FRAME))
        if g["kind"] in ("turn", "interrupt"):
            res["turns"] += 1
            window = agent_speech[e: min(n, e + 25)]  # 2 s
            on = np.nonzero(window)[0]
            if len(on):
                res["answered"] += 1
                res["gaps"].append(float(on[0] * FRAME))
            res["guest_frames"] += max(0, min(e, n) - s)
            talking = agent_speech[s:min(e, n)]
            long_runs = [b - a for a, b in regions(talking) if b - a > 8]  # > 0.64 s: more than a backchannel
            res["over"] += int(sum(long_runs))
            if s < n and agent_speech[max(0, s - 1)]:
                res["stop_events"] += 1
                after = agent_speech[s:min(n, s + 25)]
                quiet = np.nonzero(~after)[0]
                res["stops"].append(float(quiet[0] * FRAME) if len(quiet) else 2.0)
        elif g["kind"] == "preface":
            res["pauses"] += 1
            pause_end = next((int(x["start"] / FRAME) for x in guest if x["event"] == g["event"] and x["kind"] != "preface"),
                             e + 8)
            if any(b - a > 8 for a, b in regions(agent_speech[e:min(pause_end, n)])):
                res["pause_answers"] += 1
        elif g["kind"] == "backchannel" and s < n and agent_speech[s]:
            res["bc_events"] += 1
            res["bc_kept"] += int(agent_speech[min(n - 1, s + 8)])
    made = [json.dumps(c["call"], sort_keys=True) for c in tools_made if c.get("call")]
    ref = [json.dumps({"name": t["call"]["name"], "arguments": t["call"]["arguments"]}, sort_keys=True) for t in tools_ref]
    res["tool_tp"] = sum(min(made.count(x), ref.count(x)) for x in set(ref))
    res["tool_made"], res["tool_ref"] = len(made), len(ref)
    res["tool_name_tp"] = sum(min([json.loads(m)["name"] for m in made].count(x), [json.loads(r)["name"] for r in ref].count(x))
                              for x in {json.loads(r)["name"] for r in ref})
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--wav", default=None, help="your own recording as the guest channel")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default="results/duplex")
    p.add_argument("--whisper", default="openai/whisper-small")
    p.add_argument("--audio-temperature", type=float, default=0.6)
    p.add_argument("--audio-top-k", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--graphs", action="store_true", help="the 80 ms frame from one CUDA graph")
    p.add_argument("--graph-max-tokens", type=int, default=4096)
    args = p.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    model, tok, cfg, _ = load_duplex(args.ckpt, device)
    from s2s.models.codec import MimiCodec

    codec = MimiCodec(cfg.codec.model, device, num_codebooks=int(cfg.codec.num_codebooks))
    pad_id = model.pad_id
    asr = None
    if args.whisper:
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=args.whisper, device=device)

    if args.wav:  # one free-form call with your own voice
        from s2s.data.hotel_v3 import _Ctx
        from s2s.models.encoders import build_encoder
        import random

        ctx = _Ctx(random.Random(0))
        enc = build_encoder("nemotron:nvidia/nemotron-speech-streaming-en-0.6b@1", device)
        g24 = load_audio(args.wav, SR)
        g24 = np.concatenate([g24, np.zeros(int(4 * SR), np.float32)])  # time to answer after you stop
        feats = enc.encode([resample(g24, SR, 16000)])[0]
        tools = list(ctx.tools.values()) or None
        sess = DuplexSession(model, tok, ctx.system, tools, GenericBackend(tools), args.audio_temperature, args.audio_top_k,
                             graphs=args.graphs, graph_max_tokens=args.graph_max_tokens)
        out = run_offline(sess, feats, codec)
        agent = out["audio"]
        n = max(len(g24), len(agent))
        stereo = np.stack([np.pad(g24, (0, n - len(g24))), np.pad(agent, (0, n - len(agent)))], 1)
        save_audio(os.path.join(args.out, "call.wav"), stereo, SR)
        text = tok.decode([t for t in out["texts"] if t != pad_id])
        print(json.dumps({"agent_text": text, "tools": out["tools"], "rtf": round(out["rtf"], 2)}, indent=1))
        return

    rows = load_manifest(args.manifest)[: args.n]
    root = os.path.dirname(os.path.abspath(args.manifest))
    tot: dict = {}
    per_call, refs_c, hyps_c, refs_i, hyps_i, rtfs = [], [], [], [], [], []
    for r in rows:
        d = os.path.join(root, r["dir"])
        feats = torch.from_numpy(np.load(os.path.join(d, "guest.npy")).astype(np.float32))
        with open(os.path.join(d, "timeline.json")) as f:
            timeline = json.load(f)
        tools = r.get("tools")
        sess = DuplexSession(model, tok, r["system"], tools, GenericBackend(tools), args.audio_temperature,
                             args.audio_top_k, seed=hash(r["id"]) % 10000,
                             graphs=args.graphs, graph_max_tokens=args.graph_max_tokens)
        out = run_offline(sess, feats, codec)
        agent = out["audio"]
        rtfs.append(out["rtf"])
        mask = speech_frames(agent)
        sc = score_call(timeline, mask, out["texts"], tok, pad_id, out["tools"], timeline["tools"])
        for k, v in sc.items():
            tot[k] = tot.get(k, 0) + v if not isinstance(v, list) else tot.get(k, []) + v
        text = tok.decode([t for t in out["texts"] if t not in (pad_id,) and t < len(tok)], skip_special_tokens=True)
        refs_c.append(" ".join(s["text"] for s in timeline["agent"]))
        hyps_c.append(text)
        if asr is not None and len(agent) and text.strip():
            heard = asr({"raw": resample(agent, SR, 16000), "sampling_rate": 16000}, return_timestamps=True,
                        generate_kwargs={"language": "en", "task": "transcribe"})["text"]
            refs_i.append(text)
            hyps_i.append(heard)
        g24 = load_audio(os.path.join(d, "guest.wav"), SR)
        n = max(len(g24), len(agent))
        save_audio(os.path.join(args.out, f"{r['id']}.wav"),
                   np.stack([np.pad(g24, (0, n - len(g24))), np.pad(agent, (0, n - len(agent)))], 1), SR)
        per_call.append({"id": r["id"], "agent_text": text, "tools": out["tools"], **{k: v for k, v in sc.items()
                                                                                       if not isinstance(v, list)}})
        print(f"{r['id']}: answered {sc['answered']}/{sc['turns']}, tools {sc['tool_tp']}/{sc['tool_ref']} "
              f"(made {sc['tool_made']}), rtf {out['rtf']:.2f}", flush=True)

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None

    summary = {
        "calls": len(rows),
        "answered_within_2s_%": pct(tot["answered"], tot["turns"]),
        "response_gap_median_s": round(float(np.median(tot["gaps"])), 2) if tot["gaps"] else None,
        "answered_during_guest_pause_%": pct(tot["pause_answers"], tot["pauses"]),
        "talking_over_guest_%": pct(tot["over"], tot["guest_frames"]),
        "stopped_within_1s_when_guest_spoke_%": pct(sum(s <= 1.0 for s in tot["stops"]), tot["stop_events"]),
        "stop_latency_median_s": round(float(np.median(tot["stops"])), 2) if tot["stops"] else None,
        "kept_talking_through_guest_mhm_%": pct(tot["bc_kept"], tot["bc_events"]),
        "tool_recall_%": pct(tot["tool_tp"], tot["tool_ref"]),
        "tool_precision_%": pct(tot["tool_tp"], tot["tool_made"]),
        "tool_name_recall_%": pct(tot["tool_name_tp"], tot["tool_ref"]),
        "intelligibility_wer_%": round(100 * spoken_wer(refs_i, hyps_i), 1) if refs_i else None,
        "content_wer_vs_script_%": round(100 * spoken_wer(refs_c, hyps_c), 1),
        "real_time_factor_mean": round(float(np.mean(rtfs)), 2),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "calls": per_call}, f, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("| metric | value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in summary.items()))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
