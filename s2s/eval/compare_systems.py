"""Compare the three speech systems on the SAME new spoken hotel requests.

    python -m s2s.eval.compare_systems --n 80 --noisy --paraphrase --out results/compare
    # Kaggle: bash scripts/kaggle_compare.sh

Systems (all from Hugging Face checkpoints):
  0.6b     Parakeet -> speech_llm_pk4 -> fine-tuned Qwen3-0.6B (thinker_merged_v3) -> talker_v3      (half duplex)
  4b       Parakeet -> speech_llm_4b_v2 -> Qwen3-4B-Instruct-2507 -> talker_4b                       (half duplex)
  duplex   Nemotron -> Qwen3-0.6B duplex model (listens and speaks every 80 ms) -> Mimi              (full duplex)
plus "0.6b-text" / "4b-text": the same thinkers given the TYPED request (the upper bound for the speech path).

Test set (never seen in training): hotel requests from the v3 generator with a new seed (new property
profiles, tool subsets, facts), optionally re-worded by Qwen3-4B (--paraphrase: new phrasings, every number
and name kept), spoken by 4 Kokoro voices that no system was trained on, and with --noisy a second copy with
room echo / noise / phone line. Each request is one guest turn with its own system prompt and tools.
The duplex model hears it as a call: it greets first, the request starts at 5 s, then 7 s of silence.

Scored per system: tool rows = exact call (name + arguments); question rows = no tool call and the answer
states the fact; wrong actions = a tool call on a question row or the wrong tool; intelligibility = Whisper
on the system's own audio vs its own text (WER); latency = time to the first audio after the request ends
(half duplex: measured from the end of the audio, i.e. WITHOUT the ~0.3-0.4 s end-of-turn wait of the live
demo; duplex: includes its own end-of-turn decision). Writes summary.md / summary.json and examples.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch

from s2s.audio import load_audio, resample, save_audio
from s2s.augment import augment
from s2s.data.hotel import calls_match, score_example
from s2s.data.hotel_v3 import GenericBackend, generate_examples_v3
from s2s.eval.spoken_wer import spoken_wer
from s2s.utils import read_jsonl, resolve_device, write_jsonl

SR = 24000
HELD_OUT_VOICES = ["af_alloy", "af_aoede", "am_fenrir", "am_santa"]  # no system was trained on these
SYSTEMS = {
    "0.6b": dict(config="configs/small.yaml", speech_llm="checkpoints/speech_llm_pk4", talker="checkpoints/talker_v3",
                 sets=["thinker.model=checkpoints/thinker_merged_v3"]),
    "4b": dict(config="configs/qwen4b.yaml", speech_llm="checkpoints/speech_llm_4b_v2", talker="checkpoints/talker_4b",
               sets=[]),
}


# --------------------------------------------------------------------------- test set
def paraphrase(rows: list[dict], device, name: str = "Qwen/Qwen3-4B-Instruct-2507") -> int:
    """Re-word each request with Qwen3-4B; keep the original when a number, time or name would change."""
    import re

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()

    numbers = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"}

    def keep(text: str) -> set[str]:
        """Numbers and number words, and names (capitalised words that do not start a sentence)."""
        names = {m.group(2) for m in re.finditer(r"([^.!?]\s)([A-Z][a-z]+)", text)}
        return set(re.findall(r"\d+", text)) | (set(re.findall(r"[a-z]+", text.lower())) & numbers) | names

    changed = 0
    for i in range(0, len(rows), 16):
        chunk = rows[i: i + 16]
        prompts = [tok.apply_chat_template([{"role": "user", "content":
                   "Rewrite this hotel guest's spoken request in different, natural words, as the guest would say it on "
                   "the phone. Keep its meaning and every number, time, quantity, item and name exactly. Reply with "
                   f"the new sentence only.\n\n{r['text']}"}], add_generation_prompt=True, enable_thinking=False,
                   tokenize=False) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95)
        for r, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            new = tok.decode(seq, skip_special_tokens=True).strip().strip('"')
            low_new = keep(new.lower()) | keep(new)
            bad = (not new or new.lower() == r["text"].lower() or len(r["text"].split()) <= 4   # greetings stay
                   or len(new) > 3 * len(r["text"]) + 40 or re.match(r"^[^:\d]{0,40}:\s", new)
                   or '"' in new or re.search(r"how (can|may) i (help|assist)|assist you|i'?m sorry", new.lower()))
            if not bad and {k.lower() for k in keep(r["text"])} <= {k.lower() for k in low_new}:
                r["original_text"], r["text"] = r["text"], new
                changed += 1
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return changed


def build_texts(args, device) -> list[dict]:
    """Step 1 (GPU): the new requests, re-worded when asked. Cached in texts.jsonl."""
    path = os.path.join(args.out, "texts.jsonl")
    if os.path.exists(path):
        return read_jsonl(path)
    rows = generate_examples_v3(args.n, seed=args.seed, follow_up_prob=0.0)
    if args.paraphrase:
        print(f"paraphrased {paraphrase(rows, device, args.paraphrase_model)} of {len(rows)} requests", flush=True)
    write_jsonl(path, rows)
    return rows


def build_testset(args, device) -> list[dict]:
    """Step 2 (Kokoro TTS, may run in a separate Python 3.12 env): speech for every request (+ noisy copies)."""
    path = os.path.join(args.out, "testset.jsonl")
    if os.path.exists(path):
        return read_jsonl(path)
    rows = build_texts(args, device)
    from kokoro import KPipeline

    pipes, rng = {}, random.Random(args.seed)
    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    items = []
    for r in rows:
        voice = rng.choice(HELD_OUT_VOICES)
        pipe = pipes.setdefault(voice[0], KPipeline(lang_code=voice[0]))
        chunks = [res[2] if isinstance(res, tuple) else res.audio for res in pipe(r["text"], voice=voice,
                                                                                    speed=rng.uniform(0.9, 1.1))]
        wav = np.concatenate([c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c) for c in chunks if c is not None])
        p = os.path.join(wav_dir, f"{r['id']}.wav")
        save_audio(p, wav.astype(np.float32), SR)
        items.append({**r, "voice": voice, "audio": p, "condition": "clean"})
        if args.noisy:
            pn = os.path.join(wav_dir, f"{r['id']}_noisy.wav")
            save_audio(pn, augment(wav.astype(np.float32), SR, f"cmp/{r['id']}", p_phone=0.5), SR)
            items.append({**r, "voice": voice, "audio": pn, "condition": "noisy"})
    write_jsonl(path, items)
    return items


# --------------------------------------------------------------------------- systems
def run_half_duplex(name: str, items: list[dict], device, text_too: bool) -> list[dict]:
    from s2s.config import load_config
    from s2s.runtime.agent import VoiceAgent

    spec = SYSTEMS[name]
    sets = list(spec["sets"])
    if torch.cuda.device_count() > 1:
        sets.append("runtime.encoder_device=cuda:1")
    cfg = load_config(spec["config"], sets)
    agent = VoiceAgent(cfg, spec["speech_llm"], spec["talker"], device=str(device))
    out = []

    def turn(r, wav=None, text=None):
        sess = agent.new_session(system_prompt=r["system"], tools=r.get("tools"), backend=GenericBackend(r.get("tools")))
        events = sess.respond(wav) if wav is not None else sess.respond_text(text)
        calls, reply, audio, t0, first = [], "", [], time.time(), None
        for ev in events:
            if ev["type"] == "tool_call":
                calls.append(ev["call"])
            elif ev["type"] == "assistant_text":
                reply += (" " if reply else "") + ev["text"]
            elif ev["type"] == "audio":
                if first is None:
                    first = time.time() - t0
                audio.append(ev["audio"])
        return calls, reply.strip(), (np.concatenate(audio) if audio else np.zeros(0, np.float32)), first

    for i, r in enumerate(items):
        calls, reply, audio, first = turn(r, wav=load_audio(r["audio"], SR))
        out.append({"system": name, "id": r["id"], "condition": r["condition"], "calls": calls, "text": reply,
                    "audio": audio, "latency_s": first})
        if text_too and r["condition"] == "clean":
            calls, reply, _, _ = turn(r, text=r["text"])
            out.append({"system": f"{name}-text", "id": r["id"], "condition": "typed", "calls": calls, "text": reply,
                        "audio": None, "latency_s": None})
        if (i + 1) % 20 == 0:
            print(f"[{name}] {i + 1}/{len(items)}", flush=True)
    agent.thinker = agent.talker = agent.input_encoder = None  # free the GPU before the next system
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_duplex(items: list[dict], device, ckpt: str, start_s: float = 5.0, tail_s: float = 7.0) -> list[dict]:
    from s2s.eval.duplex_eval import speech_frames
    from s2s.models.codec import MimiCodec
    from s2s.models.encoders import build_encoder
    from s2s.runtime.duplex import DuplexSession, load_duplex, run_offline

    model, tok, cfg, _ = load_duplex(ckpt, device)
    codec = MimiCodec(cfg.codec.model, device, num_codebooks=int(cfg.codec.num_codebooks))
    enc_dev = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    enc = build_encoder("nemotron:nvidia/nemotron-speech-streaming-en-0.6b@1", enc_dev)
    out = []
    for i, r in enumerate(items):
        req = load_audio(r["audio"], SR)
        guest = np.concatenate([np.zeros(int(start_s * SR), np.float32), req, np.zeros(int(tail_s * SR), np.float32)])
        feats = enc.encode([resample(guest, SR, 16000)])[0]
        sess = DuplexSession(model, tok, r["system"], r.get("tools"), GenericBackend(r.get("tools")),
                             seed=hash(r["id"]) % 10000)
        res = run_offline(sess, feats, codec)
        g0, g1 = int(start_s / 0.08), int((start_s + len(req) / SR) / 0.08)
        calls = [t["call"] for t in res["tools"] if t.get("call") and t["frame"] >= g0]
        text = tok.decode([t for f, t in enumerate(res["texts"]) if f >= g0 and t != model.pad_id], skip_special_tokens=True)
        audio = res["audio"]
        hop = int(0.08 * SR)
        mask = speech_frames(audio)
        on = np.nonzero(mask[g1:])[0]
        reply_audio = audio[g0 * hop:]
        out.append({"system": "duplex", "id": r["id"], "condition": r["condition"], "calls": calls, "text": text.strip(),
                    "audio": reply_audio, "latency_s": float(on[0] * 0.08) if len(on) else None,
                    "spoke_during_request": bool(mask[g0:g1].sum() > 8)})
        if (i + 1) % 20 == 0:
            print(f"[duplex] {i + 1}/{len(items)}", flush=True)
    del model, codec, enc
    gc.collect()
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- scoring
def score(results: list[dict], items: dict, asr) -> tuple[list[dict], list[dict]]:
    table, examples = [], []
    keys = sorted({(r["system"], r["condition"]) for r in results}, key=lambda k: (k[0], k[1]))
    for system, cond in keys:
        rs = [r for r in results if r["system"] == system and r["condition"] == cond]
        ok = tool_ok = tool_n = fact_ok = fact_n = wrong = name_ok = 0
        refs, hyps, lats = [], [], []
        for r in rs:
            row = items[r["id"]]
            good = score_example(row, r["calls"], r["text"])
            ok += int(good)
            if row.get("tool_calls"):
                tool_n += 1
                tool_ok += int(good)
                name_ok += int(sorted(c.get("name") for c in r["calls"]) == sorted(c["name"] for c in row["tool_calls"]))
                wrong += int(bool(r["calls"]) and not calls_match(r["calls"], row["tool_calls"]))
            else:
                fact_n += 1
                fact_ok += int(good)
                wrong += int(bool(r["calls"]))
            if r.get("latency_s") is not None:
                lats.append(r["latency_s"])
            if asr is not None and r.get("audio") is not None and len(r["audio"]) > SR // 4 and r["text"]:
                refs.append(r["text"])
                hyps.append(asr({"raw": resample(r["audio"], SR, 16000), "sampling_rate": 16000}, return_timestamps=True,
                                generate_kwargs={"language": "en", "task": "transcribe"})["text"])
            if not good and sum(1 for e in examples if e["system"] == system and e["condition"] == cond) < 6:
                examples.append({"system": system, "condition": cond, "request": row["text"],
                                 "expected": row.get("tool_calls") or row.get("answer_contains"),
                                 "got_calls": r["calls"], "got_text": r["text"][:200]})
        n = len(rs)

        def pct(a, b):
            return round(100 * a / b, 1) if b else None

        table.append({"system": system, "condition": cond, "n": n, "overall_%": pct(ok, n),
                      "tool_exact_%": pct(tool_ok, tool_n), "tool_name_%": pct(name_ok, tool_n),
                      "question_answered_%": pct(fact_ok, fact_n), "wrong_action_%": pct(wrong, n),
                      "intelligibility_wer_%": round(100 * spoken_wer(refs, hyps), 1) if refs else None,
                      "latency_median_s": round(float(np.median(lats)), 2) if lats else None,
                      "spoke_during_request_%": pct(sum(r.get("spoke_during_request", False) for r in rs), n)
                      if system == "duplex" else None})
    return table, examples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", default="0.6b,4b,duplex")
    p.add_argument("--n", type=int, default=80, help="new requests (x2 with --noisy)")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--noisy", action="store_true")
    p.add_argument("--paraphrase", action="store_true", help="re-word the requests with Qwen3-4B (new phrasings)")
    p.add_argument("--paraphrase-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--no-text", action="store_true", help="skip the typed-text reference runs")
    p.add_argument("--duplex-ckpt", default="checkpoints/duplex")
    p.add_argument("--whisper", default="openai/whisper-small")
    p.add_argument("--out", default="results/compare")
    p.add_argument("--device", default="auto")
    p.add_argument("--stage", default="all", choices=["all", "texts", "audio"],
                   help="texts: only write the requests; audio: only speak them (Kokoro); all: everything")
    args = p.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "texts":
        build_texts(args, device)
        return
    items = build_testset(args, device)
    if args.stage == "audio":
        return
    by_id = {r["id"]: r for r in items}
    print(f"{len(items)} test recordings ({len(by_id)} requests)", flush=True)

    results = []
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        t = time.time()
        if name == "duplex":
            results += run_duplex(items, device, args.duplex_ckpt)
        else:
            results += run_half_duplex(name, items, device, not args.no_text)
        print(f"{name} done in {(time.time() - t) / 60:.1f} min", flush=True)

    asr = None
    if args.whisper:
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=args.whisper,
                       device=torch.device("cuda:1") if torch.cuda.device_count() > 1 else device)
    table, examples = score(results, by_id, asr)
    cols = ["system", "condition", "n", "overall_%", "tool_exact_%", "tool_name_%", "question_answered_%",
            "wrong_action_%", "intelligibility_wer_%", "latency_median_s", "spoke_during_request_%"]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    md += "".join("| " + " | ".join("" if row.get(c) is None else str(row[c]) for c in cols) + " |\n" for row in table)
    md += "\nhalf-duplex latency excludes the live end-of-turn wait (~0.3-0.4 s); duplex latency includes it.\n"
    md += "\n## Misses (up to 6 per system and condition)\n\n" + "".join(
        f"- **{e['system']} / {e['condition']}**: \"{e['request']}\"  \n  expected `{e['expected']}`; got calls "
        f"`{e['got_calls']}`, said \"{e['got_text']}\"\n" for e in examples)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"table": table, "examples": examples,
                   "outputs": [{k: v for k, v in r.items() if k != "audio"} for r in results]}, f, indent=1, default=str)
    print(md)


if __name__ == "__main__":
    main()
