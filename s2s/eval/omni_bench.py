"""Concurrency benchmark for a Qwen-Omni model served by vLLM-Omni (speech in -> speech out, one model).

    python -m s2s.eval.omni_bench --url http://127.0.0.1:8091 --model Qwen/Qwen2.5-Omni-7B \
        --clips demo/video/public/audio/guest_*.wav --concurrency 1 8 16 32 64 --turns 3 --out omni.json

Reference point for our own pipeline (s2s/cli/batch_live.py + s2s/eval/load_test.py) on the same GPU. Each guest
sends its whole spoken question (a wav clip) in one OpenAI chat request with text + audio output, streamed, then
"listens" to the reply (waits for its length) before the next question, as in a call. Per turn:

  first_text    request sent -> first text token
  first_audio   request sent -> first reply audio chunk (vLLM-Omni streams text; audio may only arrive whole,
                after the talker and the vocoder finish: then first_audio == total)
  total         request sent -> response complete

These start when the guest's whole clip is in, so they compare with the SERVER-SIDE numbers of our pipeline
(scripts/turn_timings.py: first_audio from the end-of-turn decision), not with load_test's end-of-speech numbers,
which add the end-of-turn wait (silence) on top.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import random
import statistics
import threading
import time
import urllib.request

# Qwen-Omni only speaks with its own system sentence first (otherwise the talker warns / may not produce audio)
QWEN_OMNI_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech.")
HOTEL = ("You are the voice concierge of the Marina View Hotel. Checkout is at 11 AM, breakfast is served from 7 to "
         "10:30 AM in the lobby restaurant, the pool is open from 7 AM to 10 PM and the Wi-Fi password is on the key "
         "card. Be brief and friendly.")


def audio_seconds(b64: str) -> float:
    try:
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
        return len(wav) / sr
    except Exception:
        return 0.0


def one_turn(url: str, model: str, clip_b64: str, system: str, max_tokens: int, stream: bool, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": [{"type": "text", "text": system}]},
                     {"role": "user", "content": [{"type": "audio_url",
                                                   "audio_url": {"url": f"data:audio/wav;base64,{clip_b64}"}}]}],
        "modalities": ["text", "audio"],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first_text = first_audio = None
    text, audio = [], []

    def got(kind: str, payload: str) -> None:
        nonlocal first_text, first_audio
        if not payload:
            return
        ms = (time.perf_counter() - t0) * 1000
        if kind == "audio":
            first_audio = first_audio or ms
            audio.append(payload)
        else:
            first_text = first_text or ms
            text.append(payload)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if stream:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    c = json.loads(line[5:])
                    for ch in c.get("choices", []):
                        d = ch.get("delta") or {}
                        mod = c.get("modality") or ch.get("modality")
                        if mod == "audio" or d.get("audio"):
                            got("audio", (d.get("audio") or {}).get("data") or d.get("content") or "")
                        else:
                            got("text", d.get("content") or "")
            else:
                c = json.loads(r.read())
                for ch in c.get("choices", []):
                    m = ch.get("message") or {}
                    if m.get("audio"):
                        got("audio", m["audio"].get("data", ""))
                    elif m.get("content"):
                        got("text", m["content"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "total_ms": (time.perf_counter() - t0) * 1000}
    total = (time.perf_counter() - t0) * 1000
    secs = sum(audio_seconds(a) for a in audio)
    return {"ok": bool(audio), "first_text_ms": first_text, "first_audio_ms": first_audio, "total_ms": total,
            "audio_chunks": len(audio), "reply_s": round(secs, 2), "text": "".join(text)[:300],
            **({} if audio else {"error": "no audio in the reply"})}


def stats(v: list[float]) -> dict:
    v = sorted(x for x in v if x is not None)
    if not v:
        return {}
    return {"p50": round(statistics.median(v)), "p95": round(v[round(0.95 * (len(v) - 1))]), "max": round(v[-1])}


def run_level(a, clips: list[str], n: int) -> dict:
    turns: list[dict] = []
    lock = threading.Lock()

    def guest(g: int) -> None:
        time.sleep(random.uniform(0, min(a.ramp, 0.25 * n)))
        for t in range(a.turns):
            res = one_turn(a.url, a.model, clips[(g + t) % len(clips)], a.system, a.max_tokens, not a.no_stream,
                           a.timeout)
            res["guest"], res["turn"] = g, t
            with lock:
                turns.append(res)
            time.sleep(res.get("reply_s", 0) + 0.5)  # the guest listens to the reply, then talks again

    t0 = time.time()
    th = [threading.Thread(target=guest, args=(g,)) for g in range(n)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    ok = [r for r in turns if r["ok"]]
    out = {"guests": n, "turns": len(turns), "failed": len(turns) - len(ok), "wall_s": round(time.time() - t0, 1),
           "first_text_ms": stats([r["first_text_ms"] for r in ok]),
           "first_audio_ms": stats([r["first_audio_ms"] for r in ok]),
           "total_ms": stats([r["total_ms"] for r in ok]),
           "audio_streamed": sum(1 for r in ok if r["first_audio_ms"] < r["total_ms"] - 50),
           "reply_s": stats([1000 * r["reply_s"] for r in ok]),
           "errors": sorted({r["error"] for r in turns if "error" in r})[:5], "detail": turns}
    fa, ft = out["first_audio_ms"], out["first_text_ms"]
    print(f"{n:4d} guests | first audio p50 {fa.get('p50', '-')} p95 {fa.get('p95', '-')} max {fa.get('max', '-')} ms"
          f" | first text p50 {ft.get('p50', '-')} ms | audio streamed {out['audio_streamed']}/{len(ok)}"
          f" | failed {out['failed']}/{len(turns)}", flush=True)
    for e in out["errors"]:
        print("      error:", e, flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--clips", nargs="+", default=sorted(glob.glob("demo/video/public/audio/guest_*.wav")))
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ramp", type=float, default=8.0, help="guests start at random within this many seconds")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--system", default=f"{QWEN_OMNI_SYSTEM} {HOTEL}")
    p.add_argument("--out", default="omni_bench.json")
    a = p.parse_args()
    if not a.clips:
        raise SystemExit("no clips: pass --clips path/*.wav")
    clips = [base64.b64encode(open(c, "rb").read()).decode() for c in a.clips]

    w = one_turn(a.url, a.model, clips[0], a.system, a.max_tokens, not a.no_stream, a.timeout)  # warm-up
    print(f"warm-up: {json.dumps({k: v for k, v in w.items() if k != 'detail'})}", flush=True)
    levels = [run_level(a, clips, n) for n in a.concurrency]
    json.dump({"model": a.model, "url": a.url, "turns_per_guest": a.turns, "max_tokens": a.max_tokens,
               "stream": not a.no_stream, "levels": levels}, open(a.out, "w"), indent=1)
    print(f"results: {a.out}", flush=True)


if __name__ == "__main__":
    main()
