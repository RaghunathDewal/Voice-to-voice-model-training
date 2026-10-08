"""Concurrency test for the /live API: N simulated guests talking at the same time.

    python -m s2s.eval.load_test --url ws://127.0.0.1:7860/live --token $S2S_LIVE_TOKEN --concurrency 1 2 4 8 16

Each guest opens its own /live session (same setup as an application would send), then speaks
`--turns` questions from real speech clips, streamed at real-time speed in 100 ms chunks followed by
silence, exactly like a microphone. Tool calls are answered at once with a small fake result. Per
concurrency level it reports, over all turns:

  first audio   end of the guest's speech -> first byte of reply audio (what the guest feels as delay;
                includes end-of-turn detection, ~0.3-1 s of silence by config)
  turn          end of speech -> turn_complete (reply generated AND played)
  failed        turns with an error or no reply within --timeout

Guests start staggered over --ramp seconds, so turns overlap the way real calls do. Run it from a
machine close to the server (the droplet itself is best) so network time does not dominate.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import time

import numpy as np

from s2s.audio import load_audio

SR = 16000
CHUNK = SR // 10  # 100 ms


def clips(paths: list[str]) -> list[bytes]:
    out = []
    for p in paths:
        wav = load_audio(p, SR)
        out.append((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
    if not out:
        raise SystemExit("no speech clips found (use --clips)")
    return out


async def guest(idx: int, args, speech: list[bytes], results: list[dict]) -> None:
    import websockets

    url = args.url + (("&" if "?" in args.url else "?") + f"token={args.token}" if args.token else "")
    rng = random.Random(idx)
    await asyncio.sleep(rng.uniform(0, args.ramp))
    try:
        async with websockets.connect(url, max_size=None, open_timeout=60) as ws:
            await ws.send(json.dumps({"type": "setup", "system_prompt": args.system_prompt, "tools": [],
                                      "input_sample_rate": SR, "output_sample_rate": 24000}))
            while json.loads(await ws.recv()).get("type") != "setup_complete":
                pass
            silence = bytes(2 * int(SR * args.silence))
            for turn in range(args.turns):
                pcm = rng.choice(speech) + silence
                t_end_speech = None
                for off in range(0, len(pcm), 2 * CHUNK):  # real time: one 100 ms chunk every 100 ms
                    await ws.send(pcm[off:off + 2 * CHUNK])
                    if off + 2 * CHUNK >= len(pcm) - len(silence) and t_end_speech is None:
                        t_end_speech = time.perf_counter()
                    await asyncio.sleep(CHUNK / SR)
                first_audio, ok, err = None, False, None
                deadline = time.perf_counter() + args.timeout
                while time.perf_counter() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.perf_counter())
                    except asyncio.TimeoutError:
                        break
                    if isinstance(msg, bytes):
                        first_audio = first_audio or time.perf_counter()
                        continue
                    m = json.loads(msg)
                    if m["type"] == "tool_call":
                        await ws.send(json.dumps({"type": "tool_response", "id": m["id"], "response": {"ok": True}}))
                    elif m["type"] == "error":
                        err = m.get("message")
                    elif m["type"] == "turn_complete":
                        ok = True
                        break
                now = time.perf_counter()
                if err is None and not ok:
                    err = f"no turn_complete within {args.timeout:.0f} s"
                elif err is None and first_audio is None:
                    err = "reply had no audio"
                results.append({"guest": idx, "turn": turn, "ok": ok and err is None and first_audio is not None,
                                "first_audio": (first_audio - t_end_speech) if first_audio else None,
                                "turn_s": now - t_end_speech, "error": err})
                await asyncio.sleep(rng.uniform(0.3, 1.5))  # the guest listens, then speaks again
    except Exception as e:  # noqa: BLE001 - a refused / dropped connection is a result too
        results.append({"guest": idx, "turn": -1, "ok": False, "first_audio": None, "turn_s": None, "error": repr(e)})


def pct(values: list[float], q: float) -> str:
    return f"{np.percentile(values, q):5.2f}s" if values else "   -  "


async def run_level(n: int, args, speech: list[bytes]) -> dict:
    results: list[dict] = []
    t0 = time.perf_counter()
    await asyncio.gather(*(guest(i, args, speech, results) for i in range(n)))
    fa = [r["first_audio"] for r in results if r["ok"]]
    turns = [r["turn_s"] for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"{n:>4} guests | first audio p50 {pct(fa, 50)} p95 {pct(fa, 95)} max {pct(fa, 100)} | "
          f"turn p50 {pct(turns, 50)} | turns ok {len(fa)}/{len(results)} | {time.perf_counter() - t0:5.1f}s",
          flush=True)
    for r in failed[:3]:
        print(f"       failed: guest {r['guest']} turn {r['turn']}: {r['error']}")
    return {"concurrency": n, "first_audio": fa, "failed": len(failed)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:7860/live")
    p.add_argument("--token", default=os.environ.get("S2S_LIVE_TOKEN"))
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--turns", type=int, default=3, help="questions per guest")
    p.add_argument("--clips", nargs="*", default=sorted(glob.glob("demo/video/public/audio/guest_*.wav")))
    p.add_argument("--silence", type=float, default=1.5, help="seconds of silence after each question")
    p.add_argument("--ramp", type=float, default=3.0, help="guests start within this many seconds")
    p.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for a reply")
    p.add_argument("--system-prompt", default="You are the voice concierge of a holiday park. Be brief.")
    p.add_argument("--out", default=None, help="write all measurements as JSON")
    args = p.parse_args()
    speech = clips(args.clips)
    print(f"{len(speech)} speech clips, {args.turns} turns per guest, url {args.url}")
    levels = [asyncio.run(run_level(n, args, speech)) for n in args.concurrency]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(levels, f)


if __name__ == "__main__":
    main()
