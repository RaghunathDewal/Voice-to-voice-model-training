"""Experiment 5: measure latency on this GPU.

Each wav is played into the agent in 80 ms chunks followed by silence, as a
microphone would deliver it. Reported per turn:

  endpoint_wait_ms   audio time between the end of speech and the end-of-turn decision
                     (a setting + EOT head behaviour, not compute)
  compute_ms         wall-clock from the end-of-turn decision to the first reply audio
  total_ms           endpoint_wait_ms + compute_ms  (excludes network/playback buffers)
  feed_ms_per_frame  compute per 80 ms of input while the user speaks (must be < 80)

    python -m s2s.eval.latency --wav data/tts/hotel_eval_audio/*.wav --max 20
"""

from __future__ import annotations

import time

import numpy as np

from s2s.audio import add_silence, load_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.runtime.agent import VoiceAgent
from s2s.utils import Timer, cuda_sync, save_json


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--wav", nargs="+", required=True)
    p.add_argument("--max", type=int, default=20)
    p.add_argument("--speech-llm-dir", default=None)
    p.add_argument("--talker-dir", default=None)
    p.add_argument("--out", default="outputs/latency.json")
    args = p.parse_args()
    cfg = config_from_args(args)
    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    sr, hop = agent.codec.sample_rate, agent.codec.hop
    frame_ms = 1000.0 * hop / sr

    # warm-up (CUDA kernels, allocator)
    res = make_reservation()
    warm = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=HotelBackend(res))
    for _ in warm.respond(add_silence(np.zeros(sr, dtype=np.float32), sr, 0.5)):
        pass

    rows = []
    for path in args.wav[: args.max]:
        res = make_reservation()
        session = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=HotelBackend(res))
        speech = load_audio(path, sr)
        wav = add_silence(speech, sr, 2.0, noise_db=-65.0)
        timer = Timer()
        turn = session.stream_input()
        turn.timer = timer
        feed_times, fed = [], 0
        ended = False
        for i in range(0, len(wav), hop):
            t0 = time.perf_counter()
            ended = turn.feed(wav[i: i + hop])
            cuda_sync(agent.device)
            feed_times.append((time.perf_counter() - t0) * 1000)
            fed = i + hop
            if ended:
                break
        endpoint_wait = max(0.0, (fed - len(speech)) / sr * 1000.0)
        timer.reset()
        timer.mark("endpoint")
        first_audio = None
        reply_seconds = 0.0
        for ev in turn.finish():
            if ev["type"] == "audio":
                reply_seconds += len(ev["audio"]) / sr
            if ev["type"] == "timings":
                first_audio = ev["ms"].get("first_audio_out")
                marks = ev["ms"]
        row = {"wav": path, "ended_by": turn.endpointer.reason or "end_of_input", "endpoint_wait_ms": endpoint_wait,
               "compute_ms": first_audio, "total_ms": (endpoint_wait + first_audio) if first_audio else None,
               "feed_ms_per_frame": float(np.mean(feed_times)), "reply_seconds": reply_seconds, "marks": marks}
        rows.append(row)
        print(f"{path}: wait {endpoint_wait:.0f} ms ({row['ended_by']}), compute {first_audio or float('nan'):.0f} ms, "
              f"feed {row['feed_ms_per_frame']:.1f} ms/{frame_ms:.0f}ms frame")

    ok = [r for r in rows if r["total_ms"] is not None]
    summary = {k: {"p50": pct([r[k] for r in ok], 50), "p90": pct([r[k] for r in ok], 90)}
               for k in ("endpoint_wait_ms", "compute_ms", "total_ms", "feed_ms_per_frame")}
    print("\nsummary (ms):")
    for k, v in summary.items():
        print(f"  {k:20s} p50 {v['p50']:.0f}   p90 {v['p90']:.0f}")
    print("Add network + client playback buffer (typically 80-180 ms) for user-perceived latency.")
    save_json(args.out, {"summary": summary, "turns": rows})


if __name__ == "__main__":
    main()
