"""Talk to the trained agent with audio files (one file per user turn).

    python -m s2s.cli.chat --wav turn1.wav turn2.wav --out-dir outputs/chat

Each reply is written to <out-dir>/reply_<n>.wav, and the transcript, tool
calls and per-stage timings are printed. --stream feeds the audio in 80 ms
chunks through the end-of-turn detector (as a microphone would).
"""

from __future__ import annotations

import os

import numpy as np

from s2s.audio import load_audio, save_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.runtime.agent import VoiceAgent
from s2s.utils import Timer


def run_turn(session, wav: np.ndarray, stream: bool, sr: int) -> tuple[list[np.ndarray], dict]:
    timer = Timer()
    if stream:
        turn = session.stream_input()
        turn.timer = timer
        chunk = int(0.08 * sr)
        for i in range(0, len(wav), chunk):
            if turn.feed(wav[i: i + chunk]):
                break
        events = turn.finish()
    else:
        events = session.respond(wav, timer=timer)
    audio, timings = [], {}
    for ev in events:
        t = ev["type"]
        if t == "user_transcript":
            print(f"USER (ctc): {ev['text']}   [endpoint: {ev['endpoint_reason']}, eot_p {ev['eot_prob_last']}]")
        elif t == "token":
            print(ev["text"], end="", flush=True)
        elif t == "assistant_text":
            print(f"\nASSISTANT: {ev['text']}")
        elif t == "tool_call":
            print(f"TOOL CALL: {ev['call']}")
        elif t == "tool_result":
            print(f"TOOL RESULT: {ev['result']}")
        elif t == "audio":
            audio.append(ev["audio"])
        elif t == "timings":
            timings = ev["ms"]
    return audio, timings


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--wav", nargs="+", required=True)
    p.add_argument("--out-dir", default="outputs/chat")
    p.add_argument("--speech-llm-dir", default=None)
    p.add_argument("--talker-dir", default=None)
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--stream", action="store_true", help="feed audio in 80 ms chunks with endpointing")
    args = p.parse_args()
    cfg = config_from_args(args)
    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    res = make_reservation()
    session = agent.new_session(context=None if args.no_tools else reservation_context(res),
                                tools=None if args.no_tools else HOTEL_TOOLS,
                                backend=None if args.no_tools else HotelBackend(res))
    os.makedirs(args.out_dir, exist_ok=True)
    sr = agent.codec.sample_rate
    for n, path in enumerate(args.wav):
        print(f"\n=== turn {n + 1}: {path}")
        wav = load_audio(path, sr)
        audio, timings = run_turn(session, wav, args.stream, sr)
        if audio:
            out = os.path.join(args.out_dir, f"reply_{n + 1}.wav")
            save_audio(out, np.concatenate(audio), sr)
            print(f"reply audio -> {out} ({sum(len(x) for x in audio) / sr:.2f}s)")
        print("timings (ms since turn start): " + ", ".join(f"{k}={v:.0f}" for k, v in timings.items()))


if __name__ == "__main__":
    main()
