"""Our voice agent (adapter v3 + Qwen3-4B, any runtime.tts_url voice) on the same spoken hotel calls as the
PersonaPlex test, scored by the same report (s2s/eval/personaplex_report.py), so the two are compared like for like.

Each guest turn is fed in 80 ms pieces from just before the guest speaks, as live: the real end-of-turn detection
decides when the guest has finished. The agent's reply audio is placed in the call at
    (where the end of the turn was detected) + (compute time until its first audio),
so the reply delay in the report = end-of-turn wait + compute (no network). Tool calls are made against the
built-in demo hotel backend and shown in the text column.

    python -m s2s.eval.v3_calls --config configs/qwen4b_v3.yaml --calls calls.json --out-dir out \\
        --speech-llm-dir checkpoints/speech_llm_v3 --talker-dir checkpoints/talker_4b --set runtime.tts_url=...
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

from s2s.audio import load_audio, resample
from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation

STYLE = " You are speaking on the phone: reply in one or two short spoken sentences, no lists, no emoji."


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--calls", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--speech-llm-dir", default=None)
    p.add_argument("--talker-dir", default=None)
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--label", default="", help="e.g. the checkpoint step, shown in the report")
    args = p.parse_args()
    cfg = config_from_args(args)
    from s2s.runtime.agent import VoiceAgent

    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    agent.warm_up()
    sr, hop = agent.codec.sample_rate, agent.codec.hop
    frame_s = hop / sr
    os.makedirs(args.out_dir, exist_ok=True)
    gpu = "cpu"
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    for call in json.load(open(args.calls)):
        turns = call.get("turns") or []
        if not turns:
            continue
        guest = load_audio(call["wav"], sr)
        out = np.zeros(len(guest), np.float32)
        text = ["PAD"] * (len(guest) // hop + 1)
        tools = None if args.no_tools else HOTEL_TOOLS
        sess = agent.new_session(system_prompt=call["prompt"] + STYLE, tools=tools,
                                 backend=None if tools is None else HotelBackend(make_reservation()))
        rows = []
        for k, t in enumerate(turns):
            nxt = turns[k + 1]["start"] if k + 1 < len(turns) else len(guest) / sr
            turn = sess.stream_input()
            pos = max(0, int((t["start"] - 0.3) * sr))
            while pos < int(nxt * sr) and not turn.feed(guest[pos: pos + hop]):
                pos += hop
            end_at = min(pos + hop, len(guest)) / sr  # audio time when the end of the turn was detected
            t0, first, pieces, said, heard, ms = time.time(), None, [], [], "", {}
            for ev in turn.finish():
                kind = ev.get("type")
                if kind == "audio" and len(ev["audio"]):
                    first = time.time() - t0 if first is None else first
                    w = np.asarray(ev["audio"], np.float32).reshape(-1)
                    pieces.append(w if ev.get("sample_rate", sr) == sr else resample(w, ev["sample_rate"], sr))
                elif kind == "assistant_text" and ev.get("text", "").strip():
                    said.append(ev["text"].strip())
                elif kind == "tool_call":
                    c = ev["call"]
                    said.append(f"[{c.get('name')} {json.dumps(c.get('arguments', {}))}]")
                elif kind == "user_transcript":
                    heard = ev.get("asr") or ev.get("text") or ""
                elif kind == "timings":
                    ms = ev["ms"]
            if pieces:
                start = int((end_at + first) * sr)
                reply = np.concatenate(pieces)[: max(0, len(out) - start)]
                out[start: start + len(reply)] += reply
                idx = min(len(text) - 1, start // hop)
                text[idx] = " " + " ".join(said)
            rows.append({"guest": t["text"], "heard": heard, "said": " ".join(said), "end_detected_s": end_at,
                         "first_audio_s": first, "timings_ms": ms})
            print(f"[v3] {call['id']} | guest: {t['text']!r} | heard: {heard!r} | said: {' '.join(said)[:150]!r} | "
                  f"end of turn detected {end_at - t['end']:+.2f}s vs clip end, first audio after "
                  f"{(first or 0) * 1000:.0f} ms", flush=True)
        sf.write(os.path.join(args.out_dir, f"{call['id']}.wav"), np.clip(out, -1, 1), sr)
        fa = [r["first_audio_s"] * 1000 for r in rows if r["first_audio_s"] is not None]
        stats = {"kind": "turn", "first_audio_ms_p50": float(np.median(fa)) if fa else None,
                 "first_audio_ms_max": float(max(fa)) if fa else None, "gpu": gpu, "label": args.label,
                 "tts": str(cfg.runtime.get("tts_url") or "talker")}
        with open(os.path.join(args.out_dir, f"{call['id']}.json"), "w") as f:
            json.dump({"id": call["id"], "frame_s": frame_s, "text": text, "stats": stats, "turns": rows}, f)


if __name__ == "__main__":
    main()
