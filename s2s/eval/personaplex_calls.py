"""Spoken hotel calls for the PersonaPlex test (scripts/kaggle_personaplex.sh): each call is a guest speaking a few
requests with pauses for the agent to answer, synthesised with Kokoro (run in the Kokoro environment).
Standalone: kokoro, numpy, soundfile.

    python personaplex_calls.py --out-dir /kaggle/working/personaplex/calls

Writes <id>.wav (24 kHz mono, guest only) and calls.json: per call the guest turns with their start / end
time and the words a correct answer should contain (any one of them).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf

SR = 24000

# a made-up hotel; the facts are in the agent's prompt (PersonaPlex's customer-service prompt style)
HOTEL_PROMPT = (
    "You work for Sunset Villa which is a hotel and your name is Aria. Information: Check-in is from 3 PM and "
    "checkout is at 11 AM. Late checkout until 1 PM costs $30. Breakfast is served from 7 to 10 AM in the Lotus "
    "restaurant on the ground floor. The rooftop pool is open from 8 AM to 9 PM. The Wi-Fi network is SunsetGuest "
    "and the password is sunset2026. The airport shuttle leaves every hour from 6 AM and costs $15. Room service "
    "is available until 11 PM. For maintenance problems, an engineer can come to the room within 20 minutes.")

# (guest voice, [(what the guest says, words a correct answer contains)])
CALLS = {
    "checkout": ("am_michael", [
        ("Hi, what time is checkout?", ["11", "eleven"]),
        ("Could I stay until one on my last day?", ["30", "thirty", "1 pm", "one pm", "one o'clock"]),
        ("Okay, and when is breakfast served?", ["7", "seven"]),
    ]),
    "amenities": ("af_bella", [
        ("Hello, what's the wifi password?", ["sunset"]),
        ("Is the pool open in the evening?", ["9", "nine"]),
        ("And where is the restaurant?", ["ground", "lotus"]),
    ]),
    "transport": ("bm_george", [
        ("Good morning. How much is the airport shuttle?", ["15", "fifteen"]),
        ("What time does the first one leave?", ["6", "six"]),
        ("And what time can my friend check in today?", ["3", "three"]),
    ]),
    "requests": ("af_nicole", [
        ("Hi, my air conditioning isn't working.", ["engineer", "maintenance", "send", "someone", "20", "twenty"]),
        ("Can I still order room service? It's ten thirty.", ["11", "eleven", "yes"]),
        ("Great, I'd like to book a table for four at eight tonight.", ["four", "4", "eight", "8"]),
    ]),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gap", type=float, default=7.0, help="seconds of silence after each guest turn (agent's turn)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    from kokoro import KModel, KPipeline

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(args.device).eval()
    pipes: dict[str, KPipeline] = {}
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    for cid, (voice, turns) in CALLS.items():
        pipe = pipes.setdefault(voice[0], KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", model=model))
        parts, t, rows = [np.zeros(SR, np.float32)], 1.0, []
        for text, keywords in turns:
            wav = np.concatenate([r.audio.detach().float().cpu().numpy().reshape(-1) for r in pipe(text, voice=voice)])
            rows.append({"text": text, "start": round(t, 3), "end": round(t + len(wav) / SR, 3), "keywords": keywords})
            parts += [wav, np.zeros(int(args.gap * SR), np.float32)]
            t += len(wav) / SR + args.gap
        path = os.path.join(args.out_dir, f"{cid}.wav")
        sf.write(path, np.concatenate(parts), SR)
        out.append({"id": cid, "wav": path, "prompt": HOTEL_PROMPT, "turns": rows})
        print(f"{cid}: {t:.1f}s, {len(rows)} guest turns ({voice})", flush=True)
    with open(os.path.join(args.out_dir, "calls.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
