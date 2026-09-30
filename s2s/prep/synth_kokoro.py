"""Synthesise speech for a text manifest with Kokoro-82M (Apache-2.0 TTS, 24 kHz).

Used ONLY to create training data (the runtime model has no TTS):
  * spoken user requests for the tool stage (several voices, for input variety)
  * a large single-voice corpus for the talker (e.g. the distilled replies)

Install first (Kaggle/Colab):  pip install "kokoro>=0.9" soundfile ; apt-get -qq install espeak-ng

    # user requests, random voice per row
    python -m s2s.prep.synth_kokoro --in data/manifests/hotel_train_text.jsonl \
        --out data/manifests/hotel_train_audio.jsonl --voices af_heart af_bella am_adam am_michael

    # talker corpus from distilled replies, one fixed voice
    python -m s2s.prep.synth_kokoro --in data/manifests/librispeech_train.jsonl --text-field response \
        --out data/manifests/talker_kokoro_raw.jsonl --voices af_heart --max-utts 20000
"""

from __future__ import annotations

import os
import random

import numpy as np
from tqdm import tqdm

from s2s.audio import save_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.utils import write_jsonl

SAMPLE_RATE = 24000


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-field", default="text")
    p.add_argument("--voices", nargs="+", default=["af_heart"])
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--speed-jitter", type=float, default=0.0, help="random speed in speed*(1 +/- jitter) per row")
    p.add_argument("--max-utts", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = config_from_args(args)

    from kokoro import KPipeline  # imported lazily: optional dependency

    rows = [r for r in load_manifest(args.inp) if r.get(args.text_field)]
    if args.max_utts:
        rows = rows[: args.max_utts]
    name = os.path.splitext(os.path.basename(args.out))[0]
    wav_dir = os.path.join(cfg.paths.data_dir, "tts", name)
    os.makedirs(wav_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    pipelines: dict[str, KPipeline] = {}
    rng = random.Random(args.seed)
    out_rows = []
    for r in tqdm(rows, desc="kokoro"):
        voice = rng.choice(args.voices)
        speed = args.speed * rng.uniform(1 - args.speed_jitter, 1 + args.speed_jitter) if args.speed_jitter else args.speed
        lang = voice[0]  # 'a' = American English, 'b' = British English
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang)
        path = os.path.join(wav_dir, f"{r['id']}.wav")
        text = r[args.text_field]
        if not os.path.exists(path):
            chunks = []
            for result in pipelines[lang](text, voice=voice, speed=speed):
                audio = result[2] if isinstance(result, tuple) else getattr(result, "audio", None)
                if audio is None:
                    continue
                chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
            if not chunks:
                print(f"skip {r['id']}: no audio")
                continue
            save_audio(path, np.concatenate(chunks).astype(np.float32), SAMPLE_RATE)
        row = {k: v for k, v in r.items() if k not in ("audio", "latent", "codes", "speech_frames", "duration")}
        row["text"] = text
        row["voice"] = voice
        row["audio"] = os.path.relpath(path, out_dir)
        out_rows.append(row)
    print(write_jsonl(args.out, out_rows), "rows ->", args.out)


if __name__ == "__main__":
    main()
