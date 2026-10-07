"""Download an English ASR dataset from the Hugging Face Hub (parquet) and write a raw manifest.

Audio is decoded and stored as 16 kHz FLAC under <data_dir>/hf_asr/<name>/, so
the rest of the pipeline (extract_mimi, distill, training) treats it exactly
like LibriSpeech. Only the parquet shards needed for --max-hours are downloaded.

    # Common Voice 17 English: many accents, real consumer microphones
    python -m s2s.prep.hf_asr --preset commonvoice --split train --max-hours 150 --name cv_train
    # People's Speech (clean): spontaneous / broadcast English
    python -m s2s.prep.hf_asr --preset peoples --split train --max-hours 60 --name peoples_train
    # SLURP: real voice-assistant commands; SpokenWOZ: real phone dialogues with history
    python -m s2s.prep.hf_asr --preset slurp --split train --max-utts 8000 --name slurp_train
    python -m s2s.prep.hf_asr --preset spokenwoz --split train --max-utts 8000 --name spokenwoz_train
    # EdAcc: accented conversational English (evaluation)
    python -m s2s.prep.hf_asr --preset edacc --split validation --name edacc_valid
    # AMI meetings (spontaneous speech; headset and distant room mic), MLS English (many speakers),
    # People's Speech "dirty" (noisier), Indian-accented Common Voice
    python -m s2s.prep.hf_asr --preset ami_sdm --split train --max-hours 15 --min-words 4 --name ami_sdm_train
    python -m s2s.prep.hf_asr --preset mls --split train --max-hours 25 --max-per-accent 0.1 --name mls_train
    python -m s2s.prep.hf_asr --preset commonvoice --split train --accent-match "India|South Asia" --name cv_india_train

Writes <data_dir>/manifests/<name>_raw.jsonl with id, audio, text, duration,
source and (when the dataset has it) accent.
"""

from __future__ import annotations

import io
import os
import random
import re

import numpy as np
import soundfile as sf
from tqdm import tqdm

from s2s.audio import resample, to_mono
from s2s.cli_common import base_parser, config_from_args
from s2s.utils import write_jsonl

PRESETS = {
    # repo, file prefix per split, text column, extra filter
    "commonvoice": {"repo": "fixie-ai/common_voice_17_0", "prefix": "en/{split}-", "text": "sentence",
                    "accent": "accent"},
    "peoples": {"repo": "MLCommons/peoples_speech", "prefix": "clean/{split}-", "text": "text"},
    "peoples_dirty": {"repo": "MLCommons/peoples_speech", "prefix": "dirty/{split}-", "text": "text"},
    # AMI meeting corpus (CC BY 4.0): ihm = close-talk headset, sdm = one distant microphone in the room
    "ami_ihm": {"repo": "edinburghcstr/ami", "prefix": "ihm/{split}-", "text": "text", "accent": "speaker_id"},
    "ami_sdm": {"repo": "edinburghcstr/ami", "prefix": "sdm/{split}-", "text": "text", "accent": "speaker_id"},
    # Multilingual LibriSpeech, English (CC BY 4.0): read speech from thousands of speakers
    "mls": {"repo": "parler-tts/mls_eng", "prefix": "data/{split}-", "text": "transcript", "accent": "speaker_id"},
    "edacc": {"repo": "edinburghcstr/edacc", "prefix": "data/{split}-", "text": "text", "accent": "accent"},
    "voxpopuli_accented": {"repo": "facebook/voxpopuli", "prefix": "en_accented/{split}-",
                           "text": "normalized_text", "accent": "accent"},
    "fleurs": {"repo": "google/fleurs", "prefix": "parquet-data/en_us/{split}-", "text": "transcription"},
    # Indian-accented English (CC BY 4.0, accept its terms on HF first); only a "test" split exists
    "svarah": {"repo": "ai4bharat/Svarah", "prefix": "data/{split}-", "text": "text", "audio": "audio_filepath",
               "accent": "primary_language"},
    # real people giving voice-assistant commands, close-talk and far-field mics (CC BY 4.0)
    "slurp": {"repo": "marcel-gohsen/slurp", "prefix": "data/{split}-", "text": "transcript", "accent": "intent"},
    # real customer-service phone calls, one user turn per row, with the dialogue so far (see the dataset card)
    "spokenwoz": {"repo": "pirxus/spokenwoz-whisper", "prefix": "data/{split}-", "text": "text", "history": "context"},
}


def spokenwoz_history(ctx, max_exchanges: int = 2) -> list[dict]:
    """SpokenWOZ `context` column -> the last user/agent exchanges as chat messages."""
    if not isinstance(ctx, dict):
        return []
    users, agents = ctx.get("text") or [], ctx.get("agent_text") or []
    msgs = []
    for u, a in list(zip(users, agents))[-max_exchanges:]:
        u, a = clean_text(u), clean_text(a)
        if u and a:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs

SR = 16000
# EdAcc / People's Speech markup that is not speech
_MARKUP = re.compile(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)")


def clean_text(text: str) -> str:
    text = _MARKUP.sub(" ", str(text or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def usable(text: str, min_words: int) -> bool:
    return len(text.split()) >= min_words and re.search(r"[A-Za-z]", text) is not None


def decode_audio(cell) -> tuple[np.ndarray, int]:
    data = cell.get("bytes") if isinstance(cell, dict) else None
    if data is None and isinstance(cell, dict) and cell.get("array") is not None:
        return np.asarray(cell["array"], dtype=np.float32), int(cell["sampling_rate"])
    if data is None:
        raise ValueError("no audio bytes")
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - mp3 on libsndfile builds without mp3 support
        import librosa

        wav, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    return to_mono(np.asarray(wav, dtype=np.float32)), int(sr)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--split", default="train", help="train | validation | test (as named in the repo)")
    p.add_argument("--name", required=True, help="manifest name, e.g. cv_train")
    p.add_argument("--max-hours", type=float, default=0.0, help="stop after this much audio (0 = all)")
    p.add_argument("--max-utts", type=int, default=0)
    p.add_argument("--max-per-accent", type=float, default=0.0,
                   help="cap hours per accent label (balances accents; unlabelled rows are capped too)")
    p.add_argument("--max-shards", type=int, default=0, help="download at most this many parquet shards (0 = no cap)")
    p.add_argument("--accent-match", default=None,
                   help="keep only rows whose accent label matches this regex (e.g. 'India|South Asia')")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = config_from_args(args)

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    preset = PRESETS[args.preset]
    prefix = preset["prefix"].format(split=args.split)
    files = sorted(s.rfilename for s in HfApi().dataset_info(preset["repo"]).siblings
                   if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet files matching {preset['repo']}/{prefix}*")
    random.Random(args.seed).shuffle(files)  # spread speakers/topics across the whole split
    if args.max_shards:
        files = files[: args.max_shards]
    print(f"{preset['repo']} {args.split}: {len(files)} parquet shards")

    audio_dir = os.path.join(cfg.paths.data_dir, "hf_asr", args.name)
    manifests = os.path.join(cfg.paths.data_dir, "manifests")
    download_dir = os.path.join(cfg.paths.data_dir, "hf_asr", "_parquet")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(manifests, exist_ok=True)
    limit_s = args.max_hours * 3600 if args.max_hours else float("inf")
    accent_cap = args.max_per_accent * 3600 if args.max_per_accent else float("inf")
    rows: list[dict] = []
    total_s, per_accent, skipped = 0.0, {}, 0
    bar = tqdm(total=None if limit_s == float("inf") else int(limit_s), unit="s", desc=args.name, mininterval=10)
    for fname in files:
        if total_s >= limit_s or (args.max_utts and len(rows) >= args.max_utts):
            break
        local = hf_hub_download(preset["repo"], fname, repo_type="dataset", local_dir=download_dir)
        for rec in (r for b in pq.ParquetFile(local).iter_batches(batch_size=64) for r in b.to_pylist()):
            if total_s >= limit_s or (args.max_utts and len(rows) >= args.max_utts):
                break
            text = clean_text(rec.get(preset["text"], ""))
            if text.isupper():  # AMI: "YEAH I THINK SO" -> "Yeah i think so"
                text = re.sub(r"\bi\b", "I", text.lower().capitalize())
            if not usable(text, args.min_words):
                skipped += 1
                continue
            if args.preset == "commonvoice" and (rec.get("down_votes") or 0) > (rec.get("up_votes") or 0):
                skipped += 1
                continue
            accent = str(rec.get(preset.get("accent", ""), "") or "").strip() or "unlabelled"
            if args.accent_match and not re.search(args.accent_match, accent, re.I):
                skipped += 1
                continue
            if per_accent.get(accent, 0.0) >= accent_cap:
                skipped += 1
                continue
            try:
                wav, sr = decode_audio(rec[preset.get("audio", "audio")])
            except Exception:  # noqa: BLE001 - skip broken clips
                skipped += 1
                continue
            dur = len(wav) / sr
            if not args.min_seconds <= dur <= args.max_seconds:
                skipped += 1
                continue
            uid = f"{args.name}_{len(rows):07d}"
            path = os.path.join(audio_dir, uid + ".flac")
            if not os.path.exists(path):
                sf.write(path, resample(wav, sr, SR), SR, format="FLAC")
            row = {"id": uid, "audio": os.path.relpath(path, manifests), "text": text,
                   "duration": round(dur, 3), "source": args.preset}
            if "accent" in preset:
                row["accent"] = accent
            if preset.get("history"):
                hist = spokenwoz_history(rec.get(preset["history"]))
                if hist:
                    row["history"] = hist
            rows.append(row)
            total_s += dur
            per_accent[accent] = per_accent.get(accent, 0.0) + dur
            bar.update(int(dur))
        os.remove(local)  # keep the disk free; the FLAC copies are what we use
    bar.close()
    out = os.path.join(manifests, f"{args.name}_raw.jsonl")
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows, {total_s / 3600:.1f} h -> {out} (skipped {skipped})")
    top = sorted(per_accent.items(), key=lambda kv: -kv[1])[:15]
    print("hours per accent:", ", ".join(f"{k[:40]}: {v / 3600:.1f}" for k, v in top))


if __name__ == "__main__":
    main()
