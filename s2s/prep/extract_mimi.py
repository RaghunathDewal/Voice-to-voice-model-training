"""Run the frozen Mimi encoder over a manifest and store features next to it.

    # speech INPUT features for the adapter (continuous latents + trailing silence)
    python -m s2s.prep.extract_mimi --mode latents --in data/manifests/librispeech_dev_raw.jsonl \
        --out data/manifests/librispeech_dev.jsonl

    # speech OUTPUT targets for the talker (discrete codes)
    python -m s2s.prep.extract_mimi --mode codes --in data/manifests/talker_train_raw.jsonl \
        --out data/manifests/talker_train.jsonl

All other fields of each row are copied. Existing feature files are reused,
so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import os

import zlib

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from s2s.audio import add_silence, load_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.codec import MimiCodec
from s2s.utils import resolve_device, write_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["latents", "codes"], required=True)
    p.add_argument("--feat-dir", default=None, help="default: <data_dir>/features/<manifest name>")
    p.add_argument("--max-seconds", type=float, default=35.0, help="skip longer utterances")
    p.add_argument("--max-utts", type=int, default=0)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)

    rows = load_manifest(args.inp)
    if args.max_utts:
        rows = rows[: args.max_utts]
    name = os.path.splitext(os.path.basename(args.out))[0]
    feat_dir = args.feat_dir or os.path.join(cfg.paths.data_dir, "features", name)
    os.makedirs(feat_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(args.out))

    codec = MimiCodec(cfg.codec.model, device, cfg.codec.num_codebooks)
    sr = codec.sample_rate
    silence = float(cfg.codec.trailing_silence_s) if args.mode == "latents" else 0.0
    bs = int(cfg.codec.extract_batch_size)
    suffix = "lat" if args.mode == "latents" else "codes"

    done, pending = [], []
    for r in rows:
        path = os.path.join(feat_dir, f"{r['id']}.{suffix}.npy")
        r["_feat"] = path
        (done if os.path.exists(path) else pending).append(r)

    out_rows, skipped = [], 0

    def finish(r: dict, n_speech_frames: int, duration: float) -> None:
        row = {k: v for k, v in r.items() if not k.startswith("_") and k != "audio"}
        row["audio"] = os.path.relpath(r["audio"], out_dir)
        key = "latent" if args.mode == "latents" else "codes"
        row[key] = os.path.relpath(r["_feat"], out_dir)
        row["duration"] = round(duration, 3)
        if args.mode == "latents":
            row["speech_frames"] = n_speech_frames
        out_rows.append(row)

    for r in tqdm(done, desc="existing", disable=not done):
        info = sf.info(r["audio"])
        n_samples = int(round(info.frames * sr / info.samplerate))
        finish(r, codec.num_frames(n_samples), n_samples / sr)

    batch: list[tuple[dict, np.ndarray]] = []

    def flush() -> None:
        if not batch:
            return
        wavs = [w for _, w in batch]
        feats = codec.encode_latents(wavs) if args.mode == "latents" else codec.encode_codes(wavs)
        for (r, w), feat in zip(batch, feats):
            n_speech = codec.num_frames(int(r["_n_samples"]))
            if args.mode == "latents":
                np.save(r["_feat"], feat.cpu().numpy().astype(np.float16))
            else:
                np.save(r["_feat"], feat.cpu().numpy().astype(np.int16))
            finish(r, n_speech, r["_n_samples"] / sr)
        batch.clear()

    for r in tqdm(pending, desc=f"mimi {args.mode}"):
        try:
            wav = load_audio(r["audio"], sr)
        except Exception as e:  # noqa: BLE001 - keep going on a broken file
            print(f"skip {r['id']}: {e}")
            skipped += 1
            continue
        if len(wav) > args.max_seconds * sr or len(wav) < sr * 0.2:
            skipped += 1
            continue
        peak = float(np.abs(wav).max())
        if peak > 1.0:
            wav = wav / peak
        r["_n_samples"] = len(wav)
        if silence:
            wav = add_silence(wav, sr, silence, seed=zlib.crc32(r["id"].encode()))
        batch.append((r, wav))
        if len(batch) >= bs:
            flush()
    flush()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    order = {r["id"]: i for i, r in enumerate(rows)}
    out_rows.sort(key=lambda x: order.get(x["id"], 0))
    write_jsonl(args.out, out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
