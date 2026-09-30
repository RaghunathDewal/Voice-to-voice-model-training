"""Run the frozen Mimi encoder over a manifest and store features next to it.

    # speech INPUT features for the adapter (continuous latents + trailing silence)
    python -m s2s.prep.extract_mimi --mode latents --in data/manifests/librispeech_dev_raw.jsonl \
        --out data/manifests/librispeech_dev.jsonl

    # the same, sounding like real microphones (noise, reverb, phone/laptop EQ) for 60% of clips
    python -m s2s.prep.extract_mimi --mode latents --augment-prob 0.6 --in data/manifests/cv_train_raw.jsonl \
        --out data/manifests/cv_train.jsonl

    # speech OUTPUT targets for the talker (discrete codes)
    python -m s2s.prep.extract_mimi --mode codes --in data/manifests/talker_train_raw.jsonl \
        --out data/manifests/talker_train.jsonl

All other fields of each row are copied. Existing feature files are reused,
so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from s2s.audio import add_silence, load_audio
from s2s.augment import augment
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.codec import MimiCodec
from s2s.dist import resolve_num_workers, run_sharded, strip_gpu_args
from s2s.utils import resolve_device, write_jsonl


def save_npy_atomic(path: str, arr: np.ndarray) -> None:
    """Write via a temp file so an interrupted run never leaves a truncated feature file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def argparse_hidden() -> str:
    import argparse

    return argparse.SUPPRESS


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["latents", "codes"], required=True)
    p.add_argument("--feat-dir", default=None, help="default: <data_dir>/features/<manifest name>")
    p.add_argument("--max-seconds", type=float, default=35.0, help="skip longer utterances")
    p.add_argument("--max-utts", type=int, default=0)
    p.add_argument("--augment-prob", type=float, default=0.0,
                   help="latents only: fraction of utterances passed through s2s.augment (noise, reverb, mic EQ)")
    p.add_argument("--gpus", type=int, default=0, help="worker processes, one per GPU (0 = all visible GPUs)")
    p.add_argument("--shard", type=int, default=None, help=argparse_hidden())
    p.add_argument("--num-shards", type=int, default=1, help=argparse_hidden())
    args = p.parse_args()
    cfg = config_from_args(args)

    n_workers = resolve_num_workers(args.gpus)
    if args.shard is None and n_workers > 1 and cfg.device != "cpu":
        # each GPU encodes its share of the files; this process then writes the manifest
        run_sharded("s2s.prep.extract_mimi", strip_gpu_args(sys.argv[1:]), n_workers)
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
    for i, r in enumerate(rows):
        path = os.path.join(feat_dir, f"{r['id']}.{suffix}.npy")
        r["_feat"] = path
        if args.shard is not None and i % args.num_shards != args.shard:
            continue  # another GPU's share
        (done if os.path.exists(path) else pending).append(r)
    if args.shard is not None:
        done = []  # shard workers only write feature files; the parent writes the manifest

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

    for r in tqdm(done, desc="existing features", disable=not done, mininterval=5):
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
                save_npy_atomic(r["_feat"], feat.cpu().numpy().astype(np.float16))
            else:
                save_npy_atomic(r["_feat"], feat.cpu().numpy().astype(np.int16))
            finish(r, n_speech, r["_n_samples"] / sr)
        batch.clear()

    tag = f" [GPU shard {args.shard}]" if args.shard is not None else ""
    for r in tqdm(pending, desc=f"mimi {args.mode}{tag}", mininterval=5):
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
        key = f"{name}/{r['id']}"  # a second manifest name gives a different augmentation of the same clip
        if args.mode == "latents" and args.augment_prob and zlib.crc32(key.encode()) % 10000 < args.augment_prob * 10000:
            wav = augment(wav, sr, key)  # after the silence: real mics hear the room after you stop, too
        batch.append((r, wav))
        if len(batch) >= bs:
            flush()
    flush()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.shard is not None:
        print(f"[shard {args.shard}] done (skipped {skipped})")
        return
    order = {r["id"]: i for i, r in enumerate(rows)}
    out_rows.sort(key=lambda x: order.get(x["id"], 0))
    write_jsonl(args.out, out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
