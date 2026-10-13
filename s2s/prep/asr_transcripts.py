"""Add the listener's own transcript to manifest rows (hybrid input): field "asr" = Nemotron's greedy RNN-T
transcript, decoded from the stored encoder features (no audio needed). See s2s/models/nemotron_text.py.

    python -m s2s.prep.asr_transcripts --in data/manifests/slurp_train_q4.jsonl data/manifests/hotel5_eval_q4.jsonl \\
        --suffix _asr --max-rows 20000

Writes <name><suffix>.jsonl next to each input (only the rows kept: --max-rows takes a random subset, the same
every time; rows without features are dropped). Rows are decoded longest first, in batches.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from s2s.data.datasets import load_manifest
from s2s.models.nemotron_text import NemotronTranscriber


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", nargs="+", required=True)
    p.add_argument("--suffix", default="_asr")
    p.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    p.add_argument("--max-rows", type=int, default=0, help="per manifest, a random subset (0 = all)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    tr = NemotronTranscriber.load(args.model, torch.device(args.device))
    for path in args.inp:
        out_path = os.path.splitext(path)[0] + args.suffix + ".jsonl"
        if os.path.exists(out_path):
            print(f"{out_path} exists: skipped", flush=True)
            continue
        rows = [r for r in load_manifest(path) if r.get("latent") and os.path.exists(r["latent"])]
        if args.max_rows and len(rows) > args.max_rows:
            rows = random.Random(args.seed).sample(rows, args.max_rows)
        frames = [int(np.load(r["latent"], mmap_mode="r").shape[0]) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: -frames[i])
        t0 = time.time()
        for b in range(0, len(order), args.batch):
            idx = order[b: b + args.batch]
            feats = [torch.from_numpy(np.load(rows[i]["latent"]).astype(np.float32)) for i in idx]
            for i, text in zip(idx, tr(feats)):
                rows[i]["asr"] = text
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        sample = next((r for r in rows if r.get("text")), None)
        print(f"{out_path}: {len(rows)} rows in {time.time() - t0:.0f}s"
              + (f" | e.g. text {sample['text'][:60]!r} -> asr {sample['asr'][:60]!r}" if sample else ""), flush=True)


if __name__ == "__main__":
    main()
