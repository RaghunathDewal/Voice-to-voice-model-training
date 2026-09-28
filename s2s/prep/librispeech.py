"""Download LibriSpeech from OpenSLR and write raw manifests (audio path + transcript).

    python -m s2s.prep.librispeech --subsets dev-clean test-clean train-clean-100

Outputs (under paths.data_dir):
    manifests/librispeech_train_raw.jsonl   (all train-* subsets requested)
    manifests/librispeech_dev_raw.jsonl     (dev-clean)
    manifests/librispeech_test_raw.jsonl    (test-clean)
"""

from __future__ import annotations

import glob
import os

import soundfile as sf

from s2s.cli_common import base_parser, config_from_args, download, extract_tar
from s2s.utils import write_jsonl

SPLIT_OF = {"dev-clean": "dev", "dev-other": "dev", "test-clean": "test", "test-other": "test"}


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--subsets", nargs="+", default=["dev-clean", "test-clean", "train-clean-100"])
    p.add_argument("--mirror", default="https://www.openslr.org/resources/12",
                   help="alternatives: https://us.openslr.org/resources/12 , https://openslr.elda.org/resources/12")
    p.add_argument("--max-utts", type=int, default=0, help="cap utterances per split (0 = all)")
    p.add_argument("--keep-archives", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    root = os.path.join(cfg.paths.data_dir, "librispeech")
    manifests = os.path.join(cfg.paths.data_dir, "manifests")

    by_split: dict[str, list[dict]] = {}
    for subset in args.subsets:
        archive = os.path.join(root, f"{subset}.tar.gz")
        subset_dir = os.path.join(root, "LibriSpeech", subset)
        if not os.path.isdir(subset_dir):
            download(f"{args.mirror}/{subset}.tar.gz", archive)
            extract_tar(archive, root, subset_dir)
            if not args.keep_archives and os.path.exists(archive):
                os.remove(archive)
        split = SPLIT_OF.get(subset, "train")
        rows = by_split.setdefault(split, [])
        for trans in sorted(glob.glob(os.path.join(subset_dir, "*", "*", "*.trans.txt"))):
            folder = os.path.dirname(trans)
            with open(trans, encoding="utf-8") as f:
                for line in f:
                    utt, text = line.strip().split(" ", 1)
                    audio = os.path.join(folder, utt + ".flac")
                    rows.append({"id": utt, "audio": os.path.relpath(audio, manifests), "text": text})

    for split, rows in by_split.items():
        if args.max_utts:
            rows = rows[: args.max_utts]
        for r in rows[:1]:
            info = sf.info(os.path.join(manifests, r["audio"]))
            print(f"[{split}] sample {r['id']}: {info.samplerate} Hz, {info.duration:.1f}s")
        out = os.path.join(manifests, f"librispeech_{split}_raw.jsonl")
        n = write_jsonl(out, rows)
        print(f"[{split}] {n} utterances -> {out}")


if __name__ == "__main__":
    main()
