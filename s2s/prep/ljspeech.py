"""Download LJSpeech 1.1 (single female speaker, ~24 h, public domain) for talker training.

    python -m s2s.prep.ljspeech

Outputs manifests/talker_train_raw.jsonl and manifests/talker_valid_raw.jsonl
with the *written* text (digits, abbreviations) so the talker learns to read
text the way the thinker writes it. Then run:

    python -m s2s.prep.extract_mimi --mode codes --in data/manifests/talker_train_raw.jsonl --out data/manifests/talker_train.jsonl
"""

from __future__ import annotations

import csv
import os

from s2s.cli_common import base_parser, config_from_args, download, extract_tar
from s2s.utils import write_jsonl

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--url", default=URL)
    p.add_argument("--valid", type=int, default=100, help="utterances held out for validation")
    p.add_argument("--max-utts", type=int, default=0)
    args = p.parse_args()
    cfg = config_from_args(args)
    root = os.path.join(cfg.paths.data_dir, "ljspeech")
    manifests = os.path.join(cfg.paths.data_dir, "manifests")
    corpus = os.path.join(root, "LJSpeech-1.1")
    if not os.path.isdir(corpus):
        archive = os.path.join(root, "LJSpeech-1.1.tar.bz2")
        download(args.url, archive)
        extract_tar(archive, root, corpus)
        os.remove(archive)

    rows = []
    with open(os.path.join(corpus, "metadata.csv"), encoding="utf-8") as f:
        for utt, raw, _normalized in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            audio = os.path.join(corpus, "wavs", utt + ".wav")
            rows.append({"id": utt, "audio": os.path.relpath(audio, manifests), "text": raw.strip()})
    if args.max_utts:
        rows = rows[: args.max_utts]
    valid, train = rows[-args.valid:], rows[: -args.valid]
    print(write_jsonl(os.path.join(manifests, "talker_train_raw.jsonl"), train), "train rows")
    print(write_jsonl(os.path.join(manifests, "talker_valid_raw.jsonl"), valid), "valid rows")


if __name__ == "__main__":
    main()
