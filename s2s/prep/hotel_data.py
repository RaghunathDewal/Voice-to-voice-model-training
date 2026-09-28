"""Generate hotel tool-call examples (text only).

    python -m s2s.prep.hotel_data --train 4000 --eval 200

Writes manifests/hotel_train_text.jsonl and manifests/hotel_eval_text.jsonl.
The eval file is used as-is for experiment 2 (text-only tool accuracy). To
train/evaluate from *speech*, synthesise them with s2s.prep.synth_kokoro and
extract Mimi latents with s2s.prep.extract_mimi.
"""

from __future__ import annotations

import os

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import generate_examples
from s2s.utils import write_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--eval", type=int, default=200)
    args = p.parse_args()
    cfg = config_from_args(args)
    out = os.path.join(cfg.paths.data_dir, "manifests")
    train = generate_examples(args.train, seed=1)
    eval_rows = generate_examples(args.eval, seed=2)
    for r in eval_rows:
        r["id"] = r["id"].replace("hotel_", "hotel_eval_")
    print(write_jsonl(os.path.join(out, "hotel_train_text.jsonl"), train), "train rows")
    print(write_jsonl(os.path.join(out, "hotel_eval_text.jsonl"), eval_rows), "eval rows")


if __name__ == "__main__":
    main()
