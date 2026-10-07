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
from s2s.data.hotel_v2 import generate_examples_v2
from s2s.data.hotel_v3 import generate_examples_v3
from s2s.utils import write_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--eval", type=int, default=200)
    p.add_argument("--version", type=int, default=1, choices=[1, 2, 3],
                   help="2 = varied phrasings, reservation summary, small talk, out-of-scope, follow-up turns; "
                        "3 = property facts and a random tool subset in each row's own system prompt")
    p.add_argument("--prefix", default="hotel", help="manifest name prefix")
    args = p.parse_args()
    cfg = config_from_args(args)
    out = os.path.join(cfg.paths.data_dir, "manifests")
    gen = {1: generate_examples, 2: generate_examples_v2, 3: generate_examples_v3}[args.version]
    train = gen(args.train, seed=1)
    eval_rows = gen(args.eval, seed=2)
    for r in eval_rows:
        r["id"] = r["id"].replace("hotel", "hotel_eval", 1)
    print(write_jsonl(os.path.join(out, f"{args.prefix}_train_text.jsonl"), train), "train rows")
    print(write_jsonl(os.path.join(out, f"{args.prefix}_eval_text.jsonl"), eval_rows), "eval rows")


if __name__ == "__main__":
    main()
