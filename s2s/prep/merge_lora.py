"""Merge the trained LoRA into the base thinker and save a standalone model.

Run this once the thinker is final (after the tool stage). The talker is
trained on hidden states of THIS merged model, so do not change the thinker
after training the talker.

    python -m s2s.prep.merge_lora --speech-llm-dir checkpoints/speech_llm_tools --out checkpoints/thinker_merged
"""

from __future__ import annotations

import os

from s2s.cli_common import base_parser, config_from_args
from s2s.models.thinker import Thinker
from s2s.utils import resolve_device, resolve_dtype


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--speech-llm-dir", required=True, help="directory with lora/ (and adapter.pt)")
    p.add_argument("--out", default="checkpoints/thinker_merged")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    lora = os.path.join(args.speech_llm_dir, "lora")
    thinker = Thinker(cfg.thinker.model, device, dtype, lora_dir=lora if os.path.isdir(lora) else None,
                      merge_lora=True, attn_implementation=cfg.thinker.attn_implementation)
    os.makedirs(args.out, exist_ok=True)
    thinker.model.save_pretrained(args.out)
    thinker.tokenizer.save_pretrained(args.out)
    print(f"merged thinker saved to {args.out}")


if __name__ == "__main__":
    main()
