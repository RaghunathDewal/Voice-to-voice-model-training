"""Behaviour-alignment targets: ask the (text-only) thinker to reply to each transcript.

The adapter is then trained so that *speech* input produces the same reply the
thinker gives for the *text* input. Replies use the voice-assistant system
prompt so they are short and speakable (these are also good talker texts).

    python -m s2s.prep.distill --in data/manifests/librispeech_train.jsonl \
        --out data/manifests/librispeech_train.jsonl --max-utts 30000

Rows that already have a `response` are kept (restartable). Rows beyond
--max-utts keep no response and are used for the transcription task only.
"""

from __future__ import annotations

import os

import torch
from tqdm import tqdm

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.thinker import Thinker
from s2s.text import sentence_case
from s2s.utils import resolve_device, resolve_dtype, write_jsonl


def relativize(rows: list[dict], out_path: str) -> list[dict]:
    base = os.path.dirname(os.path.abspath(out_path))
    out = []
    for r in rows:
        r = dict(r)
        for key in ("latent", "codes", "audio"):
            if r.get(key) and os.path.isabs(r[key]):
                r[key] = os.path.relpath(r[key], base)
        out.append(r)
    return out


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-utts", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--lora-dir", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)

    rows = load_manifest(args.inp)
    todo = [r for r in rows[: args.max_utts] if not r.get("response")]
    print(f"{len(todo)} rows to distill ({len(rows)} total)")
    if todo:
        thinker = Thinker(cfg.thinker.model, device, dtype, lora_dir=args.lora_dir, merge_lora=True,
                          attn_implementation=cfg.thinker.attn_implementation)
        tok = thinker.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = thinker.model.eval()
        system = cfg.thinker.system_prompt
        todo.sort(key=lambda r: len(r["text"]))
        for i in tqdm(range(0, len(todo), args.batch_size), desc="distill"):
            chunk = todo[i: i + args.batch_size]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": sentence_case(r["text"])}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for r, seq in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                text = tok.decode(seq, skip_special_tokens=True).strip()
                if text and "<think>" not in text:
                    r["response"] = text
            if (i // args.batch_size) % 50 == 0:
                write_jsonl(args.out, relativize(rows, args.out))
    write_jsonl(args.out, relativize(rows, args.out))
    print(f"{sum(1 for r in rows if r.get('response'))} rows with responses -> {args.out}")


if __name__ == "__main__":
    main()
