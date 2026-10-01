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

import argparse
import glob
import json
import os
import re
import sys

import torch
from tqdm import tqdm

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.thinker import Thinker
from s2s.text import sentence_case
from s2s.dist import resolve_num_workers, run_sharded, strip_gpu_args
from s2s.utils import read_jsonl, resolve_device, resolve_dtype, write_jsonl


def merge_shard_files(rows: list[dict], out: str) -> int:
    """Copy responses written by GPU shard workers (out.shard*.jsonl) into rows."""
    by_id = {r["id"]: r for r in rows}
    n = 0
    for path in sorted(glob.glob(out + ".shard*.jsonl")):
        for x in read_jsonl(path):
            r = by_id.get(x.get("id"))
            if r is not None and ("response" in x or x.get("tool_calls")) and not (r.get("response") or r.get("tool_calls")):
                for k in ("response", "tool_calls", "context", "tools"):
                    if k in x:
                        r[k] = x[k]
                n += 1
    return n


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
    p.add_argument("--hotel-context", action="store_true",
                   help="reply as the hotel agent: a random reservation + the hotel tools in the prompt (stored in the "
                        "row); valid tool calls become tool_calls rows, other tool output is dropped")
    p.add_argument("--gpus", type=int, default=0, help="worker processes, one per GPU (0 = all visible GPUs)")
    p.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args()
    cfg = config_from_args(args)

    rows = load_manifest(args.inp)
    merge_shard_files(rows, args.out)  # resume an interrupted multi-GPU run
    todo = [r for r in rows[: args.max_utts] if not (r.get("response") or r.get("tool_calls"))]
    n_workers = resolve_num_workers(args.gpus)
    if args.shard is None and n_workers > 1 and len(todo) > 1 and cfg.device != "cpu":
        print(f"{len(todo)} rows to distill on {n_workers} GPUs")
        run_sharded("s2s.prep.distill", strip_gpu_args(sys.argv[1:]), n_workers)
        merge_shard_files(rows, args.out)
        todo = []  # every row was attempted by a GPU worker; rows left empty stay transcription-only
    shard_file = None
    if args.shard is not None:
        todo = todo[args.shard:: args.num_shards]
        shard_file = open(f"{args.out}.shard{args.shard}.jsonl", "a", encoding="utf-8")
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    print(f"{len(todo)} rows to distill ({len(rows)} total)")
    if todo:
        thinker = Thinker(cfg.thinker.model, device, dtype, lora_dir=args.lora_dir, merge_lora=True,
                          attn_implementation=cfg.thinker.attn_implementation)
        tok = thinker.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = thinker.model.eval()
        base_system = cfg.thinker.system_prompt
        if args.hotel_context:
            import random as _random

            from s2s.data.hotel import HOTEL_TOOLS, make_reservation, reservation_context, validate_call
            rng = _random.Random(1234 + (args.shard or 0))
            for r in todo:
                r.setdefault("context", reservation_context(make_reservation(rng)))
                r["tools"] = "hotel"

        def system_for(r: dict) -> str:
            return Thinker.system_content(base_system, r.get("context")) if args.hotel_context else base_system
        todo.sort(key=lambda r: len(r["text"]))
        tag = f" [GPU shard {args.shard}]" if args.shard is not None else ""
        for i in tqdm(range(0, len(todo), args.batch_size), desc=f"distill{tag}", mininterval=5):
            chunk = todo[i: i + args.batch_size]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": system_for(r)}] + list(r.get("history") or [])
                + [{"role": "user", "content": sentence_case(r["text"])}],
                tools=HOTEL_TOOLS if args.hotel_context else None,
                add_generation_prompt=True, enable_thinking=False, tokenize=False) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for r, seq in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                text = tok.decode(seq, skip_special_tokens=True).strip()
                if not text or "<think>" in text:
                    continue
                result = {"id": r["id"]}
                if "<tool_call>" in text or text.lstrip().startswith("{"):
                    if not args.hotel_context:
                        continue
                    calls = []
                    for blob in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S) or [text]:
                        call = Thinker.parse_tool_call(blob)
                        if call is None or validate_call(call) is not None:
                            calls = []
                            break
                        calls.append(call)
                    if not calls:
                        continue  # malformed or invalid tool output: not a usable target
                    r["tool_calls"] = result["tool_calls"] = calls
                    r["response"] = result["response"] = ""
                else:
                    r["response"] = result["response"] = text
                if args.hotel_context:
                    result.update(context=r["context"], tools="hotel")
                if shard_file:
                    shard_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            if shard_file:
                shard_file.flush()
            elif (i // args.batch_size) % 50 == 0:
                write_jsonl(args.out, relativize(rows, args.out))
    if shard_file:
        shard_file.close()
        return  # the parent process merges shard files and writes the manifest
    write_jsonl(args.out, relativize(rows, args.out))
    for path in glob.glob(args.out + ".shard*.jsonl"):
        os.remove(path)
    print(f"{sum(1 for r in rows if r.get('response'))} rows with responses, "
          f"{sum(1 for r in rows if r.get('tool_calls'))} with tool calls -> {args.out}")


if __name__ == "__main__":
    main()
