"""Thinker LoRA from text only: hotel tool use, post-tool replies, reservation answers (no audio).

Fits a single free Colab / Kaggle T4. Data from s2s.prep.thinker_sft_data.

    python -m s2s.train.thinker_text --config configs/small.yaml \
        --base checkpoints/thinker_merged --out checkpoints/thinker_text \
        --steps 1500 --batch 4 --accum 4 [--upload-repo user/s2s-checkpoints] [--resume]

    # then make a standalone model (the new thinker for the adapter and talker):
    python -m s2s.prep.merge_lora --config configs/small.yaml --set thinker.model=checkpoints/thinker_merged \
        --speech-llm-dir checkpoints/thinker_text --out checkpoints/thinker_merged_v2

Every --eval-every steps: loss on held-out hotel conversations and exact tool-call / answer
accuracy on --eval-rows hotel_v2 rows (seed 2, the same rows as the speech eval), from text.
With --upload-repo each saved LoRA is also pushed to the Hub, so a dropped Colab session
loses at most --save-every steps (--resume continues from <out>/lora).
"""

from __future__ import annotations

import collections
import os
import random
import sys
import threading
import time

import torch

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import score_example, tools_by_name
from s2s.data.hotel_v2 import generate_examples_v2
from s2s.eval.text_tools import extract_calls
from s2s.models.thinker import Thinker
from s2s.train.common import make_grad_scaler, optimizer_step
from s2s.utils import (autocast_ctx, cosine_lr, load_json, read_jsonl, resolve_device, resolve_dtype, save_json,
                       set_seed)


def encode_rows(prompts, rows: list[dict], max_len: int) -> list[tuple[list[int], list[int]]]:
    out, skipped = [], 0
    for r in rows:
        try:
            ids, labels = prompts.conversation_ids(r["messages"], r.get("tools"))
        except (ValueError, RuntimeError, TypeError):
            skipped += 1
            continue
        if len(ids) > max_len:
            skipped += 1
            continue
        out.append((ids, labels))
    print(f"encoded {len(out)} conversations (skipped {skipped}: too long or unusable)")
    return out


def collate(samples, pad_id: int, device):
    n = max(len(s[0]) for s in samples)
    ids = torch.full((len(samples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), n), -100, dtype=torch.long)
    mask = torch.zeros((len(samples), n), dtype=torch.long)
    for i, (x, y) in enumerate(samples):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


@torch.no_grad()
def valid_loss(thinker: Thinker, data, batch: int) -> float:
    total, n = 0.0, 0
    for i in range(0, len(data), batch):
        ids, labels, mask = collate(data[i:i + batch], thinker.pad_id, thinker.device)
        with autocast_ctx(thinker.device, thinker.dtype):
            total += float(thinker.model(input_ids=ids, attention_mask=mask, labels=labels).loss)
        n += 1
    return total / max(1, n)


@torch.no_grad()
def hotel_accuracy(thinker: Thinker, system_prompt: str, rows: list[dict], show: int = 4) -> float:
    per = collections.defaultdict(lambda: [0, 0])
    misses = []
    for r in rows:
        system = Thinker.system_content(system_prompt, r.get("context"))
        ids = thinker.prompts.text_prompt_ids(system, r["text"], tools_by_name(r.get("tools")), r.get("history"))
        with autocast_ctx(thinker.device, thinker.dtype):
            gen = thinker.model.generate(torch.tensor([ids], device=thinker.device), max_new_tokens=100, use_cache=True,
                                         do_sample=False, eos_token_id=sorted(thinker.eos_ids),
                                         pad_token_id=thinker.pad_id)
        out = thinker.tokenizer.decode(gen[0, len(ids):], skip_special_tokens=False)
        ok = score_example(r, extract_calls(out), out)
        name = r["tool_calls"][0]["name"] if r.get("tool_calls") else "(answer from context)"
        per[name][0] += int(ok)
        per[name][1] += 1
        if not ok and len(misses) < show:
            misses.append(f"    MISS: {r['text']!r}\n      gold {r.get('tool_calls') or r.get('reply')}\n      got  {out!r}")
    acc = sum(v[0] for v in per.values()) / max(1, sum(v[1] for v in per.values()))
    print(f"  hotel accuracy (text) {acc * 100:.1f}% on {len(rows)} rows: "
          + ", ".join(f"{k} {c}/{n}" for k, (c, n) in sorted(per.items())))
    for m in misses:
        print(m)
    return acc


def upload_async(repo: str, folder: str, path_in_repo: str) -> None:
    def run():
        try:
            from huggingface_hub import HfApi

            HfApi().upload_folder(repo_id=repo, folder_path=folder, path_in_repo=path_in_repo,
                                  commit_message=f"thinker_text checkpoint {os.path.basename(folder)}")
            print(f"[upload] {folder} -> {repo}/{path_in_repo}")
        except Exception as e:  # noqa: BLE001 - never stop training because of an upload
            print(f"[upload] failed: {e}")

    threading.Thread(target=run, daemon=True).start()


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--base", default="checkpoints/thinker_merged", help="thinker to fine-tune (HF id or local dir)")
    p.add_argument("--train", default=None, help="default: <data_dir>/manifests/thinker_sft_train.jsonl")
    p.add_argument("--valid", default=None, help="default: <data_dir>/manifests/thinker_sft_valid.jsonl")
    p.add_argument("--out", default="checkpoints/thinker_text")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--max-len", type=int, default=1536)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-rows", type=int, default=150, help="hotel_v2 seed-2 rows for the accuracy check")
    p.add_argument("--upload-repo", default=None, help="push each saved LoRA to this HF model repo")
    p.add_argument("--resume", action="store_true", help="continue from <out>/lora and <out>/state.json")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when piped (Colab/Kaggle `| tee`)
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    manifests = os.path.join(cfg.paths.data_dir, "manifests")
    train_path = args.train or os.path.join(manifests, "thinker_sft_train.jsonl")
    valid_path = args.valid or os.path.join(manifests, "thinker_sft_valid.jsonl")

    lora_dir = os.path.join(args.out, "lora")
    start = 0
    if args.resume and os.path.isdir(lora_dir):
        start = int(load_json(os.path.join(args.out, "state.json")).get("step", 0))
        print(f"resuming from {lora_dir} at step {start}")
    else:
        lora_dir = None
    thinker = Thinker(args.base, device, dtype, lora_dir=lora_dir, new_lora=dict(cfg.thinker.lora),
                      attn_implementation=cfg.thinker.attn_implementation)
    if cfg.thinker.gradient_checkpointing:
        thinker.enable_gradient_checkpointing()
    params = [q for q in thinker.model.parameters() if q.requires_grad]
    print(f"thinker {args.base} ({dtype}) on {device}, trainable LoRA params {sum(q.numel() for q in params) / 1e6:.1f}M")

    train = encode_rows(thinker.prompts, read_jsonl(train_path), args.max_len)
    valid = encode_rows(thinker.prompts, read_jsonl(valid_path)[:200], args.max_len)
    eval_rows = generate_examples_v2(args.eval_rows, seed=2)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = make_grad_scaler(enabled=device.type == "cuda" and dtype == torch.float16)
    rng = random.Random(cfg.seed + start)

    def save(step: int) -> None:
        os.makedirs(args.out, exist_ok=True)
        thinker.model.save_pretrained(os.path.join(args.out, "lora"))
        save_json(os.path.join(args.out, "state.json"), {"step": step, "base": args.base})
        print(f"saved LoRA at step {step} -> {args.out}/lora")
        if args.upload_repo:
            upload_async(args.upload_repo, args.out, "checkpoints/" + os.path.basename(os.path.normpath(args.out)))

    if start == 0:
        thinker.model.eval()
        print(f"[eval step 0] valid loss {valid_loss(thinker, valid, args.batch):.4f}")
        hotel_accuracy(thinker, cfg.thinker.system_prompt, eval_rows[:50])
    thinker.model.train()
    t0, running = time.time(), 0.0
    for step in range(start, args.steps):
        for _ in range(args.accum):
            ids, labels, mask = collate(rng.sample(train, args.batch), thinker.pad_id, device)
            with autocast_ctx(device, dtype):
                loss = thinker.model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss / args.accum).backward()
            running += float(loss.detach()) / args.accum
        scale = cosine_lr(step, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = args.lr * scale
        gnorm = optimizer_step(opt, scaler, params, 1.0)
        done = step + 1
        if done % 25 == 0:
            rate = (time.time() - t0) / (done - start)
            print(f"step {done}/{args.steps} loss {running / 25:.4f} gnorm {gnorm:.2f} lr {args.lr * scale:.2e} "
                  f"{rate:.2f}s/step, ~{rate * (args.steps - done) / 60:.0f} min left", flush=True)
            running = 0.0
        if done % args.eval_every == 0 or done == args.steps:
            thinker.model.eval()
            print(f"[eval step {done}] valid loss {valid_loss(thinker, valid, args.batch):.4f}")
            hotel_accuracy(thinker, cfg.thinker.system_prompt, eval_rows)
            thinker.model.train()
        if done % args.save_every == 0 or done == args.steps:
            save(done)


if __name__ == "__main__":
    main()
