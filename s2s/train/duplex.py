"""Train the full-duplex experiment model (s2s/models/duplex.py) on two-channel hotel calls.

    python -m s2s.train.duplex --config configs/duplex.yaml                     # 1 GPU
    torchrun --standalone --nproc_per_node=2 -m s2s.train.duplex --config configs/duplex.yaml

Data: s2s.prep.duplex_data (script -> audio -> features). Writes duplex.pt (LoRA + new layers, small),
config.yaml and state.json to train_duplex.output_dir; a later run with init_from continues from it.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from s2s.cli_common import base_parser, config_from_args
from s2s.config import save_config
from s2s.data.datasets import load_manifest
from s2s.data.duplex import PAD_TOKEN, DuplexDataset, collate_duplex
from s2s.dist import arm_watchdog, barrier, cleanup, disarm_watchdog, init_distributed, mean_across_ranks
from s2s.models.duplex import DuplexModel
from s2s.train.common import apply_time_budget, fmt, make_grad_scaler, optimizer_step
from s2s.utils import autocast_ctx, cosine_lr, count_params, resolve_dtype, save_json, set_seed

STAT_KEYS = ("loss", "text", "audio", "tool", "text_acc", "pad_acc", "c0_acc", "c_acc")


def build_model(cfg, device, dtype):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tc = cfg.train_duplex
    tok = AutoTokenizer.from_pretrained(tc.llm)
    llm = AutoModelForCausalLM.from_pretrained(tc.llm, dtype=dtype, attn_implementation="sdpa")
    for p in llm.parameters():
        p.requires_grad_(False)
    from s2s.models.thinker import ignore_incompatible_torchao

    ignore_incompatible_torchao()  # Kaggle ships torchao 0.10, which recent PEFT refuses
    lc = tc.lora
    llm = get_peft_model(llm, LoraConfig(r=int(lc.r), lora_alpha=int(lc.alpha), lora_dropout=float(lc.dropout),
                                         target_modules=list(lc.target_modules), task_type="CAUSAL_LM"))
    if tc.gradient_checkpointing:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.enable_input_require_grads()
    model = DuplexModel(llm, feat_dim=int(tc.feat_dim), pad_id=tok.convert_tokens_to_ids(PAD_TOKEN),
                        num_codebooks=int(cfg.codec.num_codebooks), card=int(tc.codebook_size),
                        depth_d_model=int(tc.depth_d_model), depth_layers=int(tc.depth_layers),
                        depth_heads=int(tc.depth_heads), delay=int(tc.acoustic_delay),
                        pad_weight=float(tc.pad_weight), semantic_weight=float(tc.semantic_weight))
    model.to(device)
    for n, p in model.named_parameters():  # trainable weights in fp32 (fp16 GradScaler on T4)
        if p.requires_grad:
            p.data = p.data.float()
    if tc.get("init_from") and os.path.exists(os.path.join(tc.init_from, "duplex.pt")):
        sd = torch.load(os.path.join(tc.init_from, "duplex.pt"), map_location="cpu", weights_only=False)["state_dict"]
        missing = model.load_state_dict(sd, strict=False)
        print(f"continued from {tc.init_from} ({len(sd)} tensors, {len(missing.unexpected_keys)} unexpected)")
    return model, tok


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = config_from_args(args)
    tc = cfg.train_duplex
    rank, world, device = init_distributed(cfg.device)
    arm_watchdog()
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    set_seed(cfg.seed + rank)
    dtype = resolve_dtype(tc.get("dtype", "auto"), device)

    model, tok = build_model(cfg, device, dtype)
    log(f"duplex model: {count_params(model, trainable_only=True) / 1e6:.1f}M trainable of "
        f"{count_params(model) / 1e6:.0f}M, {world} process(es), dtype {dtype}")
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else model

    def dataset(path):
        return DuplexDataset(load_manifest(path), tok, os.path.dirname(os.path.abspath(path)), int(tc.max_frames))

    train_ds, valid_ds = dataset(tc.train_manifest), dataset(tc.valid_manifest)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, seed=cfg.seed) if world > 1 else \
        RandomSampler(train_ds, replacement=True, num_samples=10**8)
    train_loader = DataLoader(train_ds, batch_size=int(tc.batch_size), sampler=sampler, collate_fn=collate_duplex,
                              num_workers=int(tc.num_workers), drop_last=True, persistent_workers=int(tc.num_workers) > 0)
    valid_loader = DataLoader(valid_ds, batch_size=int(tc.batch_size), collate_fn=collate_duplex, num_workers=0)
    log(f"train calls {len(train_ds)}, valid calls {len(valid_ds)}")

    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    new = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": lora, "lr": float(tc.lora_lr), "base_lr": float(tc.lora_lr)},
                             {"params": new, "lr": float(tc.lr), "base_lr": float(tc.lr)}],
                            weight_decay=float(tc.weight_decay))
    params = lora + new
    scaler = make_grad_scaler(enabled=dtype == torch.float16)
    accum = int(tc.grad_accum)

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        tot, n = {}, 0
        for bi, batch in enumerate(valid_loader):
            if bi >= int(tc.eval_batches):
                break
            with autocast_ctx(device, dtype):
                out = model(batch)
            for k in STAT_KEYS:
                tot[k] = tot.get(k, 0.0) + float(out[k])
            n += 1
        model.train()
        return {k: v / max(1, n) for k, v in tot.items()}

    def save(step: int) -> None:
        os.makedirs(tc.output_dir, exist_ok=True)
        model.save(os.path.join(tc.output_dir, "duplex.pt"), {"llm": tc.llm, "lora": dict(tc.lora)})
        save_config(cfg, os.path.join(tc.output_dir, "config.yaml"))
        save_json(os.path.join(tc.output_dir, "state.json"), {"step": step, "world_size": world})
        log(f"saved {tc.output_dir}")

    model.train()
    step, micro, running, t0 = 0, 0, {}, time.time()
    epoch = 0
    while step < int(tc.max_steps):
        if world > 1:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            sync = (micro + 1) % accum == 0
            ctx = ddp.no_sync() if world > 1 and not sync else contextlib.nullcontext()
            with ctx, autocast_ctx(device, dtype):
                out = ddp(batch)
                scaler.scale(out["loss"] / accum).backward()
            for k in STAT_KEYS:
                running[k] = running.get(k, 0.0) + float(out[k].detach()) / accum
            micro += 1
            if not sync:
                continue
            scale = cosine_lr(step, int(tc.warmup_steps), int(tc.max_steps))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            gnorm = optimizer_step(opt, scaler, params, float(tc.max_grad_norm))
            step += 1
            arm_watchdog()
            apply_time_budget(tc, step, t0, device, log)
            if step % int(tc.log_every) == 0:
                avg = {k: mean_across_ranks(v / int(tc.log_every), device) for k, v in running.items()}
                log(f"step {step} {fmt(avg)} gnorm {gnorm:.2f} lr_scale {scale:.3f} {(time.time() - t0) / step:.2f}s/step")
                running = {}
            if step % int(tc.eval_every) == 0 or step >= int(tc.max_steps):
                if main_proc:
                    r = evaluate()
                    log(f"[eval step {step}] {fmt(r)}")
                barrier()
            if step % int(tc.save_every) == 0 or step >= int(tc.max_steps):
                if main_proc:
                    save(step)
                barrier()
            if step >= int(tc.max_steps):
                break
    with contextlib.suppress(Exception):
        disarm_watchdog()
    cleanup()


if __name__ == "__main__":
    main()
