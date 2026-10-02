"""Stages 2 and 3: train the speech adapter (+ CTC and end-of-turn heads) and the thinker LoRA.

Stage 2 (alignment):  LibriSpeech latents; tasks = transcribe + respond (distilled replies)
    python -m s2s.train.speech_llm --config configs/default.yaml

Stage 3 (tools): continue from stage 2 with spoken hotel requests mixed with alignment data
    python -m s2s.train.speech_llm --config configs/default.yaml \
        --set train_speech_llm.init_from=checkpoints/speech_llm_align \
              train_speech_llm.output_dir=checkpoints/speech_llm_tools \
              'train_speech_llm.train_manifests=[{path: data/manifests/hotel_train.jsonl, weight: 0.5}, {path: data/manifests/librispeech_train.jsonl, weight: 0.5}]' \
              train_speech_llm.valid_manifest=data/manifests/hotel_eval.jsonl train_speech_llm.max_steps=4000

Robustness (adapter only): retrain the speech input on accented / noisy speech while the
thinker stays exactly as merged in stage 3, so the talker (trained on that thinker) stays valid
    mkdir -p checkpoints/adapter_init && cp checkpoints/speech_llm_tools/adapter.pt checkpoints/adapter_init/
    python -m s2s.train.speech_llm --config configs/default.yaml \
        --set thinker.model=checkpoints/thinker_merged train_speech_llm.train_lora=false \
              train_speech_llm.init_from=checkpoints/adapter_init train_speech_llm.output_dir=checkpoints/speech_llm_robust ...

Another input encoder (features from extract_mimi --encoder ...): train a new adapter from scratch
    python -m s2s.train.speech_llm --config configs/default.yaml \
        --set adapter.encoder=parakeet:nvidia/parakeet-ctc-0.6b thinker.model=checkpoints/thinker_merged \
              train_speech_llm.train_lora=false train_speech_llm.output_dir=checkpoints/speech_llm_parakeet ...

Checkpoint directory layout: adapter.pt (stores the encoder spec), lora/ (PEFT, only when a LoRA is trained), config.yaml, state.json
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # forked DataLoader workers + tokenizer threads can deadlock

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from s2s.cli_common import base_parser, config_from_args
from s2s.config import save_config
from s2s.dist import arm_watchdog, barrier, cleanup, disarm_watchdog, init_distributed
from s2s.data.datasets import SpeechLLMDataset, collate_speech_llm, load_manifest, load_mixture, mixture_sampler
from s2s.models.adapter import SpeechAdapter
from s2s.models.speech_llm import assemble_inputs, greedy_generate, speech_llm_losses
from s2s.models.thinker import Thinker
from s2s.text import ctc_greedy_decode, wer
from s2s.train.common import fmt, make_grad_scaler, optimizer_step
from s2s.utils import autocast_ctx, cosine_lr, count_params, resolve_dtype, save_json, set_seed


class SpeechLLMTrainModule(torch.nn.Module):
    """Adapter + (LoRA) thinker as one module so DDP can synchronise their gradients."""

    def __init__(self, thinker: Thinker, adapter: SpeechAdapter, ctc_weight: float, eot_weight: float):
        super().__init__()
        self.adapter = adapter
        self.lm = thinker.model
        self._thinker = [thinker]  # plain list: not registered as a submodule twice
        self.ctc_weight, self.eot_weight = ctc_weight, eot_weight

    def forward(self, batch: dict) -> dict:
        return speech_llm_losses(self._thinker[0], self.adapter, batch, self.ctc_weight, self.eot_weight)


def build_models(cfg, device, dtype, init_from: str | None, train_lora: bool, latent_dim: int):
    lora_dir = os.path.join(init_from, "lora") if init_from else None
    if lora_dir and not os.path.isdir(lora_dir):
        lora_dir = None
    thinker = Thinker(cfg.thinker.model, device, dtype, lora_dir=lora_dir,
                      new_lora=dict(cfg.thinker.lora) if train_lora else None,
                      attn_implementation=cfg.thinker.attn_implementation)
    if init_from and os.path.exists(os.path.join(init_from, "adapter.pt")):
        adapter = SpeechAdapter.load(os.path.join(init_from, "adapter.pt"))
        print(f"loaded adapter from {init_from}")
    else:
        adapter = SpeechAdapter.from_config(cfg.adapter, latent_dim, thinker.hidden_size)
        adapter.init_scale(thinker.text_embedding_rms())
    adapter.to(device)
    if cfg.thinker.gradient_checkpointing:
        # also without LoRA: gradients still flow through the frozen thinker back to the adapter
        thinker.enable_gradient_checkpointing()
    return thinker, adapter


def save_checkpoint(out_dir: str, cfg, thinker: Thinker, adapter: SpeechAdapter, state: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    adapter.save(os.path.join(out_dir, "adapter.pt"))
    if hasattr(thinker.model, "peft_config"):
        thinker.model.save_pretrained(os.path.join(out_dir, "lora"))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    save_json(os.path.join(out_dir, "state.json"), state)
    print(f"saved checkpoint -> {out_dir}")


@torch.no_grad()
def evaluate(cfg, thinker: Thinker, adapter: SpeechAdapter, loader: DataLoader, max_batches: int,
             n_generate: int = 4) -> dict:
    tc = cfg.train_speech_llm
    adapter.eval()
    thinker.model.eval()
    lm, n = 0.0, 0
    refs, hyps = [], []
    eot_correct, eot_total = 0, 0
    samples = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        with autocast_ctx(thinker.device, thinker.dtype):
            out = speech_llm_losses(thinker, adapter, batch, tc.ctc_weight, tc.eot_weight)
        lm += float(out["lm"])
        n += 1
        ids = out["ctc_logits"].argmax(-1).cpu()
        for i, length in enumerate(batch["lengths"]):
            hyps.append(ctc_greedy_decode(ids[i, : int(length) * adapter.ctc_upsample].tolist()))
            refs.append(batch["texts"][i])
        eot = batch["eot"].to(thinker.device)
        valid = eot >= 0
        eot_correct += int(((out["eot_logits"] > 0).long() == eot)[valid].sum())
        eot_total += int(valid.sum())
        if len(samples) < n_generate:
            lat = batch["latents"][:1].to(thinker.device)
            with autocast_ctx(thinker.device, thinker.dtype):
                emb = adapter(lat)["embeds"]
                inputs, _, _ = assemble_inputs(thinker, adapter, emb, batch["lengths"][:1],
                                               batch["prefix"][:1], batch["suffix"][:1], None)
                gen = greedy_generate(thinker, inputs, max_new_tokens=60)
            samples.append({"task": batch["tasks"][0], "ref": batch["texts"][0],
                            "target": thinker.tokenizer.decode(batch["target"][0], skip_special_tokens=False),
                            "generated": thinker.tokenizer.decode(gen, skip_special_tokens=False)})
    adapter.train()
    thinker.model.train()
    return {"lm_loss": lm / max(1, n), "ctc_wer": wer(refs, hyps),
            "eot_acc": eot_correct / max(1, eot_total), "samples": samples}


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # live logs when piped (Kaggle/Colab `| tee`)
    cfg = config_from_args(args)
    tc = cfg.train_speech_llm
    rank, world, device = init_distributed(cfg.device)
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    set_seed(cfg.seed)  # same seed on every rank -> identical initial weights
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    grad_accum = int(tc.grad_accum)
    if world > 1 and grad_accum % world == 0:
        grad_accum //= world  # keep the effective batch size, finish in ~1/world of the time
    log(f"device {device} x {world} process(es), thinker dtype {dtype}, "
        f"effective batch {int(tc.batch_size) * grad_accum * world} (batch {tc.batch_size} x accum {grad_accum} x {world} GPU)")

    rows, weights = load_mixture(tc.train_manifests)
    valid_rows = load_manifest(tc.valid_manifest)
    latent_dim = np.load(rows[0]["latent"], mmap_mode="r").shape[1]
    thinker, adapter = build_models(cfg, device, dtype, tc.init_from, bool(tc.train_lora), latent_dim)
    enc_spec = adapter.hparams.get("encoder", "mimi")
    found = {r.get("encoder", "mimi") for r in rows + valid_rows}
    if found != {enc_spec} or adapter.hparams["latent_dim"] != latent_dim:
        raise ValueError(f"adapter expects {enc_spec} features ({adapter.hparams['latent_dim']}-d) but the manifests "
                         f"contain {sorted(found)} ({latent_dim}-d); set adapter.encoder / init_from to match")
    log(f"input encoder: {enc_spec} ({latent_dim}-d features)")
    thinker.model.train()
    adapter.train()
    module = SpeechLLMTrainModule(thinker, adapter, tc.ctc_weight, tc.eot_weight)
    model = DDP(module, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else module

    common = dict(prompts=thinker.prompts, system_prompt=cfg.thinker.system_prompt,
                  transcribe_instruction=cfg.thinker.transcribe_instruction, transcribe_prob=tc.transcribe_prob,
                  max_trailing_frames=tc.max_trailing_frames, max_frames=cfg.adapter.max_frames,
                  max_target_tokens=tc.max_target_tokens)
    train_ds = SpeechLLMDataset(rows, train=True, **common)
    valid_ds = SpeechLLMDataset(valid_rows, train=False, **common)
    total_samples = int(tc.max_steps) * int(tc.batch_size) * grad_accum
    # every rank draws its own stream of samples (different seed per rank)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size,
                              sampler=mixture_sampler(weights, total_samples, cfg.seed + 1000 * rank),
                              collate_fn=collate_speech_llm, num_workers=tc.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_speech_llm,
                              num_workers=tc.num_workers)

    lora_params = [p for p in thinker.model.parameters() if p.requires_grad]
    adapter_params = list(adapter.parameters())
    groups = [{"params": adapter_params, "lr": tc.adapter_lr, "base_lr": tc.adapter_lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": tc.lora_lr, "base_lr": tc.lora_lr})
    opt = torch.optim.AdamW(groups, weight_decay=tc.weight_decay)
    scaler = make_grad_scaler(enabled=device.type == "cuda" and dtype == torch.float16)
    log(f"adapter params {count_params(adapter) / 1e6:.1f}M, trainable thinker params {sum(p.numel() for p in lora_params) / 1e6:.1f}M")
    log(f"train rows {len(rows)}, valid rows {len(valid_rows)}")

    step, accum = 0, 0
    t0 = time.time()
    running: dict[str, float] = {}
    arm_watchdog()
    for batch in train_loader:
        last_micro = accum + 1 == grad_accum
        sync = model.no_sync() if (world > 1 and not last_micro) else contextlib.nullcontext()
        with sync:
            with autocast_ctx(device, dtype):
                out = model(batch)
            scaler.scale(out["loss"] / grad_accum).backward()
        for k in ("loss", "lm", "ctc", "eot"):
            running[k] = running.get(k, 0.0) + float(out[k].detach()) / grad_accum
        accum += 1
        if accum < grad_accum:
            continue
        accum = 0
        scale = cosine_lr(step, tc.warmup_steps, tc.max_steps)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * scale
        gnorm = optimizer_step(opt, scaler, adapter_params + lora_params, tc.max_grad_norm)
        step += 1
        arm_watchdog()
        if step % tc.log_every == 0:
            avg = {k: v / tc.log_every for k, v in running.items()}
            log(f"step {step} {fmt(avg)} gnorm {gnorm:.2f} lr_scale {scale:.3f} {(time.time() - t0) / step:.2f}s/step")
            running = {}
        if step % tc.eval_every == 0 or step == tc.max_steps:
            if main_proc:
                res = evaluate(cfg, thinker, adapter, valid_loader, tc.eval_batches)
                log(f"[eval step {step}] lm_loss {res['lm_loss']:.4f} ctc_wer {res['ctc_wer'] * 100:.2f}% eot_acc {res['eot_acc'] * 100:.1f}%")
                for s in res["samples"]:
                    log(f"  [{s['task']}] REF: {s['ref']}\n      TARGET: {s['target']!r}\n      GEN:    {s['generated']!r}")
            barrier()
        if step % tc.save_every == 0 or step == tc.max_steps:
            if main_proc:
                save_checkpoint(tc.output_dir, cfg, thinker, adapter, {"step": step, "world_size": world})
            barrier()
        if step >= tc.max_steps:
            break
    disarm_watchdog()
    cleanup()


if __name__ == "__main__":
    main()
