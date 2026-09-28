"""Stage 4: train the talker (fusion + temporal transformer + depth transformer).

The thinker is frozen (use the merged model from stage 3). For every training
utterance the thinker is run teacher-forced on [prompt + text] to get the token
embeddings and hidden states that the talker would receive at inference time.

    python -m s2s.train.talker --config configs/default.yaml

Writes talker.pt, meta.json (which thinker/layers it was trained on), config.yaml
and a few generated samples/*.wav at every evaluation.
"""

from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from s2s.audio import save_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.config import save_config
from s2s.data.datasets import TalkerDataset, collate_talker, load_manifest, load_mixture, mixture_sampler
from s2s.models.speech_llm import talker_features
from s2s.models.talker import Talker, TalkerStream, apply_delay
from s2s.models.thinker import Thinker
from s2s.train.common import fmt, make_grad_scaler, optimizer_step
from s2s.utils import (autocast_ctx, cosine_lr, count_params, load_json, resolve_device, resolve_dtype,
                       save_json, set_seed)


def resolve_thinker_path(cfg) -> str:
    path = cfg.train_talker.thinker_dir
    if path and os.path.isdir(path):
        return path
    print(f"WARNING: merged thinker '{path}' not found; using base {cfg.thinker.model}. "
          "If you later change the thinker (LoRA), retrain the talker.")
    return cfg.thinker.model


def codec_card(cfg) -> int:
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(cfg.codec.model).codebook_size)


def features_for_batch(thinker: Thinker, prefix: list[int], text_ids: list[list[int]], layer_idx: list[int]):
    with autocast_ctx(thinker.device, thinker.dtype):
        toks, hids = talker_features(thinker, prefix, text_ids, layer_idx)
    return [t.float() for t in toks], [h.float() for h in hids]


def talker_loss(talker: Talker, thinker: Thinker, prefix, batch, layer_idx, device, dtype) -> dict:
    toks, hids = features_for_batch(thinker, prefix, batch["text_ids"], layer_idx)
    grids = [apply_delay(c.to(device), talker.delay, talker.card) for c in batch["codes"]]
    with autocast_ctx(device, dtype):
        return talker(toks, hids, grids)


@torch.no_grad()
def generate_samples(talker: Talker, thinker: Thinker, prefix, layer_idx, texts: list[str], out_dir: str,
                     codec, rt) -> None:
    talker.eval()
    os.makedirs(out_dir, exist_ok=True)
    for i, text in enumerate(texts):
        ids = thinker.prompts.ids(text)
        toks, hids = features_for_batch(thinker, prefix, [ids], layer_idx)
        stream = TalkerStream(talker, temperature=rt.talker_temperature, top_k=rt.talker_top_k)
        stream.push_text(talker.fuse(toks[0], hids[0]))
        stream.end_text()
        frames = []
        while stream.can_step():
            frames += stream.step()
        if frames:
            wav = codec.decode(torch.stack(frames, dim=1))
            save_audio(os.path.join(out_dir, f"sample_{i}.wav"), wav, codec.sample_rate)
            print(f"  sample_{i}.wav ({len(frames)} frames): {text}")
    talker.train()


@torch.no_grad()
def evaluate(talker, thinker, prefix, loader, layer_idx, device, dtype, max_batches) -> dict:
    talker.eval()
    losses, accs, n = 0.0, None, 0
    for bi, batch in enumerate(loader):
        if batch is None:
            continue
        if bi >= max_batches:
            break
        out = talker_loss(talker, thinker, prefix, batch, layer_idx, device, dtype)
        losses += float(out["loss"])
        accs = out["acc_per_codebook"] if accs is None else accs + out["acc_per_codebook"]
        n += 1
    talker.train()
    return {"loss": losses / max(1, n), "acc": (accs / max(1, n)).tolist() if accs is not None else []}


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--no-samples", action="store_true", help="skip audio sample generation at eval")
    args = p.parse_args()
    cfg = config_from_args(args)
    tc = cfg.train_talker
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)

    thinker_path = resolve_thinker_path(cfg)
    thinker = Thinker(thinker_path, device, dtype, attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    layer_idx = thinker.hidden_layer_indices(list(cfg.talker.hidden_layers))
    prefix = thinker.prompts.text_prompt_ids(cfg.thinker.system_prompt, cfg.talker.talker_prompt_user)
    K = int(cfg.codec.num_codebooks)

    if tc.init_from and os.path.exists(os.path.join(tc.init_from, "talker.pt")):
        talker = Talker.load(os.path.join(tc.init_from, "talker.pt"))
        meta = load_json(os.path.join(tc.init_from, "meta.json"))
        if meta.get("layer_idx") != layer_idx:
            raise ValueError(f"init_from talker used layers {meta.get('layer_idx')}, config gives {layer_idx}")
    else:
        talker = Talker.from_config(cfg.talker, thinker.hidden_size, len(layer_idx), K, codec_card(cfg))
    talker.to(device).train()
    print(f"talker params {count_params(talker) / 1e6:.1f}M, thinker layers {layer_idx}, thinker {thinker_path}")

    rows, weights = load_mixture(tc.train_manifests)
    valid_rows = load_manifest(tc.valid_manifest)
    train_ds = TalkerDataset(rows, thinker.prompts, K, tc.max_frames)
    valid_ds = TalkerDataset(valid_rows, thinker.prompts, K, tc.max_frames)
    weights = [w for r, w in zip(rows, weights) if r.get("codes")]
    total = int(tc.max_steps) * int(tc.batch_size) * int(tc.grad_accum)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, sampler=mixture_sampler(weights, total, cfg.seed),
                              collate_fn=collate_talker, num_workers=tc.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=tc.batch_size, collate_fn=collate_talker,
                              num_workers=tc.num_workers)
    sample_texts = [r["text"] for r in valid_rows[:3]]

    codec = None
    if not args.no_samples:
        from s2s.models.codec import MimiCodec

        codec = MimiCodec(cfg.codec.model, device, K)

    params = list(talker.parameters())
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.95))
    scaler = make_grad_scaler(enabled=device.type == "cuda" and dtype == torch.float16)
    meta = {"thinker": thinker_path, "layer_idx": layer_idx, "num_codebooks": K, "codec": cfg.codec.model}

    step, accum, running, t0 = 0, 0, {}, time.time()
    for batch in train_loader:
        if batch is None:
            continue
        out = talker_loss(talker, thinker, prefix, batch, layer_idx, device, dtype)
        scaler.scale(out["loss"] / tc.grad_accum).backward()
        running["loss"] = running.get("loss", 0.0) + float(out["loss"].detach()) / tc.grad_accum
        running["acc_cb1"] = running.get("acc_cb1", 0.0) + float(out["acc_per_codebook"][0]) / tc.grad_accum
        accum += 1
        if accum < tc.grad_accum:
            continue
        accum = 0
        scale = cosine_lr(step, tc.warmup_steps, tc.max_steps)
        for g in opt.param_groups:
            g["lr"] = tc.lr * scale
        gnorm = optimizer_step(opt, scaler, params, tc.max_grad_norm)
        step += 1
        if step % tc.log_every == 0:
            avg = {k: v / tc.log_every for k, v in running.items()}
            print(f"step {step} {fmt(avg)} gnorm {gnorm:.2f} lr_scale {scale:.3f} {(time.time() - t0) / step:.2f}s/step")
            running = {}
        if step % tc.eval_every == 0 or step == tc.max_steps:
            res = evaluate(talker, thinker, prefix, valid_loader, layer_idx, device, dtype, tc.eval_batches)
            accs = " ".join(f"{a * 100:.1f}" for a in res["acc"])
            print(f"[eval step {step}] loss {res['loss']:.4f} acc per codebook % [{accs}]")
            if codec is not None:
                generate_samples(talker, thinker, prefix, layer_idx, sample_texts,
                                 os.path.join(tc.output_dir, "samples", f"step_{step}"), codec, cfg.runtime)
        if step % tc.save_every == 0 or step == tc.max_steps:
            os.makedirs(tc.output_dir, exist_ok=True)
            talker.save(os.path.join(tc.output_dir, "talker.pt"))
            save_json(os.path.join(tc.output_dir, "meta.json"), meta)
            save_config(cfg, os.path.join(tc.output_dir, "config.yaml"))
            save_json(os.path.join(tc.output_dir, "state.json"), {"step": step})
            print(f"saved talker -> {tc.output_dir}")
        if step >= tc.max_steps:
            break


if __name__ == "__main__":
    main()
