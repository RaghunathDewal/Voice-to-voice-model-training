"""Stage A of adapter v3: align the speech adapter to the thinker's word vectors WITHOUT running the thinker.

The adapter's output vectors are scored against the thinker's own (frozen) input embedding table - the
same table its LM head uses (tie_word_embeddings) - with a word-level CTC loss: every speech frame either
is "blank" or lands next to the embedding of the token being said. Only that table (one matrix) is loaded,
never the 36 transformer layers, so a step costs about as much as the adapter itself: hundreds of thousands
of utterances fit in one Kaggle session. Plus the character CTC head (confidence) and the end-of-turn head
(trained to fire `eot_early_frames` before the speech ends).

    python -m s2s.train.align_adapter --config configs/qwen4b_v3.yaml
    torchrun --standalone --nproc_per_node=2 -m s2s.train.align_adapter --config configs/qwen4b_v3.yaml

Stage B (s2s.train.speech_llm with kd_weight > 0) then trains the same adapter with the thinker in the loop.
Writes adapter.pt (with the token head and its vocabulary), config.yaml, state.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from s2s.cli_common import base_parser, config_from_args
from s2s.config import save_config
from s2s.data.datasets import SpeechLLMDataset, collate_speech_llm, load_manifest, load_mixture, mixture_sampler
from s2s.dist import arm_watchdog, barrier, cleanup, disarm_watchdog, init_distributed
from s2s.eval.spoken_wer import spoken_wer
from s2s.models.adapter import SpeechAdapter
from s2s.models.speech_llm import token_ctc_from_logits
from s2s.models.thinker import PromptBuilder
from s2s.text import ctc_greedy_decode, sentence_case
from s2s.train.common import apply_time_budget, fmt, make_grad_scaler, optimizer_step
from s2s.utils import cosine_lr, count_params, load_matching, save_json, set_seed

OUTPUT_PARAMS = ("out_proj.weight", "out_proj.bias", "out_norm.weight", "out_scale", "speech_start", "speech_end")


def load_embedding_table(model: str) -> torch.Tensor:
    """The thinker's input embedding matrix only (no transformer layers), from a local dir or the Hub."""
    from safetensors import safe_open

    def local_or_hub(name: str) -> str | None:
        if os.path.isdir(model):
            path = os.path.join(model, name)
            return path if os.path.exists(path) else None
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(model, name)
        except Exception:  # noqa: BLE001 - not in this repo
            return None

    index = local_or_hub("model.safetensors.index.json")
    if index:
        weight_map = json.load(open(index))["weight_map"]
        key = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
        shard = local_or_hub(weight_map[key])
    else:
        shard = local_or_hub("model.safetensors")
        with safe_open(shard, "pt") as f:
            key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
    with safe_open(shard, "pt") as f:
        return f.get_tensor(key)


class AlignModule(torch.nn.Module):
    """Adapter + its token head scored against the (frozen, not trained) thinker embedding table, in one
    forward so DDP sees every trained parameter used inside forward()."""

    def __init__(self, adapter: SpeechAdapter, table: torch.Tensor):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("table", table, persistent=False)

    def forward(self, latents: torch.Tensor) -> dict:
        out = self.adapter(latents)
        out["tok_logits"] = self.adapter.token_logits(out["embeds"], self.table)
        return out


def build_token_vocab(rows: list[dict], prompts: PromptBuilder) -> list[int]:
    ids = set()
    for r in rows:
        if r.get("text") and not r.get("unclear"):
            ids.update(prompts.ids(sentence_case(r["text"])))
    return sorted(ids)


def token_greedy_text(adapter: SpeechAdapter, logits: torch.Tensor, length: int, tokenizer) -> str:
    best = logits[:length].argmax(-1).tolist()
    out, prev = [], None
    for i in best:
        if i != prev and i != 0:
            out.append(int(adapter.tok_vocab[i - 1]))
        prev = i
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = config_from_args(args)
    tc = cfg.train_align
    rank, world, device = init_distributed(cfg.device)
    arm_watchdog()
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    set_seed(cfg.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.thinker.model)
    prompts = PromptBuilder(tokenizer)
    log("loading manifests ...")
    rows, weights = load_mixture(tc.train_manifests)
    valid_rows = load_manifest(tc.valid_manifest)
    latent_dim = np.load(rows[0]["latent"], mmap_mode="r").shape[1]
    table = load_embedding_table(cfg.thinker.model).to(device)
    vocab = build_token_vocab(rows + valid_rows, prompts)
    log(f"thinker embedding table {tuple(table.shape)}, token head over {len(vocab)} thinker tokens")

    acfg = dict(cfg.adapter)
    acfg["token_vocab_size"] = len(vocab)
    adapter = SpeechAdapter.from_config(type(cfg.adapter)(acfg), latent_dim, table.shape[1])
    adapter.set_token_vocab(vocab)
    rms = float(table.float().pow(2).mean().sqrt())
    adapter.init_scale(rms)
    src = tc.get("init_output_from")
    if src and os.path.exists(os.path.join(src, "adapter.pt")):  # vectors already in the thinker's space
        old = torch.load(os.path.join(src, "adapter.pt"), map_location="cpu", weights_only=False)["state_dict"]
        own = adapter.state_dict()
        copied = [k for k in OUTPUT_PARAMS if k in old and old[k].shape == own[k].shape]
        adapter.load_state_dict({k: old[k] for k in copied}, strict=False)
        log(f"output layers from {src}: {copied}")
    resume = tc.get("resume_from")
    if resume and os.path.exists(os.path.join(resume, "adapter.pt")):  # a previous session of this stage
        old = torch.load(os.path.join(resume, "adapter.pt"), map_location="cpu", weights_only=False)["state_dict"]
        fresh = load_matching(adapter, old)
        log(f"resumed from {resume}; left fresh: {fresh or 'nothing'}")
    enc = acfg.get("encoder", "mimi")
    found = {r.get("encoder", "mimi") for r in rows + valid_rows}
    if found != {enc}:
        raise ValueError(f"adapter.encoder is {enc} but the manifests contain {sorted(found)}")
    adapter.to(device).train()
    module = AlignModule(adapter, table)
    # speech_start/end are only used in stage B (no gradient here); the table is identical on every rank
    model = DDP(module, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True,
                broadcast_buffers=False) if world > 1 else module
    log(f"adapter {count_params(adapter) / 1e6:.1f}M params, input {enc} ({latent_dim}-d), {world} process(es)")

    tok_map = {int(t): i + 1 for i, t in enumerate(vocab)}
    common = dict(prompts=prompts, system_prompt=cfg.thinker.system_prompt,
                  transcribe_instruction=cfg.thinker.transcribe_instruction, transcribe_prob=1.0,
                  max_trailing_frames=cfg.train_speech_llm.max_trailing_frames, max_frames=cfg.adapter.max_frames,
                  max_target_tokens=8, tok_map=tok_map, eot_early=int(tc.eot_early_frames),
                  unclear_reply=cfg.runtime.get("unclear_reply", ""))
    train_ds = SpeechLLMDataset(rows, train=True, **common)
    valid_ds = SpeechLLMDataset(valid_rows, train=False, **common)
    total = int(tc.max_steps) * int(tc.batch_size)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, sampler=mixture_sampler(weights, total, cfg.seed + 1000 * rank),
                              collate_fn=collate_speech_llm, num_workers=tc.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=tc.batch_size, collate_fn=collate_speech_llm,
                              num_workers=tc.num_workers)
    params = list(adapter.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": float(tc.lr), "base_lr": float(tc.lr)}],
                            weight_decay=float(tc.weight_decay))
    scaler = make_grad_scaler(enabled=False)  # the adapter trains in fp32 (it is small)

    def losses(batch) -> dict:
        lat = batch["latents"].to(device)
        lengths = batch["lengths"].to(device)
        out = model(lat)
        tok = token_ctc_from_logits(out["tok_logits"], lengths, batch["tok"])
        logp = F.log_softmax(out["ctc_logits"].float(), dim=-1).transpose(0, 1)
        ctc = F.ctc_loss(logp, batch["ctc_targets"].to(device), lengths * adapter.ctc_upsample,
                         batch["ctc_lengths"].to(device), blank=0, zero_infinity=True)
        eot = batch["eot"].to(device)
        valid = eot >= 0
        eot_l = F.binary_cross_entropy_with_logits(out["eot_logits"].float()[valid], eot[valid].float()) \
            if valid.any() else tok.new_zeros(())
        loss = float(tc.tok_weight) * tok + float(tc.ctc_weight) * ctc + float(tc.eot_weight) * eot_l
        return {"loss": loss, "tok": tok.detach(), "ctc": ctc.detach(), "eot": eot_l.detach(), "out": out}

    @torch.no_grad()
    def evaluate() -> dict:
        adapter.eval()
        refs, tok_hyps, chr_hyps, ok, n = [], [], [], 0, 0
        for bi, batch in enumerate(valid_loader):
            if bi >= int(tc.eval_batches):
                break
            res = losses(batch)
            lg = res["out"]["tok_logits"]
            for i, length in enumerate(batch["lengths"].tolist()):
                if batch["tasks"][i] == "unclear":
                    continue
                refs.append(batch["texts"][i])
                tok_hyps.append(token_greedy_text(adapter, lg[i], length, tokenizer))
                chr_hyps.append(ctc_greedy_decode(res["out"]["ctc_logits"][i, : length * adapter.ctc_upsample]
                                                  .argmax(-1).tolist()))
            eot = batch["eot"].to(device)
            v = eot >= 0
            ok += int(((res["out"]["eot_logits"] > 0).long() == eot)[v].sum())
            n += int(v.sum())
        adapter.train()
        return {"tok_wer": spoken_wer(refs, tok_hyps), "ctc_wer": spoken_wer(refs, chr_hyps), "eot_acc": ok / max(1, n),
                "samples": list(zip(refs[:3], tok_hyps[:3]))}

    def save(at_step: int) -> None:
        os.makedirs(tc.output_dir, exist_ok=True)
        adapter.save(os.path.join(tc.output_dir, "adapter.pt"))
        save_config(cfg, os.path.join(tc.output_dir, "config.yaml"))
        save_json(os.path.join(tc.output_dir, "state.json"), {"step": at_step, "stage": "align", "world_size": world})
        log(f"saved {tc.output_dir}")

    log(f"train rows {len(rows)}, valid rows {len(valid_rows)}; starting")
    step, running, t0 = 0, {}, time.time()
    for batch in train_loader:
        res = losses(batch)
        res["loss"].backward()
        for k in ("loss", "tok", "ctc", "eot"):
            running[k] = running.get(k, 0.0) + float(res[k])
        scale = cosine_lr(step, tc.warmup_steps, tc.max_steps)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * scale
        gnorm = optimizer_step(opt, scaler, params, tc.max_grad_norm)
        step += 1
        arm_watchdog()
        apply_time_budget(tc, step, t0, device, log)
        if step % tc.log_every == 0:
            log(f"step {step} {fmt({k: v / tc.log_every for k, v in running.items()})} gnorm {gnorm:.2f} "
                f"lr_scale {scale:.3f} {(time.time() - t0) / step:.3f}s/step")
            running = {}
        if step % tc.eval_every == 0 or step == tc.max_steps:
            if main_proc:
                r = evaluate()
                log(f"[eval step {step}] token WER {r['tok_wer'] * 100:.2f}%  char CTC WER {r['ctc_wer'] * 100:.2f}%  "
                    f"eot_acc {r['eot_acc'] * 100:.1f}%")
                for ref, hyp in r["samples"]:
                    log(f"   REF: {ref}\n   HYP: {hyp}")
            barrier()
        if step % tc.save_every == 0 or step == tc.max_steps:
            if main_proc:
                save(step)
            barrier()
        if step >= tc.max_steps:
            break
    with contextlib.suppress(Exception):
        disarm_watchdog()
    cleanup()


if __name__ == "__main__":
    main()
