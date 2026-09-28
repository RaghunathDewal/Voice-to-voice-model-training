"""Experiment 1 (go / no-go): can frozen Mimi latents be recognised?

Trains a small causal transformer + CTC head (characters) on Mimi latents and
reports greedy WER on the validation manifest. No LLM involved. If WER is poor
here, the thinker will mishear too, and the input front-end must change.

    python -m s2s.train.probe_ctc --config configs/default.yaml
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.adapter import SpeechAdapter
from s2s.text import ctc_encode, ctc_greedy_decode, wer
from s2s.utils import cosine_lr, count_params, infinite, resolve_device, save_json, set_seed


class LatentCTCDataset(Dataset):
    def __init__(self, rows: list[dict], max_frames: int):
        self.rows = rows
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        lat = np.load(r["latent"]).astype(np.float32)[: self.max_frames]
        return {"latent": torch.from_numpy(lat), "ctc": ctc_encode(r["text"]), "text": r["text"]}


def collate(items: list[dict]) -> dict:
    lengths = torch.tensor([it["latent"].shape[0] for it in items])
    x = torch.zeros(len(items), int(lengths.max()), items[0]["latent"].shape[1])
    for i, it in enumerate(items):
        x[i, : lengths[i]] = it["latent"]
    return {"latents": x, "lengths": lengths,
            "targets": torch.tensor([c for it in items for c in it["ctc"]], dtype=torch.long),
            "target_lengths": torch.tensor([len(it["ctc"]) for it in items]), "texts": [it["text"] for it in items]}


@torch.no_grad()
def evaluate(model: SpeechAdapter, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict:
    model.eval()
    refs, hyps = [], []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        logits = model(batch["latents"].to(device))["ctc_logits"]
        ids = logits.argmax(-1).cpu()
        for i, n in enumerate(batch["lengths"]):
            hyps.append(ctc_greedy_decode(ids[i, : int(n) * model.ctc_upsample].tolist()))
            refs.append(batch["texts"][i])
    model.train()
    return {"wer": wer(refs, hyps), "examples": list(zip(refs[:5], hyps[:5])), "n": len(refs)}


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--train", default=None, help="default: first train_speech_llm manifest")
    p.add_argument("--valid", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    pc = cfg.train_probe
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    train_rows = load_manifest(args.train or cfg.train_speech_llm.train_manifests[0]["path"])
    valid_rows = load_manifest(args.valid or cfg.train_speech_llm.valid_manifest)
    out_dir = args.out or os.path.join(cfg.paths.ckpt_dir, "probe_ctc")
    os.makedirs(out_dir, exist_ok=True)

    latent_dim = np.load(train_rows[0]["latent"], mmap_mode="r").shape[1]
    model = SpeechAdapter(latent_dim, llm_dim=8, d_model=pc.d_model, n_layers=pc.n_layers, n_heads=pc.n_heads,
                          dropout=0.1, ctc_upsample=cfg.adapter.ctc_upsample).to(device)
    print(f"probe params: {count_params(model) / 1e6:.1f}M")
    max_frames = int(cfg.adapter.max_frames)
    train_loader = DataLoader(LatentCTCDataset(train_rows, max_frames), batch_size=pc.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=True)
    valid_loader = DataLoader(LatentCTCDataset(valid_rows, max_frames), batch_size=pc.batch_size,
                              collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=pc.lr, weight_decay=0.01)
    it = infinite(train_loader)
    best = None
    for step in range(int(pc.max_steps)):
        for g in opt.param_groups:
            g["lr"] = pc.lr * cosine_lr(step, 200, pc.max_steps)
        batch = next(it)
        lengths = batch["lengths"].to(device)
        logits = model(batch["latents"].to(device))["ctc_logits"]
        logp = F.log_softmax(logits.float(), -1).transpose(0, 1)
        loss = F.ctc_loss(logp, batch["targets"].to(device), lengths * model.ctc_upsample,
                          batch["target_lengths"].to(device), zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} ctc_loss {loss.item():.3f}")
        if (step + 1) % pc.eval_every == 0 or step + 1 == pc.max_steps:
            res = evaluate(model, valid_loader, device)
            print(f"step {step + 1} valid WER {res['wer'] * 100:.2f}% on {res['n']} utts")
            for ref, hyp in res["examples"][:3]:
                print(f"  REF: {ref.lower()}\n  HYP: {hyp}")
            if best is None or res["wer"] < best:
                best = res["wer"]
                model.save(os.path.join(out_dir, "probe.pt"))
            save_json(os.path.join(out_dir, "result.json"), {"step": step + 1, "valid_wer": res["wer"], "best_wer": best})
    print(f"best valid WER: {best * 100:.2f}%  (compare against your pass threshold)")


if __name__ == "__main__":
    main()
