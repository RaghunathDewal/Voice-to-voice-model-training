"""Training helpers shared by the stage scripts."""

from __future__ import annotations

import torch


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_step(opt, scaler, params, max_grad_norm: float) -> float:
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return float(norm)


def fmt(d: dict) -> str:
    return " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())
