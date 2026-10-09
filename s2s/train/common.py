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


def apply_time_budget(tc, step: int, t0: float, device, log) -> None:
    """`max_minutes` (0/unset = off): end training inside a fixed time (e.g. a 12 h Kaggle session).

    At steps 30 and 150 (still in warm-up) max_steps is shrunk to what fits in 90% of the budget at the
    measured speed, so the cosine schedule still ends at a low learning rate; if the budget runs out
    anyway (evaluations are slower than steps), max_steps becomes the current step, which triggers the
    final evaluation and save. Rank 0 decides for every rank."""
    import time

    from s2s.dist import broadcast_int

    minutes = float(tc.get("max_minutes") or 0)
    if minutes <= 0:
        return
    elapsed = time.time() - t0
    if step in (30, 150):
        fit = min(int(tc.max_steps), max(step + 1, int(minutes * 60 * 0.9 / (elapsed / step))))
        fit = broadcast_int(fit, device)
        if fit != int(tc.max_steps):
            log(f"time budget {minutes:g} min at {elapsed / step:.2f}s/step: max_steps {tc.max_steps} -> {fit}")
            tc.max_steps = fit
    elif step % int(tc.log_every) == 0 and broadcast_int(elapsed > minutes * 60, device):
        log(f"time budget of {minutes:g} min used up at step {step}: final evaluation and save")
        tc.max_steps = step
