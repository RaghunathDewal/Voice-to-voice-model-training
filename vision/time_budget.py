"""(vision/time_budget.py) ms-swift plugin: fit the training into a Kaggle session, like max_minutes in the speech trainer.

    swift sft ... --external_plugins time_budget.py --callbacks time_budget

  1. After CALIB_STEPS updates it measures the time per update and sets the total number of updates to what fits in
     TOTAL_HOURS (never more than the full epochs). TOTAL_HOURS > TRAIN_HOURS spreads one run over several sessions:
     e.g. TRAIN_HOURS=9 TOTAL_HOURS=18 = two Kaggle commits, the second resumes and finishes the decay. The cosine learning rate is rebuilt for that total, so the run
     ends with a fully decayed learning rate instead of being cut in the middle.
  2. At the deadline it saves and stops, whatever happens.
  3. Every saved checkpoint is uploaded (in the background) to HF_CKPT_REPO/last-checkpoint, with the optimizer, so a
     new session resumes exactly where this one stopped. Older uploads are purged from the repo's storage.
Env: TRAIN_HOURS (this session, default 9), TOTAL_HOURS (whole run, default = TRAIN_HOURS), CALIB_STEPS (20), HF_CKPT_REPO (empty = no upload), HF_TOKEN.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import torch
import torch.distributed as dist
from transformers import TrainerControl, TrainerState

from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback

T0 = time.time()
PLAN = "time_budget_plan.json"


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _from_rank0(x: float) -> float:
    """Every GPU must take the same decision (else one stops and the other hangs)."""
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=f"cuda:{torch.cuda.current_device()}")
    dist.broadcast(t, 0)
    return t.item()


def _cosine(optimizer, warmup: int, total: int, min_ratio: float = 0.1):
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class TimeBudgetCallback(TrainerCallback):

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.hours = float(os.environ.get("TRAIN_HOURS", 9))
        self.total_hours = float(os.environ.get("TOTAL_HOURS") or self.hours)
        self.calib = int(os.environ.get("CALIB_STEPS", 20))
        self.repo = os.environ.get("HF_CKPT_REPO", "")
        self.plan = None
        self.t_calib = None
        self.upload_thread = None
        self.pending = None

    # ---------------------------------------------------------------- schedule
    def _set_total(self, state: TrainerState, total: int) -> None:
        total = max(state.global_step + 1, min(total, state.max_steps))
        warmup = self.args.get_warmup_steps(total)
        sched = _cosine(self.trainer.optimizer, warmup, total)
        for _ in range(state.global_step):
            sched.step()
        self.trainer.lr_scheduler = sched
        self.plan = {"total_steps": total, "warmup_steps": warmup}
        if _rank0():
            print(f"[time_budget] training for {total} updates (full epochs = {state.max_steps}), "
                  f"lr cosine over {total}, warmup {warmup}", flush=True)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        ck = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")   # the checkpoint we resumed from
        if state.global_step and os.path.exists(os.path.join(ck, PLAN)):
            self._set_total(state, json.load(open(os.path.join(ck, PLAN)))["total_steps"])
            if _rank0():
                print(f"[time_budget] resumed at update {state.global_step}", flush=True)
        self.start_step = state.global_step

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        done = state.global_step - self.start_step
        if done == 2:
            self.t_calib = time.time()                   # skip the slow first updates
        if self.plan is None and done == self.calib + 2:
            per = _from_rank0((time.time() - self.t_calib) / self.calib) * 1.08   # +8% for evals and saves
            left = self.total_hours * 3600 - (time.time() - T0)
            self._set_total(state, state.global_step + int(left / per))
            if _rank0():
                print(f"[time_budget] {per:.1f} s per update, {left / 3600:.2f} h left", flush=True)
        if self.plan and state.global_step >= self.plan["total_steps"]:
            control.should_save, control.should_training_stop = True, True
        over = _from_rank0(float(time.time() - T0 > self.hours * 3600))
        if over:
            if _rank0():
                print(f"[time_budget] {self.hours} h reached at update {state.global_step}: save and stop", flush=True)
            control.should_save, control.should_training_stop = True, True
        return control

    # ---------------------------------------------------------------- checkpoints -> Hugging Face
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kw):
        ck = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not _rank0() or not os.path.isdir(ck):
            return
        if self.plan:
            json.dump(self.plan, open(os.path.join(ck, PLAN), "w"))
        if not self.repo:
            return
        self.pending = ck                                # newest checkpoint wins; the worker uploads it next
        if not (self.upload_thread and self.upload_thread.is_alive()):
            self.upload_thread = threading.Thread(target=self._worker, daemon=False)
            self.upload_thread.start()

    def _worker(self) -> None:
        while self.pending:
            ck, self.pending = self.pending, None
            self._upload(ck)

    def _upload(self, ck: str) -> None:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(self.repo, private=True, exist_ok=True)
            api.upload_folder(folder_path=ck, path_in_repo="last-checkpoint", repo_id=self.repo,
                              delete_patterns=["*"], commit_message=f"checkpoint {os.path.basename(ck)}")
            cur = {f.lfs.sha256 for f in api.repo_info(self.repo, files_metadata=True).siblings if f.lfs}
            old = [f for f in api.list_lfs_files(self.repo) if f.file_oid not in cur]
            if old:
                api.permanently_delete_lfs_files(self.repo, old, rewrite_history=True)   # keep storage flat
            print(f"[time_budget] uploaded {os.path.basename(ck)} to {self.repo}/last-checkpoint", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[time_budget] upload failed (checkpoint is still on disk): {e}", flush=True)

    def on_train_end(self, args, state, control, **kw):
        if self.upload_thread:
            self.upload_thread.join()                    # the final checkpoint is on HF before the process exits
        if self.pending:
            self._worker()


callbacks_map["time_budget"] = TimeBudgetCallback
