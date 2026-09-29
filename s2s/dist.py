"""Multi-GPU helpers.

* Training uses PyTorch DDP. Launch with torchrun, e.g. on Kaggle's 2 x T4:
      torchrun --standalone --nproc_per_node=2 -m s2s.train.speech_llm --config configs/small.yaml
  Without torchrun the scripts run on one device exactly as before.
* Data preparation (Mimi extraction, reply distillation) is split into one
  worker process per GPU with `run_sharded` (use `--gpus N`, default: all).
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys

import torch
import torch.distributed as dist


# ------------------------------------------------------------------- DDP
def init_distributed(device_name: str = "auto") -> tuple[int, int, torch.device]:
    """Returns (rank, world_size, device). Initialises a process group under torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        from s2s.utils import resolve_device

        return 0, 1, resolve_device(device_name)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and device_name != "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    # long timeout: rank 0 runs evaluation / sample generation while the others wait
    dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(minutes=60))
    return dist.get_rank(), dist.get_world_size(), device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def mean_across_ranks(value: float, device: torch.device) -> float:
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return float(t.item()) / dist.get_world_size()


# --------------------------------------------------------- sharded workers
def gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_num_workers(requested: int | None) -> int:
    """--gpus value -> number of worker processes (0/None = all visible GPUs)."""
    n = gpu_count()
    if requested:
        return max(1, min(requested, n)) if n else 1
    return max(1, n)


def run_sharded(module: str, argv: list[str], num_shards: int) -> None:
    """Run `python -m module argv --shard k --num-shards n` once per GPU, in parallel."""
    procs = []
    for k in range(num_shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        cmd = [sys.executable, "-m", module, *argv, "--shard", str(k), "--num-shards", str(num_shards)]
        print(f"[shard {k}/{num_shards}] GPU {k}: started")
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    failed = [k for k, c in enumerate(codes) if c != 0]
    if failed:
        raise SystemExit(f"shard(s) {failed} failed; fix the error above and re-run (finished work is reused)")


def strip_gpu_args(argv: list[str]) -> list[str]:
    """Remove --gpus N from an argv list before forwarding it to shard workers."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--gpus":
            skip = True
            continue
        if a.startswith("--gpus="):
            continue
        out.append(a)
    return out
