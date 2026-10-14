"""Load test of the batched thinker (runtime/batch_engine.py): N simulated hotel calls at once on one GPU.

    python -m s2s.eval.batch_load_test --config configs/qwen4b_v3.yaml --speech-llm-dir checkpoints/speech_llm_v4 \\
        --manifest data/manifests/hotel5_eval_q4.jsonl --levels 1,8,16,32,48 --out results/load_v4

Each simulated call: its hotel's system prompt and tools are prefilled when the call starts; then for each turn
the guest "speaks" (a recorded hotel request's listener features; the call waits as long as the speech lasts),
the turn is sent at its end, the 4B answers (tool calls get a mock result and a second round), and the guest
listens for as long as the reply would take to say (~0.07 s per character) and thinks 0.5-1.5 s before the
next turn. Calls start spread over the first seconds, so turns overlap as in real traffic.

Measured per turn, from the end of the guest's speech: first reply token, first SPOKEN token (after any tool
round), and the reply speed (tokens per second while answering; speaking needs about 4-6). Per level: p50/p95,
the largest decode batch, the decode step time and peak GPU memory. Not included: the listener (stored
features; it runs while the guest speaks) and the voice (Kokoro, ~0.1-0.2 s for the first phrase on a T4).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time

import numpy as np
import torch

from s2s.config import load_config
from s2s.data.datasets import load_manifest
from s2s.data.hotel import row_system, tools_by_name
from s2s.models.adapter import SpeechAdapter
from s2s.models.thinker import Thinker
from s2s.runtime.batch_engine import BatchedThinker, BatchScheduler, Job
from s2s.utils import resolve_device, resolve_dtype


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


class Caller(threading.Thread):
    def __init__(self, idx: int, ctx: dict, rows: list[dict], first: dict, start_delay: float, args, rng: random.Random):
        super().__init__(daemon=True)
        self.idx, self.ctx, self.rows, self.first = idx, ctx, rows, first
        self.delay, self.args, self.rng = start_delay, args, rng
        self.turns: list[dict] = []
        self.error: str | None = None
        self.setup_ms = None

    def _run_job(self, pieces: list, generate: bool) -> tuple[list[tuple[int, float]], dict]:
        sched, done, out, stats = self.ctx["sched"], threading.Event(), [], {}

        def on_done(st):
            stats.update(st)
            done.set()

        sched.submit(Job(self.conv, pieces, generate=generate, max_new=self.args.max_new,
                         temperature=self.args.temperature, on_token=lambda t, now: out.append((t, now)),
                         on_done=on_done))
        if not done.wait(600):
            raise RuntimeError("no answer within 600 s")
        if "error" in stats:
            raise RuntimeError(stats["error"])
        return out, stats

    def run(self) -> None:
        try:
            self._call()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)

    def _call(self) -> None:
        a, ctx = self.args, self.ctx
        prompts, tok = ctx["prompts"], ctx["tokenizer"]
        time.sleep(self.delay)
        system = row_system(self.first, ctx["system_prompt"])
        tools = tools_by_name(self.first.get("tools"))
        prefix, first_close = prompts.prompt_parts(system, tools, None)
        next_open, next_close = prompts.user_turn_parts()
        self.conv = ctx["sched"].open()
        try:
            t0 = time.perf_counter()
            self._run_job([("ids", prefix)], generate=False)
            self.setup_ms = (time.perf_counter() - t0) * 1000
            for k in range(a.turns):
                row = self.first if k == 0 else self.rng.choice(self.rows)
                lat = torch.from_numpy(np.load(row["latent"]).astype(np.float32))
                time.sleep(lat.shape[0] * 0.08 * a.speech_speed)  # the guest speaks (the listener runs meanwhile)
                t_end = time.perf_counter()
                pieces = ([] if k == 0 else [("ids", next_open)]) + [("speech", lat)] + \
                    [("ids", first_close if k == 0 else next_close)]
                first_tok = first_spoken = None
                n_tokens, spoken, gaps, rounds, queued = 0, [], [], 0, []
                while True:
                    out, st = self._run_job(pieces, generate=True)
                    queued.append(st.get("queued_ms", 0.0))
                    ids = [t for t, _ in out]
                    n_tokens += len(ids)
                    if out:
                        first_tok = first_tok if first_tok is not None else out[0][1]
                        gaps += [b - a_ for (_, a_), (_, b) in zip(out, out[1:])]
                    in_tool = False
                    for t, when in out:
                        if t == prompts.tool_call_start_id:
                            in_tool = True
                        elif t == prompts.tool_call_end_id:
                            in_tool = False
                        elif not in_tool and tok.decode([t]).strip():
                            spoken.append(t)
                            if first_spoken is None:
                                first_spoken = when
                    if prompts.tool_call_start_id in ids and rounds < a.tool_rounds:
                        rounds += 1
                        pieces = [("ids", prompts.tool_response_ids([{"success": True}]))]
                        continue
                    break
                text = tok.decode(spoken, skip_special_tokens=True)
                self.turns.append({
                    "call": self.idx, "turn": k, "speech_s": round(lat.shape[0] * 0.08, 2),
                    "first_token_ms": None if first_tok is None else (first_tok - t_end) * 1000,
                    "first_spoken_ms": None if first_spoken is None else (first_spoken - t_end) * 1000,
                    "tokens": n_tokens, "tool_rounds": rounds, "queued_ms": max(queued) if queued else 0.0,
                    "tok_per_s": (len(gaps) / sum(gaps)) if gaps and sum(gaps) > 0 else None, "reply": text[:200]})
                # the guest listens to the reply, then thinks
                time.sleep(min(15.0, 0.07 * len(text)) * a.speech_speed + self.rng.uniform(0.5, 1.5) * a.speech_speed)
        finally:
            ctx["sched"].close(self.conv)


def run_level(n: int, ctx: dict, rows: list[dict], args) -> dict:
    sched = BatchScheduler(ctx["engine"], ctx["embed"], ctx["adapter"], ctx["boundary"], eos_ids=ctx["eos_ids"],
                           im_end_id=ctx["im_end_id"], prefill_budget=args.prefill_budget)
    ctx["sched"] = sched
    sched.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(args.seed + n)
    spread = min(args.stagger, 0.4 * n)
    callers = [Caller(i, ctx, rows, rows[rng.randrange(len(rows))], rng.uniform(0, spread), args,
                      random.Random(args.seed * 7919 + n * 131 + i)) for i in range(n)]
    t0 = time.time()
    for c in callers:
        c.start()
    for c in callers:
        c.join()
    wall = time.time() - t0
    sched.shutdown()
    sched.join(10)
    turns = [t for c in callers for t in c.turns]
    errors = [c.error for c in callers if c.error] + ([repr(sched.error)] if sched.error else [])
    ft = [t["first_token_ms"] for t in turns if t["first_token_ms"] is not None]
    fs = [t["first_spoken_ms"] for t in turns if t["first_spoken_ms"] is not None]
    rate = [t["tok_per_s"] for t in turns if t["tok_per_s"]]
    st = sched.stats
    res = {
        "calls": n, "turns": len(turns), "errors": errors[:5], "n_errors": len(errors), "wall_s": round(wall, 1),
        "first_token_p50_ms": pct(ft, 50), "first_token_p95_ms": pct(ft, 95),
        "first_spoken_p50_ms": pct(fs, 50), "first_spoken_p95_ms": pct(fs, 95),
        "tok_per_s_p50": pct(rate, 50), "tok_per_s_p5": pct(rate, 5),
        "call_setup_p50_ms": pct([c.setup_ms for c in callers if c.setup_ms], 50),
        "max_batch": st["max_batch"], "avg_batch": st["decode_rows"] / max(1, st["decode_steps"]),
        "decode_step_ms": 1000 * st["decode_s"] / max(1, st["decode_steps"]),
        "prefill_tok_per_s": st["prefill_tokens"] / max(1e-9, st["prefill_s"]),
        "gen_tok_per_s_total": sum(t["tokens"] for t in turns) / max(1e-9, wall),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
    }
    res["sample_turns"] = turns[:3]
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/qwen4b_v3.yaml")
    p.add_argument("--speech-llm-dir", default="checkpoints/speech_llm_v4")
    p.add_argument("--manifest", default="data/manifests/hotel5_eval_q4.jsonl")
    p.add_argument("--levels", default="1,8,16,32,48", help="simultaneous calls, one run each")
    p.add_argument("--turns", type=int, default=4, help="guest turns per call")
    p.add_argument("--max-tokens", type=int, default=4096, help="KV cache positions per call")
    p.add_argument("--prefill-budget", type=int, default=1024, help="prefill tokens per scheduler step")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tool-rounds", type=int, default=2)
    p.add_argument("--stagger", type=float, default=10.0, help="calls start spread over this many seconds")
    p.add_argument("--speech-speed", type=float, default=1.0, help="<1 squeezes the waiting (quick smoke runs)")
    p.add_argument("--max-rows", type=int, default=300)
    p.add_argument("--label", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="config overrides, key=value")
    p.add_argument("--out", default="results/batch_load_test")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    device = resolve_device(args.device or cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    lora = os.path.join(args.speech_llm_dir, "lora")
    th = Thinker(cfg.thinker.model, device, dtype, lora_dir=lora if os.path.isdir(lora) else None, merge_lora=True)
    th.model.eval()
    adapter = SpeechAdapter.load(os.path.join(args.speech_llm_dir, "adapter.pt")).to(device).eval()
    start, end = adapter.boundary_embeddings()
    rows = [r for r in load_manifest(args.manifest) if r.get("latent") and os.path.exists(r["latent"])][: args.max_rows]
    if not rows:
        raise SystemExit(f"no rows with stored features in {args.manifest}")
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    engine = BatchedThinker(th.model, slots=max(levels), max_tokens=args.max_tokens, device=device, dtype=dtype)
    embed_layer = th.model.get_input_embeddings()
    ctx = {"engine": engine, "adapter": adapter, "boundary": (start.detach(), end.detach()), "prompts": th.prompts,
           "tokenizer": th.tokenizer, "eos_ids": th.eos_ids, "im_end_id": th.im_end_id,
           "system_prompt": cfg.thinker.system_prompt,
           "embed": lambda ids: embed_layer(torch.tensor(ids, device=device))}
    label = args.label or os.path.basename(os.path.normpath(args.speech_llm_dir))
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"[load] {label} on {gpu}: {len(rows)} hotel turns, levels {levels}, KV cache {engine.cache_gb:.1f} GB "
          f"({max(levels)} calls x {args.max_tokens} positions)", flush=True)
    # warm-up: one short call (kernel selection, allocations)
    warm = argparse.Namespace(**{**vars(args), "turns": 1, "speech_speed": 0.0})
    run_level(1, ctx, rows, warm)
    os.makedirs(args.out, exist_ok=True)
    results = []
    for n in levels:
        res = run_level(n, ctx, rows, args)
        results.append(res)
        print(f"RESULT {label} calls={n:3d} turns={res['turns']:4d} | first token p50 {res['first_token_p50_ms']:6.0f} "
              f"p95 {res['first_token_p95_ms']:6.0f} ms | first spoken p50 {res['first_spoken_p50_ms']:6.0f} "
              f"p95 {res['first_spoken_p95_ms']:6.0f} ms | per-call {res['tok_per_s_p50']:5.1f} tok/s "
              f"(slowest 5%: {res['tok_per_s_p5']:5.1f}) | batch max {res['max_batch']} avg {res['avg_batch']:.1f} | "
              f"step {res['decode_step_ms']:5.1f} ms | GPU {res['peak_gpu_gb']:.1f} GB | errors {res['n_errors']}",
              flush=True)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump({"label": label, "gpu": gpu, "args": vars(args), "levels": results}, f, indent=1)
    lines = [f"# Load test: {label} on {gpu}", "",
             "Times from the end of the guest's speech (listener and voice not included; see s2s/eval/batch_load_test.py).", "",
             "| Calls at once | First token p50 / p95 | First spoken token p50 / p95 | Reply speed per call p50 (slowest 5%) "
             "| Batch max / avg | Decode step | Peak GPU memory | Errors |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['calls']} | {r['first_token_p50_ms']:.0f} / {r['first_token_p95_ms']:.0f} ms | "
                     f"{r['first_spoken_p50_ms']:.0f} / {r['first_spoken_p95_ms']:.0f} ms | {r['tok_per_s_p50']:.1f} "
                     f"({r['tok_per_s_p5']:.1f}) tok/s | {r['max_batch']} / {r['avg_batch']:.1f} | "
                     f"{r['decode_step_ms']:.1f} ms | {r['peak_gpu_gb']:.1f} GB | {r['n_errors']} |")
    lines += ["", "Speaking needs about 4-6 reply tokens per second per call; below that the voice would wait for text."]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
