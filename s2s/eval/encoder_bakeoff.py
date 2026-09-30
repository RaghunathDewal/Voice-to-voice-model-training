"""Encoder bake-off: which frozen speech encoder should feed the thinker?

For every encoder, on identical data:
  1. extract 12.5 Hz features (s2s/models/encoders.py) for the train set and each test set
  2. train the same small CTC probe (2-layer transformer + character CTC) on the train set
  3. report greedy WER on every test set (e.g. accented / noisy speech)
  4. measure end-of-turn encode latency for a 5 s utterance and peak GPU memory
Features are deleted after each encoder unless --keep-features.

    python -m s2s.eval.encoder_bakeoff --config configs/small.yaml \
        --train data/manifests/cv_train_raw.jsonl --train-hours 15 --train-augment 0.5 \
        --test edacc=data/manifests/edacc_valid_raw.jsonl cv=data/manifests/cv_test_raw.jsonl \
               edacc_noisy=data/manifests/edacc_valid_raw.jsonl:aug \
        --encoders mimi whisper:openai/whisper-small parakeet:nvidia/parakeet-ctc-0.6b

With several GPUs the encoders are spread over them (one worker process per GPU).
Results: <out-dir>/results.json, <out-dir>/<encoder>/result.json and a table on stdout.
The probe WER is a *relative* measure of how much speech content an encoder exposes
to a small model; the adapter + thinker reach much lower WER than the probe.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from s2s.audio import load_audio
from s2s.augment import augment
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.adapter import SpeechAdapter
from s2s.models.encoders import build_encoder
from s2s.train.probe_ctc import LatentCTCDataset, collate, evaluate
from s2s.utils import cosine_lr, infinite, save_json, set_seed

DEFAULT_ENCODERS = ["mimi", "whisper:openai/whisper-small", "whisper:openai/whisper-large-v3-turbo",
                    "parakeet:nvidia/parakeet-ctc-0.6b", "moonshine:moonshine-ai/moonshine-streaming-small"]


def slug(spec: str) -> str:
    return spec.replace(":", "_").replace("/", "_")


def parse_tests(items: list[str]) -> list[tuple[str, str, bool]]:
    out = []
    for it in items:
        name, _, rest = it.partition("=")
        aug = rest.endswith(":aug")
        out.append((name, rest[: -len(":aug")] if aug else rest, aug))
    return out


def take_hours(rows: list[dict], hours: float) -> list[dict]:
    if not hours:
        return rows
    out, total = [], 0.0
    for r in rows:
        if total >= hours * 3600:
            break
        out.append(r)
        total += float(r.get("duration") or 5.0)
    return out


def extract(enc, rows: list[dict], feat_dir: str, aug_prob: float, tag: str, batch_size: int) -> list[dict]:
    os.makedirs(feat_dir, exist_ok=True)
    out, batch = [], []

    def flush():
        feats = enc.encode([w for _, w in batch])
        for (r, _), f in zip(batch, feats):
            path = os.path.join(feat_dir, f"{r['id']}.npy")
            np.save(path, f.numpy().astype(np.float16))
            out.append({"id": r["id"], "text": r["text"], "latent": path})
        batch.clear()

    for r in tqdm(rows, desc=f"{tag}", mininterval=20):
        try:
            wav = load_audio(r["audio"], enc.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"skip {r['id']}: {e}")
            continue
        if len(wav) < enc.sample_rate * 0.3:
            continue
        key = f"bakeoff/{tag}/{r['id']}"
        if aug_prob >= 1.0 or (aug_prob > 0 and zlib.crc32(key.encode()) % 10000 < aug_prob * 10000):
            wav = augment(wav, enc.sample_rate, key)
        batch.append((r, wav))
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    return out


def measure_latency(enc, device: torch.device, seconds: float = 5.0, runs: int = 10) -> dict:
    rng = np.random.default_rng(0)
    wav = (0.05 * rng.standard_normal(int(seconds * enc.sample_rate))).astype(np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        enc.encode([wav])
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        enc.encode([wav])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    return {"encode_ms_5s_p50": float(np.median(times)), "encode_ms_5s_max": float(np.max(times)),
            "peak_gpu_mb": round(mem)}


def train_probe(cfg, train_rows, valid_rows, device, max_steps: int, log) -> SpeechAdapter:
    pc = cfg.train_probe
    set_seed(cfg.seed)
    dim = np.load(train_rows[0]["latent"], mmap_mode="r").shape[1]
    model = SpeechAdapter(dim, llm_dim=8, d_model=pc.d_model, n_layers=pc.n_layers, n_heads=pc.n_heads,
                          dropout=0.1, ctc_upsample=cfg.adapter.ctc_upsample).to(device)
    max_frames = int(cfg.adapter.max_frames)
    tl = DataLoader(LatentCTCDataset(train_rows, max_frames), batch_size=pc.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=2, drop_last=True)
    vl = DataLoader(LatentCTCDataset(valid_rows, max_frames), batch_size=pc.batch_size, collate_fn=collate,
                    num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=pc.lr, weight_decay=0.01)
    it, best, best_state = infinite(tl), None, None
    for step in range(max_steps):
        for g in opt.param_groups:
            g["lr"] = pc.lr * cosine_lr(step, 200, max_steps)
        b = next(it)
        lengths = b["lengths"].to(device)
        logits = model(b["latents"].to(device))["ctc_logits"]
        logp = F.log_softmax(logits.float(), -1).transpose(0, 1)
        loss = F.ctc_loss(logp, b["targets"].to(device), lengths * model.ctc_upsample,
                          b["target_lengths"].to(device), zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 250 == 0:
            log(f"  step {step + 1} ctc_loss {loss.item():.3f}")
        if (step + 1) % pc.eval_every == 0 or step + 1 == max_steps:
            res = evaluate(model, vl, device)
            log(f"  step {step + 1} held-out WER {res['wer'] * 100:.2f}%")
            if best is None or res["wer"] < best:
                best = res["wer"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def run_one(cfg, args, spec: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    out_dir = os.path.join(args.out_dir, slug(spec))
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")

    def log(msg: str) -> None:
        print(f"[{spec}] {msg}", flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    t0 = time.time()
    enc = build_encoder(spec, device)
    res = {"encoder": spec, "dim": enc.dim, "params_m": round(enc.params_m, 1), "streaming": enc.streaming}
    res.update(measure_latency(enc, device))
    log(f"loaded: {res}")
    feat_root = os.path.join(args.feat_dir or out_dir, "features", slug(spec))
    all_train = take_hours(load_manifest(args.train), args.train_hours)
    tests_spec = parse_tests(args.test)
    # fail fast (and cleanly) instead of filling the disk: features are float16 at 12.5 Hz
    hours = sum(float(r.get("duration") or 5.0) for r in all_train) / 3600 + 0.5 * len(tests_spec)
    need_gb = hours * 3600 * 12.5 * enc.dim * 2 / 1e9 * 1.1
    os.makedirs(feat_root, exist_ok=True)
    free_gb = shutil.disk_usage(feat_root).free / 1e9
    log(f"features need ~{need_gb:.1f} GB, {free_gb:.1f} GB free")
    if need_gb > free_gb - args.min_free_gb:
        raise RuntimeError(f"not enough disk for {spec}: need ~{need_gb:.1f} GB, free {free_gb:.1f} GB "
                           f"(keeping {args.min_free_gb} GB spare); lower --train-hours")
    try:
        n_hold = min(300, max(20, len(all_train) // 20))
        train = extract(enc, all_train[n_hold:], os.path.join(feat_root, "train"), args.train_augment, "train",
                        args.batch_size)
        held = extract(enc, all_train[:n_hold], os.path.join(feat_root, "held"), args.train_augment, "held-out",
                       args.batch_size)
        tests = {}
        for name, path, aug in tests_spec:
            rows = load_manifest(path)[: args.test_max]
            tests[name] = extract(enc, rows, os.path.join(feat_root, name), 1.0 if aug else 0.0, name,
                                  args.batch_size)
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        res["train_hours"] = round(sum(float(r.get("duration") or 0) for r in all_train[n_hold:]) / 3600, 1)
        log(f"features extracted ({res['train_hours']} h train) in {(time.time() - t0) / 60:.1f} min; training probe")
        model = train_probe(cfg, train, held, device, args.steps, log)
        for name, rows in tests.items():
            dl = DataLoader(LatentCTCDataset(rows, int(cfg.adapter.max_frames)), batch_size=32, collate_fn=collate)
            r = evaluate(model, dl, device)
            res[f"wer_{name}"] = round(r["wer"] * 100, 2)
            log(f"{name}: WER {res[f'wer_{name}']}% on {r['n']} utts")
            for ref, hyp in r["examples"][:2]:
                log(f"    REF: {ref.lower()}\n    HYP: {hyp}")
        res["minutes"] = round((time.time() - t0) / 60, 1)
        save_json(os.path.join(out_dir, "result.json"), res)
    finally:  # also on failure: leftover features filled the disk once
        if not args.keep_features:
            shutil.rmtree(feat_root, ignore_errors=True)
    return res


def table(results: list[dict]) -> str:
    wer_keys = sorted({k for r in results for k in r if k.startswith("wer_")})
    head = ["encoder", "streams", "params M", "encode ms (5 s)", "GPU MB"] + [k[4:] + " WER %" for k in wer_keys]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['encoder']} | failed: {r['error'][:80]} |")
            continue
        cells = [r["encoder"], "yes" if r["streaming"] else "no", str(r["params_m"]),
                 f"{r['encode_ms_5s_p50']:.0f}", str(r["peak_gpu_mb"])] + [str(r.get(k, "")) for k in wer_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--train", required=True, help="raw manifest (audio + text)")
    p.add_argument("--train-hours", type=float, default=15.0)
    p.add_argument("--train-augment", type=float, default=0.5)
    p.add_argument("--test", nargs="+", required=True, help="name=manifest[:aug] (':aug' = noisy/reverberant copy)")
    p.add_argument("--test-max", type=int, default=400)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16, help="extraction batch size")
    p.add_argument("--out-dir", default="outputs/encoder_bakeoff")
    p.add_argument("--gpus", type=int, default=0, help="parallel workers (0 = all visible GPUs)")
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--feat-dir", default=None, help="where temporary features go (default: --out-dir); use a big disk")
    p.add_argument("--min-free-gb", type=float, default=3.0, help="skip an encoder rather than leave less free disk")
    p.add_argument("--worker", default=None, help=__import__("argparse").SUPPRESS)
    args = p.parse_args()
    cfg = config_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.worker is not None:  # a worker handles a comma-separated list of encoders, one after another
        for spec in args.worker.split(","):
            try:
                run_one(cfg, args, spec)
            except Exception as e:  # noqa: BLE001 - one broken encoder must not stop the others
                import traceback

                traceback.print_exc()
                save_json(os.path.join(args.out_dir, slug(spec), "result.json"), {"encoder": spec, "error": repr(e)})
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and cfg.device != "cpu" else 0
    n_workers = max(1, min(args.gpus or n_gpu or 1, len(args.encoders)))
    groups = [args.encoders[i::n_workers] for i in range(n_workers)]
    argv = [a for a in sys.argv[1:]]
    procs = []
    for k, group in enumerate(groups):
        env = dict(os.environ)
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(k % n_gpu)
        cmd = [sys.executable, "-m", "s2s.eval.encoder_bakeoff", *argv, "--worker", ",".join(group)]
        print(f"worker {k} (GPU {k % max(1, n_gpu)}): {group}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    for pr in procs:
        pr.wait()
    results = []
    for spec in args.encoders:
        path = os.path.join(args.out_dir, slug(spec), "result.json")
        results.append(json.load(open(path)) if os.path.exists(path) else {"encoder": spec, "error": "no result"})
    save_json(os.path.join(args.out_dir, "results.json"), results)
    md = table(results)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
