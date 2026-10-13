"""Duplex frame speed: the normal path vs the whole frame from one CUDA graph, on the real checkpoint.

    python -m s2s.eval.duplex_speed --ckpt checkpoints/duplex --frames 750

Feeds the same random listener vectors through a session both ways (a 60 s call = 750 frames) and prints ms per
80 ms frame (LLM + text + 8 depth steps, + Mimi decoding) and the real-time factor: below 1.0 the model keeps up,
above 1.0 it falls further behind every second (the 10-20 s lag seen live on a T4 at 1.53).
"""

from __future__ import annotations

import argparse
import time

import torch

from s2s.runtime.duplex import DuplexSession, load_duplex


def run(model, tok, codec, feats, graphs: bool, max_tokens: int) -> dict:
    from s2s.models.codec import StreamingDecoder

    sess = DuplexSession(model, tok, "You are a hotel concierge.", None, None, 0.6, 30, graphs=graphs,
                         graph_max_tokens=max_tokens)
    dec = StreamingDecoder(codec) if codec is not None else None
    dev = model.in_scale.device
    for t in range(10):  # warm-up
        sess.step(feats[t])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0, mimi = time.perf_counter(), 0.0
    for t in range(10, feats.shape[0]):
        out = sess.step(feats[t])
        if dec is not None and out["frame"] is not None:
            t1 = time.perf_counter()
            dec.push([out["frame"].clamp(min=0)])
            mimi += time.perf_counter() - t1
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    n = feats.shape[0] - 10
    total = (time.perf_counter() - t0) / n * 1000
    return {"graphs": sess.g is not None, "ms_frame": total, "ms_mimi": mimi / n * 1000, "rtf": total / 80}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/duplex")
    p.add_argument("--frames", type=int, default=750)
    p.add_argument("--graph-max-tokens", type=int, default=4096)
    p.add_argument("--no-mimi", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    from s2s.models.codec import MimiCodec

    device = torch.device(args.device)
    model, tok, cfg, _ = load_duplex(args.ckpt, device)
    codec = None if args.no_mimi else MimiCodec(cfg.codec.model, device, num_codebooks=int(cfg.codec.num_codebooks))
    feats = torch.randn(args.frames, model.feat_proj.in_features, generator=torch.Generator().manual_seed(0))
    print(f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}, {args.frames} frames "
          f"({args.frames * 0.08:.0f} s of call)", flush=True)
    for graphs in (False, True):
        r = run(model, tok, codec, feats, graphs, args.graph_max_tokens)
        label = "CUDA graph" if r["graphs"] else ("normal (graph failed)" if graphs else "normal")
        print(f"RESULT {label:22s} {r['ms_frame']:6.1f} ms per 80 ms frame (Mimi {r['ms_mimi']:.1f} ms) | "
              f"real-time factor {r['rtf']:.2f} {'OK' if r['rtf'] < 1 else 'FALLS BEHIND'}", flush=True)


if __name__ == "__main__":
    main()
