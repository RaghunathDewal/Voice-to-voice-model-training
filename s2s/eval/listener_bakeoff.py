"""Listener bake-off: Parakeet (v2 adapter's listener) vs Nemotron streaming (v3's), same clips, same GPU.

Accuracy: each model's own transcript (Parakeet: its CTC head; Nemotron: its RNNT decoder, streaming at
160 ms chunks, and full-context for reference), word error rate against the reference text.

Speed (the encoder output is what the adapter reads, so that is what is timed):
  ready after speech ends   how long after the guest's last word the listener vectors for it are ready.
                            Parakeet can only start when the utterance has ended: its whole-utterance encode
                            time. Nemotron runs while the guest speaks (audio fed in 80 ms pieces, as live):
                            the wait for its 160 ms chunk / look-ahead plus the compute of that last chunk.
  compute / real time       total encoder compute divided by the audio length (below 1 = keeps up)
  chunk compute             Nemotron, per 160 ms chunk (must stay well under 160 ms to keep up live)

The audio is prepared as s2s.eval.asr_nemotron prepares it (same trailing silence; --noisy-name adds the
same noise as that feature manifest), so the clips match the earlier tests.

    python -m s2s.eval.listener_bakeoff --config configs/qwen4b_v3.yaml \\
        --manifest data/manifests/slurp_test_raw.jsonl --max 150 --name slurp --out-dir outputs/listeners
"""

from __future__ import annotations

import json
import os
import time
import zlib

import numpy as np
import torch
from tqdm import tqdm

from s2s.audio import add_silence, load_audio
from s2s.augment import augment
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.eval.asr_nemotron import LOOKAHEAD
from s2s.eval.spoken_wer import spoken_wer
from s2s.models.encoders import NemotronEncoder, ParakeetEncoder
from s2s.models.streaming_listener import HOP, StreamingListener
from s2s.utils import resolve_device

SR = 16000


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--manifest", required=True, help="raw manifest (audio paths + text)")
    p.add_argument("--name", required=True)
    p.add_argument("--noisy-name", default=None)
    p.add_argument("--max", type=int, default=150)
    p.add_argument("--parakeet", default="nvidia/parakeet-ctc-0.6b")
    p.add_argument("--nemotron", default="nvidia/nemotron-speech-streaming-en-0.6b")
    p.add_argument("--no-offline", action="store_true", help="skip Nemotron's full-context transcript")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    from transformers import AutoModelForRNNT, AutoProcessor

    para = ParakeetEncoder(args.parakeet, device, torch.float32)
    nem = NemotronEncoder(args.nemotron, device, torch.float32, lookahead=1)
    proc = AutoProcessor.from_pretrained(args.nemotron)
    rnnt = AutoModelForRNNT.from_pretrained(args.nemotron).to(device).eval()
    hop, nfft = proc.feature_extractor.hop_length, proc.feature_extractor.n_fft
    silence = float(cfg.codec.trailing_silence_s)

    def to_dev(batch):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    @torch.no_grad()
    def nemotron_text(wav: np.ndarray, offline: bool) -> str:
        if offline:
            out = rnnt.generate(**to_dev(proc(wav, sampling_rate=SR, return_tensors="pt")), return_dict_in_generate=True)
            return proc.decode(out.sequences, skip_special_tokens=True)[0]
        proc.set_num_lookahead_tokens(LOOKAHEAD[160])
        wav = np.concatenate([wav, np.zeros(2 * SR, np.float32)])
        first = to_dev(proc(wav[: proc.num_samples_first_audio_chunk], sampling_rate=SR, is_streaming=True,
                            is_first_audio_chunk=True, return_tensors="pt"))

        def chunks():
            yield first["input_features"][:, : proc.num_mel_frames_first_audio_chunk, :]
            mel = proc.num_mel_frames_first_audio_chunk
            start = mel * hop - nfft // 2
            while (end := start + proc.num_samples_per_audio_chunk) < wav.shape[0]:
                yield proc(wav[start:end], sampling_rate=SR, is_streaming=True, is_first_audio_chunk=False,
                           return_tensors="pt").input_features.to(device)
                mel += proc.num_mel_frames_per_audio_chunk
                start = mel * hop - nfft // 2

        out = rnnt.generate(**{**first, "input_features": chunks()}, return_dict_in_generate=True)
        return proc.decode(out.sequences, skip_special_tokens=True)[0]

    @torch.no_grad()
    def parakeet_run(wav: np.ndarray) -> tuple[str, float]:
        sync(device)
        t0 = time.perf_counter()
        feats = para.encode([wav])
        sync(device)
        took = time.perf_counter() - t0
        return para.transcribe(feats)[0], took

    @torch.no_grad()
    def nemotron_live(wav: np.ndarray, speech_end: int) -> tuple[float, list[float]]:
        """Feed 80 ms pieces on a virtual real-time clock; returns (seconds after the speech end until the
        frame holding the last speech sample is out, compute per call that encoded something)."""
        lis = StreamingListener(nem, lookahead=1)
        target = -(-speech_end // HOP)  # frames needed to cover the speech
        ready, comp = None, []
        for pos in range(HOP, len(wav) + HOP, HOP):
            lis.push(wav[pos - HOP: pos])
            sync(device)
            t0 = time.perf_counter()
            n = len(lis.new_frames())
            sync(device)
            c = time.perf_counter() - t0
            if n:
                comp.append(c)
            if ready is None and lis.done >= target:
                ready = max(0.0, min(pos, len(wav)) - speech_end) / SR + c
        return (ready if ready is not None else float("nan")), comp

    # warm-up (CUDA kernels, cuDNN autotune) so the first clip's timing is not an outlier
    w = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.01
    parakeet_run(w)
    nemotron_live(w, len(w))

    rows = [r for r in load_manifest(args.manifest) if r.get("text")][: args.max or None]
    results = []
    for r in tqdm(rows, desc=f"listeners [{args.name}]", mininterval=30):
        try:
            wav = load_audio(r["audio"], SR)
        except Exception as e:  # noqa: BLE001
            print(f"skip {r.get('id')}: {e}")
            continue
        peak = float(np.abs(wav).max()) if len(wav) else 0.0
        if peak > 1.0:
            wav = wav / peak
        speech_end = len(wav)
        if silence:
            wav = add_silence(wav, SR, silence, seed=zlib.crc32(r["id"].encode()))
        if args.noisy_name:
            wav = augment(wav, SR, f"{args.noisy_name}/{r['id']}")
        p_text, p_time = parakeet_run(wav)
        n_ready, n_comp = nemotron_live(wav, speech_end)
        x = {"id": r["id"], "text": r["text"], "seconds": len(wav) / SR,
             "parakeet": p_text, "parakeet_ready_s": p_time, "parakeet_rtf": p_time / (len(wav) / SR),
             "nemotron": nemotron_text(wav, False), "nemotron_ready_s": n_ready,
             "nemotron_rtf": sum(n_comp) / (len(wav) / SR), "nemotron_chunk_s": n_comp}
        if not args.no_offline:
            x["nemotron_offline"] = nemotron_text(wav, True)
        results.append(x)

    refs = [x["text"] for x in results]
    s = {"set": args.name, "n": len(results), "seconds_median": pct([x["seconds"] for x in results], 50),
         "wer_parakeet": spoken_wer(refs, [x["parakeet"] for x in results]),
         "wer_nemotron_160ms": spoken_wer(refs, [x["nemotron"] for x in results])}
    if not args.no_offline:
        s["wer_nemotron_offline"] = spoken_wer(refs, [x["nemotron_offline"] for x in results])
    chunks = [c for x in results for c in x["nemotron_chunk_s"]]
    for k in ("parakeet", "nemotron"):
        s[f"{k}_ready_ms_p50"] = pct([x[f"{k}_ready_s"] * 1000 for x in results], 50)
        s[f"{k}_ready_ms_p90"] = pct([x[f"{k}_ready_s"] * 1000 for x in results], 90)
        s[f"{k}_rtf"] = pct([x[f"{k}_rtf"] for x in results], 50)
    s["nemotron_chunk_ms_p50"] = pct([c * 1000 for c in chunks], 50)
    s["nemotron_chunk_ms_p95"] = pct([c * 1000 for c in chunks], 95)
    s["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as f:
        json.dump({"summary": s, "results": results}, f, indent=1)
    print(f"[{args.name}] n={s['n']}  WER parakeet {s['wer_parakeet'] * 100:.1f}%  nemotron-160ms "
          f"{s['wer_nemotron_160ms'] * 100:.1f}%" + (f"  nemotron-offline {s['wer_nemotron_offline'] * 100:.1f}%"
                                                    if "wer_nemotron_offline" in s else "")
          + f" | ready after speech ends (p50/p90): parakeet {s['parakeet_ready_ms_p50']:.0f}/"
          f"{s['parakeet_ready_ms_p90']:.0f} ms, nemotron {s['nemotron_ready_ms_p50']:.0f}/{s['nemotron_ready_ms_p90']:.0f} ms"
          f" | compute/real time: parakeet {s['parakeet_rtf']:.3f}, nemotron {s['nemotron_rtf']:.3f}"
          f" | nemotron chunk {s['nemotron_chunk_ms_p50']:.0f}/{s['nemotron_chunk_ms_p95']:.0f} ms (p50/p95)", flush=True)


if __name__ == "__main__":
    main()
