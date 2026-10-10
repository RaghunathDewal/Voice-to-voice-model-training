"""Transcribe a raw manifest with NVIDIA Nemotron streaming ASR, at several streaming chunk sizes.

The audio is prepared exactly as s2s.prep.extract_mimi prepares it for the adapter (same trailing silence;
with --noisy-name, the same s2s.augment noise/echo/phone-band as the `<noisy-name>` feature manifest), so
the transcripts can be compared row by row with what our adapter heard.

    python -m s2s.eval.asr_nemotron --config configs/qwen4b.yaml --manifest data/manifests/slurp_test_raw.jsonl \
        --max 150 --out outputs/asr/slurp.jsonl
    # same clips with the noise used for slurp_noisy_pk
    python -m s2s.eval.asr_nemotron ... --noisy-name slurp_noisy_pk --out outputs/asr/slurp_noisy.jsonl

Needs transformers >= 5.13. Chunk sizes: 160, 560, 1120 ms (80 ms fails in transformers' streaming
processor: its first chunk is shorter than one FFT window). Writes one row per clip:
{"id", "text", "hyp": {"160": ..., "560": ..., "1120": ..., "offline": ...}} and prints the WER per setting.
"""

from __future__ import annotations

import json
import os
import zlib

import numpy as np
import torch
from tqdm import tqdm

from s2s.audio import add_silence, load_audio
from s2s.augment import augment
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.eval.spoken_wer import spoken_wer
from s2s.utils import resolve_device

LOOKAHEAD = {160: 1, 560: 6, 1120: 13}  # chunk ms -> lookahead frames (80 ms each), from the model card
SR = 16000


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--manifest", required=True, help="raw manifest (audio paths), e.g. *_raw.jsonl")
    p.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    p.add_argument("--chunks", default="160,560,1120", help="streaming chunk sizes in ms")
    p.add_argument("--no-offline", action="store_true", help="skip the full-context (non-streaming) pass")
    p.add_argument("--noisy-name", default=None, help="add the noise s2s.prep.extract_mimi used for this manifest name")
    p.add_argument("--max", type=int, default=150)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    from transformers import AutoModelForRNNT, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForRNNT.from_pretrained(args.model).to(device).eval()
    chunks = [int(c) for c in args.chunks.split(",") if c]
    silence = float(cfg.codec.trailing_silence_s)
    hop, nfft = proc.feature_extractor.hop_length, proc.feature_extractor.n_fft

    def to_dev(batch):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    @torch.no_grad()
    def offline(wav: np.ndarray) -> str:
        out = model.generate(**to_dev(proc(wav, sampling_rate=SR, return_tensors="pt")), return_dict_in_generate=True)
        return proc.decode(out.sequences, skip_special_tokens=True)[0]

    @torch.no_grad()
    def stream(wav: np.ndarray, chunk_ms: int) -> str:
        proc.set_num_lookahead_tokens(LOOKAHEAD[chunk_ms])
        wav = np.concatenate([wav, np.zeros(2 * SR, np.float32)])  # let the last words leave the lookahead
        first = to_dev(proc(wav[: proc.num_samples_first_audio_chunk], sampling_rate=SR, is_streaming=True,
                            is_first_audio_chunk=True, return_tensors="pt"))

        def chunks_of_audio():
            yield first["input_features"][:, : proc.num_mel_frames_first_audio_chunk, :]
            mel = proc.num_mel_frames_first_audio_chunk
            start = mel * hop - nfft // 2
            while (end := start + proc.num_samples_per_audio_chunk) < wav.shape[0]:
                yield proc(wav[start:end], sampling_rate=SR, is_streaming=True, is_first_audio_chunk=False,
                           return_tensors="pt").input_features.to(device)
                mel += proc.num_mel_frames_per_audio_chunk
                start = mel * hop - nfft // 2

        out = model.generate(**{**first, "input_features": chunks_of_audio()}, return_dict_in_generate=True)
        return proc.decode(out.sequences, skip_special_tokens=True)[0]

    rows = [r for r in load_manifest(args.manifest) if r.get("text")][: args.max or None]
    results = []
    for r in tqdm(rows, desc="nemotron", mininterval=30):
        try:
            wav = load_audio(r["audio"], SR)
        except Exception as e:  # noqa: BLE001
            print(f"skip {r.get('id')}: {e}")
            continue
        peak = float(np.abs(wav).max()) if len(wav) else 0.0
        if peak > 1.0:
            wav = wav / peak
        if silence:
            wav = add_silence(wav, SR, silence, seed=zlib.crc32(r["id"].encode()))
        if args.noisy_name:
            wav = augment(wav, SR, f"{args.noisy_name}/{r['id']}")
        hyp = {str(c): stream(wav, c) for c in chunks}
        if not args.no_offline:
            hyp["offline"] = offline(wav)
        results.append({"id": r["id"], "text": r["text"], "hyp": hyp})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    refs = [x["text"] for x in results]
    keys = list(results[0]["hyp"]) if results else []
    print(f"[{os.path.basename(args.out)}] n={len(results)}  " + "  ".join(
        f"WER@{k}{'ms' if k != 'offline' else ''} {spoken_wer(refs, [x['hyp'][k] for x in results]) * 100:.1f}%"
        for k in keys), flush=True)


if __name__ == "__main__":
    main()
