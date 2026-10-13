"""A small local Qwen3-TTS server for the voice agent (runtime.tts_url), run in its own Python environment
(qwen-tts pins transformers==4.57.3, so it is kept apart from the model stack). Standalone: only qwen-tts,
torch and numpy are needed.

    python qwen_tts_server.py --device cuda:1 --speaker Ryan --port 7790

POST /tts {"text": "...", "speaker"?: "...", "language"?: "...", "instruct"?: "..."}
    -> raw int16 mono PCM, sample rate in the X-Sample-Rate header
GET  /health -> {"ok": true, "model": ..., "speaker": ..., "sample_rate": ..., "dtype": ...}

Guards: generation is capped at ~2.5x the text's expected spoken length, and a phrase that still comes out far
too long (a runaway) is regenerated once, then cut. On a T4 this model is slow (no bf16 / flash-attention):
use it on L4 / L40S / A100-class GPUs; on a T4 use kokoro_tts_server.py.

dtype auto: bf16 where the GPU supports it, else fp16; if fp16 produces non-finite or silent audio on the
warm-up sentence (possible on a T4), the model is reloaded in fp32.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch
from tts_http import expected_seconds, fade_out, serve, trim_silence

CODEC_HZ = 12.5  # Qwen3-TTS-12Hz: speech tokens per second of audio


def load(model_id: str, device: str, dtype: torch.dtype):
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype, attn_implementation="sdpa")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--speaker", default="Ryan", help="a built-in voice: Ryan, Aiden (English), Vivian, Serena, ...")
    p.add_argument("--language", default="English")
    p.add_argument("--instruct", default="", help="optional style instruction (models that support it)")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--keep-silence", action="store_true", help="do not trim the silence around each phrase")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7790)
    args = p.parse_args()

    dev_ok_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    names = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = names[args.dtype] if args.dtype != "auto" else (torch.bfloat16 if dev_ok_bf16 else torch.float16)
    lock = threading.Lock()
    state = {}

    def generate(kw: dict, max_new: int):
        try:
            return state["model"].generate_custom_voice(**kw, max_new_tokens=max_new)
        except TypeError:  # an older qwen-tts without generate kwargs: no cap (the length check below still applies)
            return state["model"].generate_custom_voice(**kw)

    def synth(text: str, req: dict | None = None):
        """An autoregressive TTS can run on past the text (babble / "crying"): the generation is capped at
        ~2.5x the expected length, and audio far longer than the text needs is regenerated once, then cut."""
        req = req or {}
        kw = dict(text=text, language=req.get("language") or args.language, speaker=req.get("speaker") or args.speaker)
        if req.get("instruct") or args.instruct:
            kw["instruct"] = req.get("instruct") or args.instruct
        limit = 2.5 * expected_seconds(text) + 1.0
        for attempt in range(2):
            with lock, torch.inference_mode():
                wavs, sr = generate(kw, int(CODEC_HZ * limit) + 4)
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            sr = int(sr)
            wav = wav if args.keep_silence else trim_silence(wav, sr)
            if len(wav) <= limit * sr:
                return wav, sr
            print(f"[tts] runaway: {len(wav) / sr:.1f}s for {text[:50]!r} (limit {limit:.1f}s)"
                  + (", regenerating" if attempt == 0 else ", cut"), flush=True)
        return fade_out(wav[: int(limit * sr)], sr), sr

    def warm(dt) -> bool:
        state["model"], state["dtype"] = load(args.model, args.device, dt), str(dt).replace("torch.", "")
        t0 = time.time()
        wav, sr = synth("Hello, welcome. How can I help you today?")
        state["sr"] = sr
        good = bool(np.isfinite(wav).all()) and float(np.abs(wav).max()) > 1e-3 and len(wav) > sr // 4
        print(f"[tts] {args.model} on {args.device} ({state['dtype']}): warm-up {time.time() - t0:.2f}s for "
              f"{len(wav) / sr:.2f}s of audio at {sr} Hz -> {'ok' if good else 'BAD (non-finite or silent)'}", flush=True)
        return good

    if not warm(dtype) and dtype != torch.float32:
        print("[tts] reloading in fp32", flush=True)
        state.pop("model", None)
        torch.cuda.empty_cache()
        if not warm(torch.float32):
            raise SystemExit("[tts] the model produces no usable audio")

    serve(synth, lambda: {"model": args.model, "speaker": args.speaker, "sample_rate": state["sr"], "dtype": state["dtype"]},
          args.host, args.port)


if __name__ == "__main__":
    main()
