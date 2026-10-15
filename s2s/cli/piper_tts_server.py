"""Piper TTS (rhasspy/piper, ONNX on the CPU) behind the same HTTP protocol as kokoro_tts_server.py.

    python s2s/cli/piper_tts_server.py --port 7791 --voice en_US-lessac-medium

A small VITS voice built for CPUs: much faster than Kokoro per phrase, a little less natural. Used where the GPU
voice is slow (AMD) or the GPU is kept for the listener and the 4B. Voices: huggingface.co/rhasspy/piper-voices.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
from tts_http import serve, trim_silence


def voice_files(name: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    lang, region_rest = name.split("_", 1)            # en_US-lessac-medium -> en, US-lessac-medium
    region, speaker, quality = region_rest.split("-")
    base = f"{lang}/{lang}_{region}/{speaker}/{quality}/{name}"
    onnx = hf_hub_download("rhasspy/piper-voices", f"{base}.onnx")
    hf_hub_download("rhasspy/piper-voices", f"{base}.onnx.json")
    return onnx, onnx + ".json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default="en_US-lessac-medium")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7791)
    args = p.parse_args()
    from piper import PiperVoice

    onnx, cfg = voice_files(args.voice)
    voice = PiperVoice.load(onnx, config_path=cfg)
    sr = int(voice.config.sample_rate)
    lock = threading.Lock()

    def synth(text: str, req: dict) -> tuple[np.ndarray, int]:
        with lock:
            out = voice.synthesize(text)
            chunks = []
            for c in out:  # piper-tts >= 1.3: AudioChunk objects
                arr = getattr(c, "audio_float_array", None)
                if arr is None:
                    arr = np.asarray(c.audio_int16_array, np.float32) / 32768.0
                chunks.append(np.asarray(arr, np.float32).reshape(-1))
        wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return trim_silence(wav, sr), sr

    for t in ("Hello, welcome. How can I help you today?", "Sure, one moment."):
        t0 = time.time()
        wav, _ = synth(t, {})
        print(f"[tts] Piper {args.voice}: warm-up {(time.time() - t0) * 1000:.0f} ms for {len(wav) / sr:.2f}s", flush=True)
    if not len(wav) or not np.isfinite(wav).all():
        raise SystemExit("[tts] Piper produced no usable audio")
    serve(synth, lambda: {"model": "piper", "voice": args.voice, "sample_rate": sr}, args.host, args.port)


if __name__ == "__main__":
    main()
