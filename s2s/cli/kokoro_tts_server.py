"""A small local Kokoro-82M TTS server for the voice agent (runtime.tts_url), run in its own Python environment.
Kokoro is not autoregressive: the length of the audio is fixed by the text, so it cannot run on / babble, and
it is fast even on a T4 (tens of ms per phrase). Apache-2.0, 24 kHz. Needs kokoro, misaki[en], torch, numpy
and espeak-ng (apt) for words outside its dictionary.

    python kokoro_tts_server.py --device cuda:1 --voice af_heart --port 7791

POST /tts {"text": "...", "voice"?: "...", "speed"?: 1.0} -> raw int16 mono PCM, sample rate in X-Sample-Rate
GET  /health

Voices: af_heart, af_bella, af_nicole (US female), am_michael, am_fenrir (US male), bf_emma (UK female),
bm_george (UK male), ... (the first letter is the language: a = US English, b = UK English).
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch
from tts_http import serve, trim_silence

SR = 24000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--keep-silence", action="store_true", help="do not trim the silence around each phrase")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7791)
    p.add_argument("--no-cudnn", action="store_true", help="plain PyTorch convolutions (AMD: skips MIOpen, whose "
                   "kernel search for every new phrase length can take seconds)")
    args = p.parse_args()
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False

    from kokoro import KModel, KPipeline

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(args.device).eval()
    pipes: dict[str, KPipeline] = {}
    lock = threading.Lock()

    def pipe(voice: str) -> KPipeline:
        lang = voice[0]
        if lang not in pipes:
            pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model)
        return pipes[lang]

    def synth(text: str, req: dict) -> tuple[np.ndarray, int]:
        voice, speed = str(req.get("voice") or args.voice), float(req.get("speed") or args.speed)
        chunks = []
        with lock, torch.inference_mode():
            for result in pipe(voice)(text, voice=voice, speed=speed):
                audio = result.audio if hasattr(result, "audio") else result[2]
                if audio is not None:
                    chunks.append(audio.detach().float().cpu().numpy().reshape(-1))
        wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return (wav if args.keep_silence else trim_silence(wav, SR)), SR

    for t in ("Hello, welcome. How can I help you today?", "Sure, one moment."):  # load voices, warm kernels
        t0 = time.time()
        wav, _ = synth(t, {})
        print(f"[tts] Kokoro-82M on {args.device}, voice {args.voice}: warm-up {(time.time() - t0) * 1000:.0f} ms "
              f"for {len(wav) / SR:.2f}s of audio", flush=True)
    if not len(wav) or not np.isfinite(wav).all():
        raise SystemExit("[tts] Kokoro produced no usable audio")
    serve(synth, lambda: {"model": "hexgrad/Kokoro-82M", "voice": args.voice, "sample_rate": SR}, args.host, args.port)


if __name__ == "__main__":
    main()
