"""A small local Qwen3-TTS server for the voice agent (runtime.tts_url), run in its own Python environment
(qwen-tts pins transformers==4.57.3, so it is kept apart from the model stack). Standalone: only qwen-tts,
torch and numpy are needed.

    python qwen_tts_server.py --device cuda:1 --speaker Ryan --port 7790

POST /tts {"text": "...", "speaker"?: "...", "language"?: "...", "instruct"?: "..."}
    -> raw int16 mono PCM, sample rate in the X-Sample-Rate header
GET  /health -> {"ok": true, "model": ..., "speaker": ..., "sample_rate": ..., "dtype": ...}

dtype auto: bf16 where the GPU supports it, else fp16; if fp16 produces non-finite or silent audio on the
warm-up sentence (possible on a T4), the model is reloaded in fp32.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch


def trim_silence(wav: np.ndarray, sr: int, head_s: float = 0.08, tail_s: float = 0.15) -> np.ndarray:
    """Cut the model's leading / trailing silence (it pads each phrase by ~0.5-1.3 s), keeping a short natural
    pause, so phrases spoken back to back do not leave long gaps."""
    if len(wav) == 0:
        return wav
    hop = max(1, sr // 100)  # 10 ms frames
    n = len(wav) // hop
    if n == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n * hop].reshape(n, hop) ** 2, axis=1))
    voiced = np.nonzero(rms > max(1e-3, 0.03 * float(rms.max())))[0]
    if len(voiced) == 0:
        return wav
    start = max(0, voiced[0] * hop - int(head_s * sr))
    end = min(len(wav), (voiced[-1] + 1) * hop + int(tail_s * sr))
    return wav[start:end]


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

    def synth(text: str, speaker: str | None = None, language: str | None = None, instruct: str | None = None):
        kw = dict(text=text, language=language or args.language, speaker=speaker or args.speaker)
        if instruct or args.instruct:
            kw["instruct"] = instruct or args.instruct
        with lock, torch.inference_mode():
            wavs, sr = state["model"].generate_custom_voice(**kw)
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return (wav if args.keep_silence else trim_silence(wav, int(sr))), int(sr)

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

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                info = {"ok": True, "model": args.model, "speaker": args.speaker, "sample_rate": state["sr"],
                        "dtype": state["dtype"]}
                self._send(200, json.dumps(info).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/tts"):
                self._send(404, b"not found", "text/plain")
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text = str(req.get("text", "")).strip()
                if not text:
                    self._send(400, b"empty text", "text/plain")
                    return
                t0 = time.time()
                wav, sr = synth(text, req.get("speaker"), req.get("language"), req.get("instruct"))
                pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
                took = time.time() - t0
                print(f"[tts] {took * 1000:.0f} ms for {len(wav) / sr:.2f}s: {text[:70]!r}", flush=True)
                self._send(200, pcm, "application/octet-stream", {"X-Sample-Rate": sr, "X-Synth-Ms": round(took * 1000)})
            except Exception as e:  # noqa: BLE001 - report, keep serving
                self._send(500, repr(e).encode(), "text/plain")

    print(f"[tts] serving on http://{args.host}:{args.port}/tts", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
