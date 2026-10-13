"""Shared by the local TTS servers (qwen_tts_server.py, kokoro_tts_server.py), which run in their own Python
environments: stdlib + numpy only, imported as a sibling module (the script's folder is on sys.path).

POST /tts {"text": "...", ...} -> raw int16 mono PCM, sample rate in the X-Sample-Rate header
GET  /health -> {"ok": true, ...}
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import numpy as np


def trim_silence(wav: np.ndarray, sr: int, head_s: float = 0.08, tail_s: float = 0.15) -> np.ndarray:
    """Cut leading / trailing silence, keeping a short natural pause, so phrases spoken back to back do not
    leave long gaps."""
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


def expected_seconds(text: str) -> float:
    """Rough spoken length of a phrase (~14 characters per second of normal speech)."""
    return max(1.0, 0.07 * len(text))


def fade_out(wav: np.ndarray, sr: int, seconds: float = 0.05) -> np.ndarray:
    n = min(len(wav), int(seconds * sr))
    if n:
        wav = wav.copy()
        wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=wav.dtype)
    return wav


def serve(synth: Callable[[str, dict], tuple[np.ndarray, int]], health: Callable[[], dict], host: str, port: int) -> None:
    """synth(text, request) -> (float wav, sample rate); health() -> the /health JSON."""

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
                self._send(200, json.dumps({"ok": True, **health()}).encode(), "application/json")
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
                wav, sr = synth(text, req)
                pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
                took = time.time() - t0
                print(f"[tts] {took * 1000:.0f} ms for {len(wav) / sr:.2f}s: {text[:70]!r}", flush=True)
                self._send(200, pcm, "application/octet-stream", {"X-Sample-Rate": sr, "X-Synth-Ms": round(took * 1000)})
            except Exception as e:  # noqa: BLE001 - report, keep serving
                print(f"[tts] error {e!r} for {text[:70]!r}", flush=True)
                self._send(500, repr(e).encode(), "text/plain")

    print(f"[tts] serving on http://{host}:{port}/tts", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
