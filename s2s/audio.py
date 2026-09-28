"""Audio I/O and resampling (soundfile + scipy; no torchaudio dependency)."""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return resample_poly(wav, sr_out // g, sr_in // g).astype(np.float32)


def to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    return resample(to_mono(wav), sr, target_sr)


def save_audio(path: str, wav: np.ndarray, sr: int) -> None:
    sf.write(path, np.clip(wav, -1.0, 1.0).astype(np.float32), sr)


def rms_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def add_silence(wav: np.ndarray, sr: int, seconds: float, noise_db: float = -70.0, seed: int | None = None) -> np.ndarray:
    """Append `seconds` of near-silence (very low level noise, like a real room)."""
    n = int(round(seconds * sr))
    if n <= 0:
        return wav
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32) * (10 ** (noise_db / 20.0))
    return np.concatenate([wav.astype(np.float32), noise])
