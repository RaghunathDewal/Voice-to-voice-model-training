"""Input-audio augmentation: make clean training speech sound like a real microphone.

Applied to the speech INPUT only (before Mimi encoding in extract_mimi), never
to talker targets. Deterministic per utterance id, so re-running extraction
gives the same features. Needs only numpy/scipy (no noise corpus download):

  * room reverb   synthetic impulse response (exponentially decaying noise)
  * microphone    band limiting (phone 300-3400 Hz, cheap laptop or phone mic)
  * noise         white / pink / brown / babble-like noise at 0-30 dB SNR
  * level         -20..+3 dB gain, occasional soft clipping
  * phone line    (p_phone, off by default) 8 kHz telephone band + 8-bit mu-law codec, as on a phone call

unintelligible(): speech buried in loud babble / noise, or noise only, for "sorry, I didn't catch that" rows.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy import signal


def _colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
    spec = spec / (f ** (0.5 if kind == "pink" else 1.0))
    return np.fft.irfft(spec, n)


def _babble(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-shaped, syllable-rate modulated noise: sounds like distant chatter."""
    out = np.zeros(n)
    for _ in range(rng.integers(3, 7)):
        base = _colored_noise(n, "pink", rng)
        b, a = signal.butter(2, [200 / (sr / 2), 3000 / (sr / 2)], btype="band")
        base = signal.lfilter(b, a, base)
        t = np.arange(n) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.5, 5.0) * t + rng.uniform(0, 6.28))
        out += base * env
    return out


def _rir(sr: int, rng: np.random.Generator) -> np.ndarray:
    rt60 = rng.uniform(0.15, 0.7)
    n = int(sr * rt60 * 1.2)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[: int(0.002 * sr)] = 0.0
    ir[0] = 1.0  # direct path
    return ir / np.sqrt(np.sum(ir ** 2))


def _snr_mix(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = np.mean(wav ** 2) + 1e-10
    pn = np.mean(noise ** 2) + 1e-10
    return wav + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def phone_line(x: np.ndarray, sr: int) -> np.ndarray:
    """Narrowband telephone channel: down to 8 kHz, G.711-style 8-bit mu-law, back up to `sr`."""
    from math import gcd

    g = gcd(8000, sr)
    y = signal.resample_poly(x, 8000 // g, sr // g)
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    mu = 255.0
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)       # compress
    q = np.round((q + 1) * 127.5) / 127.5 - 1                       # 8-bit codes
    y = np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu        # expand
    return signal.resample_poly(y * peak, sr // g, 8000 // g)[: len(x)]


def unintelligible(wav: np.ndarray, sr: int, key: str) -> np.ndarray:
    """Audio nobody could transcribe: speech under loud chatter/noise (-12..-4 dB SNR), or noise alone."""
    rng = np.random.default_rng(zlib.crc32(("unclear/" + key).encode()))
    x = wav.astype(np.float64)
    kind = rng.choice(["white", "pink", "brown", "babble"])
    noise = _babble(len(x), sr, rng) if kind == "babble" else _colored_noise(len(x), kind, rng)
    if rng.random() < 0.3:  # no speech at all, only the room
        x = noise
    else:
        x = _snr_mix(x, noise, rng.uniform(-12, -4))
        if rng.random() < 0.5:
            x = phone_line(x, sr)
    x = x / (np.max(np.abs(x)) + 1e-9) * 10 ** (rng.uniform(-20, 0) / 20) * 0.9
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def augment(wav: np.ndarray, sr: int, key: str, p_reverb: float = 0.4, p_band: float = 0.35,
            p_noise: float = 0.7, p_clip: float = 0.05, p_phone: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    x = wav.astype(np.float64)
    if len(x) == 0:
        return wav
    if rng.random() < p_reverb:
        wet = signal.fftconvolve(x, _rir(sr, rng))[: len(x)]
        mix = rng.uniform(0.3, 1.0)
        x = (1 - mix) * x + mix * wet
    if rng.random() < p_band:
        if rng.random() < 0.4:  # telephone
            lo, hi = 300.0, 3400.0
        else:                   # laptop / phone mic
            lo, hi = rng.uniform(80, 250), rng.uniform(4000, min(8000, sr / 2 - 100))
        sos = signal.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
        x = signal.sosfilt(sos, x)
    if rng.random() < p_noise:
        kind = rng.choice(["white", "pink", "brown", "babble"], p=[0.2, 0.35, 0.2, 0.25])
        noise = _babble(len(x), sr, rng) if kind == "babble" else _colored_noise(len(x), kind, rng)
        x = _snr_mix(x, noise, rng.uniform(0, 30))
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak * 10 ** (rng.uniform(-20, 3) / 20) * 0.9
    if rng.random() < p_clip:
        x = np.tanh(3 * x) / np.tanh(3)
    if p_phone > 0 and rng.random() < p_phone:  # drawn last: with p_phone=0 the result is as before
        x = phone_line(x, sr)
    return np.clip(x, -1.0, 1.0).astype(np.float32)
