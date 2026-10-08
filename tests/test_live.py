import numpy as np

from s2s.runtime.endpoint import Endpointer
from s2s.runtime.live import LiveListener

SR, HOP = 24000, 1920


def tone(seconds, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def make(eot=0.0):
    calls = []

    def eot_fn(wav):
        calls.append(len(wav))
        return eot

    ep = Endpointer(frame_ms=80, energy_threshold_db=-45, min_speech_ms=240, min_silence_ms=240,
                    max_silence_ms=800, eot_threshold=0.5)
    return LiveListener(ep, HOP, eot_fn, sample_rate=SR), calls


def feed_all(listener, wav, chunk=int(0.24 * SR)):
    out = []
    for i in range(0, len(wav), chunk):
        u = listener.feed(wav[i:i + chunk])
        if u is not None:
            out.append(u)
    return out


def test_leading_silence_dropped_and_turn_ends_on_eot_head():
    listener, calls = make(eot=0.9)
    utts = feed_all(listener, np.concatenate([silence(3.0), tone(1.0), silence(1.0)]))
    assert len(utts) == 1 and listener.reason == "eot_head"
    # pre-roll (4 frames) + 1 s speech + ~240 ms silence; the 3 s of leading silence is not kept
    assert len(utts[0]) / SR < 1.0 + 0.32 + 0.4
    assert calls, "EOT head consulted"


def test_max_silence_without_eot_agreement():
    listener, _ = make(eot=0.1)
    utts = feed_all(listener, np.concatenate([tone(1.0), silence(1.2)]))
    assert len(utts) == 1 and listener.reason == "max_silence"


def test_short_noise_is_discarded_and_next_turn_detected():
    listener, calls = make(eot=0.9)
    utts = feed_all(listener, np.concatenate([tone(0.08), silence(1.0)]))
    assert utts == [] and not listener.in_turn and not calls
    utts = feed_all(listener, np.concatenate([tone(0.8), silence(0.6)]))
    assert len(utts) == 1


def test_two_turns_in_one_stream():
    listener, _ = make(eot=0.9)
    utts = feed_all(listener, np.concatenate([tone(0.8), silence(0.6), tone(0.8), silence(0.6)]))
    assert len(utts) == 2


# ------------------------------------------------------------- Silero VAD
def _speech_clip():
    import os

    import pytest
    import soundfile as sf

    from s2s.audio import resample

    path = os.path.join(os.path.dirname(__file__), "..", "demo", "video", "public", "audio", "guest_1.wav")
    pytest.importorskip("silero_vad")
    if not os.path.exists(path):
        pytest.skip("demo speech clip not available")
    wav, sr = sf.read(path, dtype="float32")
    return resample(wav, sr, SR)


def _room_noise(seconds, seed=0):
    """Fan-like noise at about -38 dB with keyboard-like clicks: louder than the energy VAD's -45 dB."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    noise = np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same").astype(np.float32) * 0.035
    for at in rng.integers(0, n - 200, size=int(seconds * 3)):
        noise[at:at + 120] += rng.standard_normal(120).astype(np.float32) * 0.5
    return noise


def _speech_frames(ep, wav):
    return sum(ep.is_speech(wav[i:i + HOP]) for i in range(0, len(wav) - HOP + 1, HOP))


def test_silero_ignores_room_noise_and_clicks_that_fool_the_energy_vad():
    _speech_clip()  # skips when silero-vad is not installed
    noise = _room_noise(6.0)
    energy = Endpointer(frame_ms=80, energy_threshold_db=-45)
    silero = Endpointer(frame_ms=80, vad="silero", sample_rate=SR)
    n = len(noise) // HOP
    assert _speech_frames(energy, noise) > 0.8 * n          # loudness alone: noise looks like speech
    assert _speech_frames(silero, noise) < 0.1 * n          # Silero: not speech


def test_silero_turn_in_a_noisy_room_ends_after_the_speech():
    speech = _speech_clip()
    lead, tail = _room_noise(1.0, seed=1), _room_noise(2.0, seed=2)
    wav = np.concatenate([lead, speech + _room_noise(len(speech) / SR, seed=3)[: len(speech)], tail])
    ep = Endpointer(frame_ms=80, min_speech_ms=240, min_silence_ms=320, max_silence_ms=1000,
                    eot_threshold=0.5, vad="silero", sample_rate=SR)
    listener = LiveListener(ep, HOP, lambda w: 0.9, sample_rate=SR)
    turns = feed_all(listener, wav)
    assert len(turns) == 1
    sec = len(turns[0]) / SR
    voiced = _speech_frames(Endpointer(frame_ms=80, vad="silero", sample_rate=SR), speech) * HOP / SR
    # the whole spoken part plus pre-roll and the closing silence, not the 2 s of noise after it
    assert voiced <= sec < voiced + 1.5, (sec, voiced)
