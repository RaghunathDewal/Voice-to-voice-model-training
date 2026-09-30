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
