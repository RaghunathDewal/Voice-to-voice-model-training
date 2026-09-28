"""End-of-turn detection: energy VAD + the adapter's end-of-turn head.

Rules (per 80 ms frame):
  * nothing happens until `min_speech_ms` of speech has been heard
  * after `min_silence_ms` of silence, end the turn if the EOT head agrees (p >= threshold)
  * after `max_silence_ms` of silence, end the turn regardless
"""

from __future__ import annotations

import numpy as np

from s2s.audio import rms_db


class Endpointer:
    def __init__(self, frame_ms: float = 80.0, energy_threshold_db: float = -45.0, min_speech_ms: float = 240,
                 min_silence_ms: float = 240, max_silence_ms: float = 800, eot_threshold: float = 0.5):
        self.frame_ms = frame_ms
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_silence_ms = max_silence_ms
        self.eot_threshold = eot_threshold
        self.reset()

    @classmethod
    def from_config(cls, cfg, frame_ms: float) -> "Endpointer":
        return cls(frame_ms=frame_ms, energy_threshold_db=cfg.energy_threshold_db, min_speech_ms=cfg.min_speech_ms,
                   min_silence_ms=cfg.min_silence_ms, max_silence_ms=cfg.max_silence_ms,
                   eot_threshold=cfg.eot_threshold)

    def reset(self) -> None:
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.reason: str | None = None

    def is_speech(self, frame: np.ndarray) -> bool:
        return rms_db(frame) > self.energy_threshold_db

    def update(self, frame: np.ndarray, eot_prob: float) -> bool:
        if self.is_speech(frame):
            self.speech_ms += self.frame_ms
            self.silence_ms = 0.0
        else:
            self.silence_ms += self.frame_ms
        if self.speech_ms < self.min_speech_ms:
            return False
        if self.silence_ms >= self.max_silence_ms:
            self.reason = "max_silence"
            return True
        if self.silence_ms >= self.min_silence_ms and eot_prob >= self.eot_threshold:
            self.reason = "eot_head"
            return True
        return False
