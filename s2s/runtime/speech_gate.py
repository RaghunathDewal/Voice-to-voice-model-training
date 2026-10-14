"""Speech gate for the full-duplex demo: the model hears the guest only while someone is speaking.

Why: the duplex model learned "nobody is talking" from clean synthetic silence. A real microphone in silence
carries fan hum, clicks, breathing and room echo; the listener turns that into vectors the model never saw in
silence, and it answers turns nobody took ("Hello Chris!", "You're welcome"). The gate replaces the audio
outside speech with digital silence, which is what training looked like.

How: Silero VAD (32 ms windows at 16 kHz) with hysteresis, a short look-ahead (the audio is delayed by
`lookahead_ms`, so a word's start is not cut while the VAD decides) and a hang-over (short pauses inside a
sentence stay open). Falls back to a loudness threshold if silero-vad is not installed.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from s2s.runtime.endpoint import SileroVAD, rms_db

WINDOW = 512  # 32 ms at 16 kHz


class SpeechGate:
    def __init__(self, threshold: float = 0.5, hangover_ms: float = 400, lookahead_ms: float = 96,
                 min_level_db: float = -60.0, energy_db: float = -42.0):
        self.threshold, self.min_level_db, self.energy_db = threshold, min_level_db, energy_db
        self.hang = max(1, int(round(hangover_ms / 32)))
        self.ahead = max(0, int(round(lookahead_ms / 32)))
        try:
            self.vad = SileroVAD(16000)
        except ImportError:
            print("WARNING: silero-vad is not installed (pip install silero-vad); speech gate uses loudness", flush=True)
            self.vad = None
        self.reset()

    @property
    def name(self) -> str:
        return "silero" if self.vad is not None else "energy"

    def reset(self) -> None:
        if self.vad is not None:
            self.vad.reset()
        self.buf = np.zeros(0, np.float32)
        self.queue: deque = deque()  # [window, keep] waiting for the look-ahead
        self.active, self.quiet = False, 0
        self.ended = 0  # speech segments that have closed (the caller watches this to finish a guest line)

    def _speech(self, w: np.ndarray) -> bool:
        if self.vad is None:
            return rms_db(w) > self.energy_db
        p = self.vad.prob(w)
        on = p >= (self.threshold - 0.15 if self.active else self.threshold)
        return on and rms_db(w) > self.min_level_db

    def push(self, x16: np.ndarray) -> np.ndarray:
        """16 kHz audio in, the same length (minus what waits in the look-ahead) out, silenced outside speech."""
        self.buf = np.concatenate([self.buf, np.asarray(x16, np.float32)])
        out = []
        while len(self.buf) >= WINDOW:
            w, self.buf = self.buf[:WINDOW], self.buf[WINDOW:]
            if self._speech(w):
                if not self.active:  # a word starts: open the windows still waiting too (its first syllable)
                    for item in self.queue:
                        item[1] = True
                self.active, self.quiet = True, 0
            elif self.active:
                self.quiet += 1
                if self.quiet > self.hang:
                    self.active = False
                    self.ended += 1
            self.queue.append([w, self.active])
            while len(self.queue) > self.ahead:
                win, keep = self.queue.popleft()
                out.append(win if keep else np.zeros_like(win))
        return np.concatenate(out) if out else np.zeros(0, np.float32)
