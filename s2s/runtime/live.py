"""Hands-free turn detection for a live microphone.

`LiveListener` receives microphone audio in chunks of any size and returns a
complete user utterance once the end of the turn is detected, with the same
rules as the streaming runtime (Silero or energy VAD + the adapter's end-of-turn head,
see s2s/runtime/endpoint.py):

    listener = LiveListener(agent)
    for chunk in mic:                              # 24 kHz float32
        utterance = listener.feed(chunk)
        if utterance is not None:
            for event in session.respond(utterance): ...

Audio before speech starts is not kept (except a short pre-roll), and noise
bursts shorter than `min_speech_ms` are discarded without touching the
thinker, so the session's KV cache only ever sees real turns.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from s2s.runtime.endpoint import Endpointer


class LiveListener:
    def __init__(self, endpointer: Endpointer, hop: int, eot_fn: Callable[[np.ndarray], float],
                 preroll_frames: int = 4, max_turn_s: float = 30.0, sample_rate: int = 24000):
        """`eot_fn(wav)` returns the end-of-turn probability at the last frame of `wav`."""
        self.ep = endpointer
        self.hop = hop
        self.eot_fn = eot_fn
        self.max_frames = int(max_turn_s * sample_rate / hop)
        self.preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self.remainder = np.zeros(0, dtype=np.float32)
        self.frames: list[np.ndarray] | None = None  # None = idle (no speech yet)
        self.reason: str | None = None

    @classmethod
    def for_agent(cls, agent, **kw) -> "LiveListener":
        codec = agent.codec
        ep = Endpointer.from_config(agent.cfg.runtime.endpoint, 1000.0 / codec.frame_rate, codec.sample_rate)

        def eot_fn(wav: np.ndarray) -> float:
            return agent.eot_probability(wav)  # encodes with the adapter's input encoder (Mimi, Parakeet, ...)

        return cls(ep, codec.hop, eot_fn, sample_rate=codec.sample_rate, **kw)

    @property
    def in_turn(self) -> bool:
        return self.frames is not None

    def reset(self) -> None:
        self.preroll.clear()
        self.remainder = np.zeros(0, dtype=np.float32)
        self.frames = None
        self.ep.reset()
        self.ep.reset_vad()  # listening resumes after the agent spoke: a new audio stream

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed 24 kHz mono float32 audio. Returns the utterance when the turn ends, else None."""
        buf = np.concatenate([self.remainder, np.asarray(chunk, dtype=np.float32)])
        n = len(buf) // self.hop
        self.remainder = buf[n * self.hop:]
        for i in range(n):
            done = self._frame(buf[i * self.hop:(i + 1) * self.hop])
            if done is not None:
                self.remainder = np.zeros(0, dtype=np.float32)  # rest of the chunk is after the turn
                return done
        return None

    def _frame(self, frame: np.ndarray) -> np.ndarray | None:
        ep = self.ep
        if self.frames is None:
            if not ep.is_speech(frame):
                self.preroll.append(frame)
                return None
            self.frames = list(self.preroll)
            self.preroll.clear()
            ep.reset()
        self.frames.append(frame)
        silent = not ep.is_speech(frame)
        # the EOT head is only consulted once enough silence has passed (it costs a Mimi encode)
        eot = 0.0
        if silent and ep.speech_ms >= ep.min_speech_ms and ep.silence_ms + ep.frame_ms >= ep.min_silence_ms:
            eot = self.eot_fn(np.concatenate(self.frames))
        if ep.update(frame, eot):
            return self._finish(ep.reason)
        if ep.speech_ms < ep.min_speech_ms and ep.silence_ms >= ep.max_silence_ms:
            self.frames = None  # a click or cough, not a turn
            ep.reset()
            return None
        if len(self.frames) >= self.max_frames:
            return self._finish("max_length")
        return None

    def _finish(self, reason: str | None) -> np.ndarray:
        wav = np.concatenate(self.frames)
        self.reason = reason
        self.frames = None
        self.ep.reset()
        return wav
