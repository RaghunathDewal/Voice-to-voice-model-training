"""End-of-turn detection: a voice-activity detector + the adapter's end-of-turn head.

Rules (per 80 ms frame):
  * nothing happens until `min_speech_ms` of speech has been heard
  * after `quick_silence_ms` of silence, end the turn if the EOT head is sure (p >= quick_eot_threshold)
  * after `min_silence_ms` of silence, end the turn if the EOT head agrees (p >= threshold)
  * after `max_silence_ms` of silence, end the turn regardless

Voice activity (`vad`):
  * "silero" (default): Silero VAD (MIT, ~2 MB, bundled in the `silero-vad` pip package, CPU). It
    tells speech from fans, traffic, keyboard clicks and room noise, which a loudness threshold
    cannot: in a noisy room the background sits at the threshold and every turn runs on until
    max length. Stateful, so every frame is classified exactly once, in order.
  * "energy": a fixed loudness threshold (`energy_threshold_db`). Fallback when silero-vad is not
    installed; fine only for a quiet room with a close microphone.
"""

from __future__ import annotations

import warnings

import numpy as np

from s2s.audio import resample, rms_db


class SileroVAD:
    """Speech probability per 32 ms window at 16 kHz (Silero VAD v5+), fed with frames at any rate."""

    WINDOW = 512  # samples at 16 kHz

    def __init__(self, sample_rate: int):
        import torch
        from silero_vad import load_silero_vad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = load_silero_vad()
        self.torch = torch
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self.buf = np.zeros(0, dtype=np.float32)
        self.last = 0.0

    def prob(self, frame: np.ndarray) -> float:
        """Highest speech probability among the 32 ms windows completed by this frame."""
        x = resample(np.asarray(frame, dtype=np.float32), self.sample_rate, 16000)
        self.buf = np.concatenate([self.buf, x])
        probs = []
        with self.torch.no_grad():
            while len(self.buf) >= self.WINDOW:
                window = self.torch.from_numpy(np.ascontiguousarray(self.buf[: self.WINDOW]))
                self.buf = self.buf[self.WINDOW:]
                probs.append(float(self.model(window, 16000)))
        if probs:
            self.last = max(probs)
        return self.last


class BatchedSileroVAD:
    """One Silero model shared by many audio streams (a server with many calls): the 32 ms windows of all streams
    go through it as ONE batch, each stream keeping its own recurrent state and context. One window costs ~1 ms of
    CPU, so 64 calls one by one (~2000 windows/s) are more than one CPU thread can do in real time; batched it is
    ~6 ms per 64 windows."""

    WINDOW, CONTEXT = 512, 64  # samples at 16 kHz

    def __init__(self):
        import threading

        import torch
        from silero_vad import load_silero_vad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = load_silero_vad()
        self.torch = torch
        self.lock = threading.Lock()

    def stream(self) -> "VADStream":
        return VADStream(self)

    def run(self, items: list) -> None:
        """items: (stream, 16 kHz frame) in order per stream. Each frame's speech probability is stored on its
        stream (taken by stream.prob(frame)); frames of different streams are classified together."""
        torch = self.torch
        per: dict[int, list] = {}
        for st, frame in items:
            per.setdefault(id(st), [st, st.gen, []])[2].append(frame)
        rounds = max((len(f) for _, _, f in per.values()), default=0)
        with self.lock, torch.no_grad():
            for r in range(rounds):
                active = [(st, gen, f[r]) for st, gen, f in per.values() if r < len(f)]
                got: dict[int, list] = {}
                for st, _, frame in active:
                    st.buf = np.concatenate([st.buf, np.asarray(frame, np.float32)])
                while True:
                    rows = [st for st, _, _ in active if len(st.buf) >= self.WINDOW]
                    if not rows:
                        break
                    x = torch.from_numpy(np.stack([st.buf[: self.WINDOW] for st in rows]))
                    m = self.model
                    m._state = torch.stack([st.state for st in rows], 1)
                    m._context = torch.stack([st.context for st in rows])
                    m._last_sr, m._last_batch_size = 16000, len(rows)
                    p = m(x, 16000).reshape(-1).tolist()
                    for i, st in enumerate(rows):
                        st.state, st.context = m._state[:, i].clone(), m._context[i].clone()
                        st.buf = st.buf[self.WINDOW:]
                        got.setdefault(id(st), []).append(p[i])
                for st, gen, frame in active:
                    if st.gen != gen:
                        continue  # reset meanwhile (a new turn): this result is stale
                    if id(st) in got:
                        st.last = max(got[id(st)])
                    st.pre[id(frame)] = st.last


class VADStream:
    """One stream's Silero state inside a BatchedSileroVAD; same prob()/reset() interface as SileroVAD."""

    def __init__(self, batch: BatchedSileroVAD):
        self.batch = batch
        self.gen = 0
        self.reset()

    def reset(self) -> None:
        torch = self.batch.torch
        self.state, self.context = torch.zeros(2, 128), torch.zeros(BatchedSileroVAD.CONTEXT)
        self.buf = np.zeros(0, dtype=np.float32)
        self.last = 0.0
        self.pre: dict[int, float] = {}
        self.gen += 1

    def prob(self, frame: np.ndarray) -> float:
        if id(frame) not in self.pre:  # not classified in a batch beforehand: alone
            self.batch.run([(self, frame)])
        return self.pre.pop(id(frame), self.last)


class Endpointer:
    def __init__(self, frame_ms: float = 80.0, energy_threshold_db: float = -45.0, min_speech_ms: float = 240,
                 min_silence_ms: float = 240, max_silence_ms: float = 800, eot_threshold: float = 0.5,
                 vad: str = "energy", vad_threshold: float = 0.5, sample_rate: int = 24000,
                 min_level_db: float = -65.0, quick_silence_ms: float = 0.0, quick_eot_threshold: float = 1.01,
                 vad_stream: "VADStream | None" = None):
        self.frame_ms = frame_ms
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_silence_ms = max_silence_ms
        self.eot_threshold = eot_threshold
        self.quick_silence_ms = quick_silence_ms or min_silence_ms  # 0: no quick rule
        self.quick_eot_threshold = quick_eot_threshold
        self.vad_threshold = vad_threshold
        self.min_level_db = min_level_db  # silero: frames quieter than this are silence (digital silence, mic off)
        self.vad = None
        if vad == "silero" and vad_stream is not None:  # shared batched model (batch_live)
            self.vad = vad_stream
        elif vad == "silero":
            try:
                self.vad = SileroVAD(sample_rate)
            except ImportError:
                print("WARNING: silero-vad is not installed (pip install silero-vad); using the energy VAD")
        self._last_frame: np.ndarray | None = None
        self._last_result = False
        self._active = False  # hysteresis state for the silero VAD
        self.reset()

    @classmethod
    def from_config(cls, cfg, frame_ms: float, sample_rate: int = 24000, vad_stream=None) -> "Endpointer":
        return cls(frame_ms=frame_ms, energy_threshold_db=cfg.energy_threshold_db, min_speech_ms=cfg.min_speech_ms,
                   min_silence_ms=cfg.min_silence_ms, max_silence_ms=cfg.max_silence_ms,
                   eot_threshold=cfg.eot_threshold, vad=str(cfg.get("vad", "energy")),
                   vad_threshold=float(cfg.get("vad_threshold", 0.5)), sample_rate=sample_rate,
                   quick_silence_ms=float(cfg.get("quick_silence_ms") or 0.0),
                   quick_eot_threshold=float(cfg.get("quick_eot_threshold") or 1.01), vad_stream=vad_stream)

    @property
    def eot_check_ms(self) -> float:
        """Silence after which the EOT head starts being asked."""
        return min(self.quick_silence_ms, self.min_silence_ms)

    @property
    def vad_name(self) -> str:
        return "silero" if self.vad is not None else "energy"

    def reset(self) -> None:
        """New turn: clear the counters (the VAD keeps its audio context; see reset_vad)."""
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.reason: str | None = None

    def reset_vad(self) -> None:
        """New audio stream (session start, or listening resumes after the agent spoke)."""
        if self.vad is not None:
            self.vad.reset()
        self._active = False
        self._last_frame = None

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame is self._last_frame:  # asked again about the same frame: the stateful VAD must not see it twice
            return self._last_result
        if self.vad is None:
            result = rms_db(frame) > self.energy_threshold_db
        else:
            p = self.vad.prob(frame)
            # hysteresis (as in Silero's own iterator): start at the threshold, stop 0.15 below it
            on = p >= (self.vad_threshold - 0.15 if self._active else self.vad_threshold)
            self._active = on
            result = on and rms_db(frame) > self.min_level_db
        self._last_frame, self._last_result = frame, result
        return result

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
        if self.silence_ms >= self.quick_silence_ms and eot_prob >= self.quick_eot_threshold:
            self.reason = "eot_quick"
            return True
        if self.silence_ms >= self.min_silence_ms and eot_prob >= self.eot_threshold:
            self.reason = "eot_head"
            return True
        return False
