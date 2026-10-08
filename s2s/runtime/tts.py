"""Reply voice from a text-to-speech model instead of the talker (`runtime.voice: kokoro`).

The thinker streams its reply token by token; `PhraseChunker` cuts the text into short phrases
(the first one as early as possible, so the reply starts quickly) and `KokoroVoice` speaks each
phrase as soon as it is complete. Kokoro-82M (Apache-2.0, 24 kHz) is the voice our talker was
trained to imitate, so this is the quality ceiling of the current talker, at a similar size
(82M vs talker 47M + Mimi decoder).

Note for a commercial launch: Kokoro's English front end (misaki) falls back to espeak-ng (GPL)
for unknown words; have that reviewed.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

_END = re.compile(r"[.!?…]['\")\]]*$")       # sentence end
_PAUSE = re.compile(r"[,;:—–]['\")\]]*$")     # clause boundary


class PhraseChunker:
    """Collects streamed text pieces and returns complete phrases.

    A phrase ends at a sentence end, or at a clause boundary once it has `min_words` words (the
    first phrase needs fewer, to start speaking early), or after `max_words` words without any
    punctuation. Boundaries are only taken before a space, so "11:30" or "3.5" are never split.
    """

    def __init__(self, first_min_words: int = 2, min_words: int = 6, max_words: int = 18):
        self.first_min_words = first_min_words
        self.min_words = min_words
        self.max_words = max_words
        self.buf = ""
        self.first = True

    def push(self, piece: str) -> list[str]:
        out = []
        # a boundary is only known once the next piece starts with a space (or the reply ends)
        if self.buf and piece[:1].isspace():
            phrase = self._boundary()
            if phrase:
                out.append(phrase)
        self.buf += piece
        return out

    def flush(self) -> list[str]:
        text, self.buf = self.buf.strip(), ""
        return [text] if text and any(ch.isalnum() for ch in text) else []

    def _boundary(self) -> str | None:
        text = self.buf.rstrip()
        words = len(text.split())
        need = self.first_min_words if self.first else self.min_words
        cut = _END.search(text) or (_PAUSE.search(text) and words >= need) or words >= self.max_words
        if not cut or not any(ch.isalnum() for ch in text):
            return None
        self.buf, self.first = "", False
        return text.strip()


class KokoroVoice:
    sample_rate = 24000

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, device: str = "cpu"):
        from kokoro import KPipeline  # optional dependency: pip install "kokoro>=0.9" "misaki[en]"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
        self.voice = voice
        self.speed = speed
        self.speak("Hello.")  # warm-up: the first call loads the voice and compiles kernels

    def speak(self, text: str) -> np.ndarray:
        parts = [r.audio for r in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=None)
                 if r.audio is not None]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([p.detach().cpu().numpy() for p in parts]).astype(np.float32)
