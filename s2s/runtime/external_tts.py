"""Speak the thinker's reply with an external TTS server instead of our talker (runtime.tts_url).

The reply text arrives token by token. It is cut into phrases: the first one at its first pause (",", ".", "?" ...)
or after a few words, so speech starts while the thinker is still writing; later ones at sentence ends. Each phrase is sent to the server in a background thread (one at a time, in order)
and its audio is handed out as soon as it is ready, so the thinker keeps generating meanwhile.

Server protocol (s2s/cli/qwen_tts_server.py): POST <url> {"text": "..."} -> raw int16 mono PCM, with the
sample rate in the X-Sample-Rate header.
"""

from __future__ import annotations

import collections
import json
import re
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

STRONG = ".!?"
WEAK = ",;:—-"
_MARKUP = re.compile(r"[*_#`~|<>\[\]{}]")
# a "." after these is not the end of a sentence
ABBREV = {"st", "dr", "mr", "mrs", "ms", "no", "ave", "rd", "blvd", "apt", "ste", "jr", "sr", "vs", "etc", "approx",
          "dept", "est", "fig", "ft", "inc", "ltd", "co", "mt", "a.m", "p.m", "e.g", "i.e", "u.s"}


def clean_for_speech(text: str) -> str:
    """Markdown / bracket characters are not spoken; whitespace collapsed."""
    return re.sub(r"\s+", " ", _MARKUP.sub(" ", text)).strip()


class ExternalTTS:
    def __init__(self, url: str, first_words: int = 2, first_max_words: int = 5, min_words: int = 6,
                 max_words: int = 28, timeout: float = 60.0):
        self.url = url
        self.first_words, self.first_max_words = first_words, first_max_words
        self.min_words, self.max_words = min_words, max_words
        self.timeout = timeout

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        req = urllib.request.Request(self.url, data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            sr = int(r.headers.get("X-Sample-Rate", 24000))
            pcm = r.read()
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, sr

    def stream(self) -> "PhraseStream":
        return PhraseStream(self)


class PhraseStream:
    """push() reply text pieces, take ready() audio after each token, finish() at the end of the reply."""

    def __init__(self, tts: ExternalTTS):
        self.tts = tts
        self.buf = ""
        self.boundary = False  # the buffer ends at punctuation; cut there if the next piece starts a new word
        self.sent = 0
        self.pool = ThreadPoolExecutor(max_workers=1)  # phrases are spoken in order
        self.futures: collections.deque[Future] = collections.deque()

    def _words(self) -> int:
        return len(self.buf.split())

    def _ready_to_cut(self) -> bool:
        n, end = self._words(), self.buf.rstrip()[-1:]
        if end == ".":
            last = self.buf.split()[-1].rstrip(".").lower() if n else ""
            if last in ABBREV or (len(last) == 1 and last.isalpha()):
                end = ""  # "St." / "Dr." / an initial: not a pause
        strong, weak = bool(end) and end in STRONG, bool(end) and end in WEAK  # ("" in "abc" is True)
        if self.sent == 0:  # the first phrase: short, so speech starts early: the first pause (even one word:
            # "Absolutely!", "I'm sorry,"), or a few words when no pause comes ("Your unit is located at | 5103 ...")
            return strong or (n >= self.tts.first_words and weak) or n >= self.tts.first_max_words
        return (n >= self.tts.min_words and strong) or (n >= 2 * self.tts.min_words and weak) or n >= self.tts.max_words

    def _submit(self) -> None:
        text = clean_for_speech(self.buf)
        self.buf, self.boundary = "", False
        if text and any(c.isalnum() for c in text):
            self.futures.append(self.pool.submit(self.tts.synth, text))
            self.sent += 1

    def push(self, piece: str) -> None:
        # cut only before a piece that starts a new word, so "12.5" or "St.#12" are never split mid-token
        if self.boundary and piece[:1].isspace():
            self._submit()
        self.buf += piece
        self.boundary = self._ready_to_cut()

    def ready(self):
        """Audio of the phrases finished so far, in order: (wav, sample_rate) tuples."""
        while self.futures and self.futures[0].done():
            yield self.futures.popleft().result()

    def finish(self):
        """Speak what is left and wait for every phrase."""
        self._submit()
        while self.futures:
            yield self.futures.popleft().result()
        self.pool.shutdown(wait=False)
