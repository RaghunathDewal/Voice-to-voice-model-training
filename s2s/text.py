"""Character vocabulary for the auxiliary CTC head, text normalisation and WER."""

from __future__ import annotations

import re

import jiwer

CTC_BLANK = 0
_CHARS = [" ", "'"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
CTC_VOCAB = ["<blank>"] + _CHARS
_CHAR_TO_ID = {c: i + 1 for i, c in enumerate(_CHARS)}


def normalize_for_ctc(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ctc_encode(text: str) -> list[int]:
    return [_CHAR_TO_ID[c] for c in normalize_for_ctc(text)]


def ctc_greedy_decode(ids: list[int]) -> str:
    out, prev = [], None
    for i in ids:
        if i != prev and i != CTC_BLANK:
            out.append(CTC_VOCAB[i])
        prev = i
    return re.sub(r"\s+", " ", "".join(out)).strip()


def sentence_case(text: str) -> str:
    """LibriSpeech transcripts are ALL CAPS; make them look like normal text."""
    text = text.strip().lower()
    text = re.sub(r"\bi\b", "I", text)
    return text[:1].upper() + text[1:] if text else text


def wer(refs: list[str], hyps: list[str]) -> float:
    refs_n = [normalize_for_ctc(r) for r in refs]
    hyps_n = [normalize_for_ctc(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs_n, hyps_n) if r]
    if not pairs:
        return 0.0
    return float(jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]))
