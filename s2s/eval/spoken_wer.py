"""Word error rate for comparing transcripts written in different styles.

References and recognisers disagree on how to write numbers ("7:45" vs "seven forty five", "10 AM" vs
"ten a m"); s2s.text.wer drops digits altogether. Here numbers are spelled out and a.m./p.m. joined before
the usual normalisation, the same way for reference and hypothesis.
"""

from __future__ import annotations

import re

from s2s.text import wer

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if 1100 <= n < 2000 or 2010 <= n < 2100:  # years: "nineteen ninety nine", "twenty twenty four"
        rest = n % 100
        tail = "hundred" if rest == 0 else number_words(rest) if rest >= 10 else "oh " + _ONES[rest]
        return number_words(n // 100) + " " + tail
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))  # long ids: digit by digit


_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def ordinal_words(n: int) -> str:
    words = number_words(n).split()
    last = words[-1]
    words[-1] = _ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(words)


def spell_numbers(text: str) -> str:
    text = re.sub(r"(\d+)[:.](\d\d)\b", lambda m: f"{m.group(1)} {m.group(2) if m.group(2) != '00' else ''}", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: " " + ordinal_words(int(m.group(1))) + " ", text)
    text = re.sub(r"\d+", lambda m: " " + number_words(int(m.group(0))) + " ", text)
    text = re.sub(r"\b([ap])\s*\.?\s*m\b\.?", r"\1m", text, flags=re.I)
    return text


def spoken_wer(refs: list[str], hyps: list[str]) -> float:
    return wer([spell_numbers(r) for r in refs], [spell_numbers(h) for h in hyps])
