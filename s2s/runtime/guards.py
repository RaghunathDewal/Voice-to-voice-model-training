"""Safety checks on the thinker's reply before it is spoken (model-independent, no training needed).

  claims_action     the reply says an order was placed / an issue was reported, but no such tool call
                    succeeded in this turn -> retry once, telling the model to call the tool or not claim it
  unsupported       the reply states a value (password, code, time, price, number) or a proper name
                    (a restaurant, a place) that is in neither the system prompt, the guest's words nor a
                    tool result -> retry once, telling the model which details it must not invent
  speakable         strips markdown, bullets and emojis so the voice does not read them out

VOICE_RULES is appended to a deployment prompt (it does not replace it): the behaviour a phone call needs
and the rules the untrained model broke most often in tests.
"""

from __future__ import annotations

import re

VOICE_RULES = """
VOICE CALL RULES (these apply on every turn):
- You are speaking on a phone call: answer in plain spoken sentences. No markdown, no bold, no lists, no emojis.
- Reservation fields that are empty, null or missing are UNKNOWN. Never guess them: no example passwords, codes,
  times, prices or names. Say: "I apologize, but I don't have that information at the moment. Is there anything
  else I can assist you with?"
- Only state details (places, restaurants, times, prices, codes) that appear in the reservation data or in a tool
  result. Do not recommend places from general knowledge.
- The tone examples above (names, places, times in them) are illustrations of style, not facts about this guest.
- Explicit requests ("please send", "can I get", "I want to order", "please report", "report these"): call the
  tool right away in this turn, without asking for confirmation. Ask for confirmation only when the guest just
  mentions a missing item or a problem without asking you to act.
- Never say that something was ordered, requested, reported or logged unless you called the tool in this turn
  and it returned success.
- Report one problem per report_unit_issue call: several problems means several calls in the same turn.
""".strip()

_CLAIM = re.compile(
    r"\b(done!?|i(?:'| ha)ve (?:placed|requested|ordered|reported|logged|filed|submitted|arranged|sent)|"
    r"(?:has|have) been (?:placed|requested|ordered|reported|logged|filed|submitted|arranged)|"
    r"i(?:'| a)m placing|is on (?:its|the) way|are on (?:their|the) way)\b", re.I)
ACTION_TOOLS = {"place_product_order", "report_unit_issue", "order_product", "create_issue", "report_issue"}

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF‍️]+")
_STOP_NAMES = {"I", "OK", "AM", "PM", "Yes", "No", "Sure", "Great", "Hello", "Hey", "Hi", "Done", "Thanks", "Thank",
               "Please", "Let", "Of", "The", "A", "An", "Your", "Our", "We", "You", "It", "This", "That", "If", "Is",
               "Would", "Could", "Can", "Here", "There", "Unfortunately", "Absolutely", "Perfect", "Welcome"}


def speakable(text: str) -> str:
    """Text as it should be spoken: no markdown, list markers or emojis; lines joined into sentences."""
    t = _EMOJI.sub("", text)
    t = re.sub(r"\*\*|__|`|#+\s*", "", t)
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def claims_action(reply: str, succeeded: list[str]) -> bool:
    """The reply claims an order / report although no action tool succeeded in this turn."""
    return bool(_CLAIM.search(reply)) and not any(n in ACTION_TOOLS for n in succeeded)


def _values(text: str) -> set[str]:
    """Value-like tokens: anything with a digit (codes, passwords, times, prices, ids)."""
    return {t.strip(".,:;!?()").lower() for t in re.findall(r"[A-Za-z@#$]*\d[\w:./@#$-]*", text)
            if t.strip(".,:;!?()")}


def _names(text: str) -> set[str]:
    """Multi-word proper names ("Casa de Campo", "Austin Beer Garden") not at a sentence start."""
    out = set()
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z'’]+(?:\s+(?:de|of|the|la|el|and|&)?\s*[A-Z][a-z'’]+)+)", text):
        words = m.group(1).split()
        if words[0] not in _STOP_NAMES:
            out.add(m.group(1).lower())
    return out


def unsupported(reply: str, sources: list[str]) -> list[str]:
    """Values and proper names in the reply that appear in none of the sources (prompt, guest, tool results)."""
    src = " ".join(sources)
    src_low = src.lower()
    src_vals = _values(src)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s)  # noqa: E731
    src_compact = norm(src_low)
    bad = []
    for v in _values(reply):
        if v in src_vals or norm(v) in src_compact:
            continue
        if re.fullmatch(r"[1-9]|10", v):  # small counts ("2 towels", "1 issue") are not facts to verify
            continue
        bad.append(v)
    for n in _names(reply):
        if n not in src_low:
            bad.append(n)
    return bad


def retry_note(kind: str, detail: list[str] | None = None) -> str:
    if kind == "claim":
        return ("(Check before answering: you said the request was done, but you did not call the tool in this "
                "turn. If the guest asked you to do it, call the tool now. Otherwise do not say it was done.)")
    return ("(Check before answering: these details are not in the reservation data, the guest's words or a tool "
            f"result, so you must not state them: {', '.join(detail or [])}. Answer again without them; if the "
            "guest needs them, say you don't have that information.)")
