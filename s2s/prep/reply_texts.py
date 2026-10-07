"""Clean distilled replies for adapter training and build the talker's text corpus.

The hotel-tuned thinker answers many general sentences (Common Voice, VoxPopuli, ...) with a
generic hand-over ("the front desk will be happy to help ...", "I'm sorry to hear that ...").
Such replies do not depend on what was said, so they teach the adapter nothing about the
speech and would bias the live agent towards handing over. Two modes:

    # adapter: drop generic replies (those rows stay in as transcription-only)
    python -m s2s.prep.reply_texts strip --in data/manifests/cv_train_pk_d.jsonl --out data/manifests/cv_train_pk_df.jsonl

    # talker: what the thinker actually says (hotel replies + distilled replies), deduplicated
    python -m s2s.prep.reply_texts talker --hotel 8000 --replies data/manifests/*_pk_d.jsonl \
        --out data/manifests/talker_v2_text.jsonl --max 16000
"""

from __future__ import annotations

import argparse
import glob
import random
import re
from collections import Counter

from s2s.utils import read_jsonl, write_jsonl

_GENERIC = re.compile(
    r"front desk|sorry to hear|i'm here to help|i can help (you )?with|anything else|"
    r"not able to (do|help)|can't (do|help)|i don't have (access|information)|call the police",
    re.I)


def is_generic(text: str) -> bool:
    return bool(_GENERIC.search(text or "")) or len((text or "").split()) < 3


def strip(inp: str, out: str) -> None:
    rows = read_jsonl(inp)
    dropped = 0
    for r in rows:
        if r.get("response") and is_generic(r["response"]):
            del r["response"]
            dropped += 1
    kept = sum(bool(r.get("response")) for r in rows)
    write_jsonl(out, rows)
    print(f"{inp}: dropped {dropped} generic replies, {kept} replies kept, {len(rows)} rows -> {out}")


def talker(hotel: int, replies: list[str], out: str, max_rows: int, max_per_text: int, seed: int,
           hotel_version: int = 2) -> None:
    from s2s.data.hotel_v2 import generate_examples_v2
    from s2s.data.hotel_v3 import generate_examples_v3

    rng = random.Random(seed)
    texts: list[str] = []
    gen = generate_examples_v3 if hotel_version == 3 else generate_examples_v2
    for r in gen(hotel, seed=0):  # the hotel replies the agent speaks
        texts += [t for t in (r.get("reply"), r.get("reply_after_tool")) if t]
    hotel_n = len(texts)
    for path in replies:
        texts += [r["response"] for r in read_jsonl(path) if r.get("response") and not is_generic(r["response"])]
    rng.shuffle(texts)
    seen: Counter = Counter()
    rows = []
    for t in texts:
        t = re.sub(r"\s+", " ", t).strip()
        if not t or len(t) > 300 or seen[t] >= max_per_text:  # templated hotel lines: a few copies each
            continue
        seen[t] += 1
        rows.append({"id": f"talker_v2_{len(rows):06d}", "text": t})
        if len(rows) >= max_rows:
            break
    write_jsonl(out, rows)
    print(f"{len(rows)} talker texts ({hotel_n} hotel replies before dedup, {len(seen)} unique) -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    t = sub.add_parser("talker")
    t.add_argument("--hotel", type=int, default=8000)
    t.add_argument("--replies", nargs="*", default=[])
    t.add_argument("--out", required=True)
    t.add_argument("--max", type=int, default=16000)
    t.add_argument("--max-per-text", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--hotel-version", type=int, default=2, choices=[2, 3], help="hotel reply generator")
    a = p.parse_args()
    if a.cmd == "strip":
        strip(a.inp, a.out)
    else:
        talker(a.hotel, [f for pat in a.replies for f in sorted(glob.glob(pat))], a.out, a.max, a.max_per_text, a.seed,
               a.hotel_version)


if __name__ == "__main__":
    main()
