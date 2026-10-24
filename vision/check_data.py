"""(vision/check_data.py) Gate between data and training: look at the FINAL training file before GPU hours are spent.

    python check_data.py data/sft/train.jsonl [--eval data/eval/eval_set.jsonl]

Fails (exit 1) when the mix repeats a known mistake:
  - a main class (pillow, towel, lamp, chair) answered 0 in less than 15% or more than 50% of its questions
    (v2: ~5% -> "on the list = there"; v1: ~75% for towel -> "towel = 0")
  - a counting question that is not the production prompt
  - a training image that is also in the eval set
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from common import COUNT_PROMPT, MAIN

LOW, HIGH = 0.15, 0.50


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train")
    p.add_argument("--eval", default="")
    a = p.parse_args()
    head = COUNT_PROMPT.split("{names}")[0]
    pres, zero, tasks, bad_prompt, imgs = Counter(), Counter(), Counter(), 0, set()
    hi_counts = Counter()
    for line in open(a.train):
        r = json.loads(line)
        imgs.add(r["images"][0])
        q, ans = r["messages"][0]["content"], r["messages"][1]["content"]
        if "Locate every" in q:
            tasks["grounding"] += 1
            continue
        tasks["counting"] += 1
        if head not in q:
            bad_prompt += 1
        for o in json.loads(ans)["objects"]:
            (zero if o["detected_count"] == 0 else pres)[o["label"]] += 1
            if o["detected_count"] >= 10:
                hi_counts[o["label"]] += 1
    errors = []
    tz, tp = sum(zero.values()), sum(pres.values())
    print(f"training file: {sum(tasks.values())} rows ({dict(tasks)}) on {len(imgs)} images")
    print(f"counting answers: {tp} present, {tz} zero -> zero share {tz / max(1, tz + tp):.0%}")
    for c in MAIN:
        sh = zero[c] / max(1, zero[c] + pres[c])
        ok = LOW <= sh <= HIGH
        print(f"  {c:8s} present {pres[c]:5d}  zero {zero[c]:5d}  share {sh:.0%}  {'ok' if ok else 'OUT OF RANGE'}")
        if pres[c] and not ok:
            errors.append(f"{c} zero share {sh:.0%} outside {LOW:.0%}-{HIGH:.0%}")
    skew = [(c, zero[c] / (zero[c] + pres[c])) for c in pres if pres[c] + zero[c] >= 100]
    skew = [f"{c} {s:.0%}" for c, s in sorted(skew, key=lambda x: x[1]) if s < 0.1 or s > 0.6]
    if skew:
        print(f"  other classes with >=100 questions and <10% or >60% zeros (info): {', '.join(skew[:15])}")
    if hi_counts:
        print(f"  answers of 10 or more (crowded scenes, info): {dict(hi_counts.most_common(6))}")
    if bad_prompt:
        errors.append(f"{bad_prompt} counting rows without the production prompt")
    if a.eval:
        ev = {json.loads(l)["images"][0] for l in open(a.eval)}
        leak = len(ev & imgs)
        print(f"eval images also in training: {leak}")
        if leak:
            errors.append(f"{leak} eval images are in the training file")
    if errors:
        print("DATA CHECK FAILED: " + "; ".join(errors))
        sys.exit(1)
    print("DATA CHECK OK")


if __name__ == "__main__":
    main()
