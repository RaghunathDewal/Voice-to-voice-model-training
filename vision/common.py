"""(vision/common.py) What every prep script and the eval share: the counting prompt and the rule for zeros.

The prompt is the production one: the "before" inventory is the list of expected items, the model counts each in the
"after" image, and code compares the two. A 0 is a missing item, i.e. an alert, so zeros are trained, but:
  - only zeros that FIT the scene (a towel missing in a bathroom, a pillow missing on a bed, never "chopsticks" in a
    street photo): the area of the zero class must fit an area of a class that is in the image
  - at most one zero per question, and per class at most ZERO_RATIO x the rows where that class is present
    (v1 had ~3 zero rows per towel row and learned to answer 0 for towels that were there)
"""

from __future__ import annotations

import json
import random
from collections import Counter

COUNT_PROMPT = ("Expected items from the previous inspection: {names}. Count how many of each are in this image now "
                '(0 if it is not there). Answer in JSON: {{"objects":[{{"label":...,"detected_count":...}}]}}')

# which areas a missing item may come from, given the areas of the items that ARE in the image
FITS = {"bedroom": {"bedroom", "living", "building"},
        "living": {"living", "bedroom", "building"},
        "bathroom": {"bathroom", "building"},
        "kitchen": {"kitchen", "building"},
        "building": {"building"}}


class ZeroBudget:
    """Per class, at most ratio x (rows where it is present) zero answers."""

    def __init__(self, present_rows: Counter, ratio: float):
        self.limit = {c: int(ratio * n) for c, n in present_rows.items()}
        self.used = Counter()

    def pick(self, candidates: list[str], area_of: dict, present: list[str], prefer=()) -> str | None:
        areas = set().union(*(FITS.get(area_of.get(p, ""), set()) for p in present)) if present else set()
        ok = [z for z in candidates if area_of.get(z) in areas and self.used[z] < self.limit.get(z, 0)]
        if not ok:
            return None
        random.shuffle(ok)
        ok.sort(key=lambda z: z not in prefer)
        self.used[ok[0]] += 1
        return ok[0]


def count_row(img: str, counts: dict) -> dict:
    names = list(counts)
    random.shuffle(names)
    answer = {"objects": [{"label": n, "detected_count": counts[n]} for n in names]}
    return {"messages": [{"role": "user", "content": "<image>" + COUNT_PROMPT.format(names=", ".join(names))},
                         {"role": "assistant", "content": json.dumps(answer)}], "images": [img]}
