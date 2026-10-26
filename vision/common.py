"""(vision/common.py) Shared by the prep scripts, the eval and production: prompts, the zero policy, label matching.

THE TASK (production): the "before" inventory is the list of expected items; the model counts each one in the "after"
photo (0 = missing = an alert) and code compares. A second, separate question lists objects that are NOT on the list
("new items"); that one is not trained (our datasets only label their own classes).

THE ZERO POLICY, from three runs:
  v0  unlabelled classes written as 0          -> answered 0 for everything
  v1  zeros from any class, ~50-75% for some    -> said 0 for towels that were there
  v2  ~13% zeros, only "scene-matched" ones      -> learned "on the list = there": claimed pillows in 31 of 40 rooms
                                                   without pillows, lamps in 33 of 43
  v3  every class answered 0 in ~ZERO_SHARE (1/3) of the questions about it, allocated per class so every class gets
      its share; a missing item can be ANY item in ANY indoor photo (a towel or a kettle asked in a living room:
      hotel checklists do that), other photos only when a class runs short; never an item whose near-twin is in the
      photo (SIBLINGS).
Only REAL zeros are used: LVIS "verified absent" (annotators checked) or HomeObjects (every one of its 12 classes is
labelled on every image). A class that is merely unlabelled is never asked.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

COUNT_PROMPT = ("Expected items from the previous inspection: {names}. Count how many of each are in this image now "
                '(0 if it is not there). Answer in JSON: {{"objects":[{{"label":...,"detected_count":...}}]}}')

EXTRA_PROMPT = ("Expected items from the previous inspection: {names}. List every OTHER object you can see in this image "
                "that is not one of those, with how many. Answer in JSON: "
                '{{"new_items":[{{"label":...,"count":...}}]}}. If there are none: {{"new_items":[]}}')

# the box-finding question the training rows use; inference can count boxes with it (vl_infer.RoomModel.locate)
GROUND_PROMPT = 'Locate every {name} in the image. Answer in JSON: [{{"bbox_2d":[x1,y1,x2,y2],"label":"{name}"}}]'

# open detection (no item list): the BEFORE photo -> inventory. {area} = "Area being inspected: bathroom. " or "".
# Your own wording can replace it at run time (demo_app.py --detect-prompt FILE), so it never has to be in this repo.
DETECT_PROMPT = ("{area}Detect every distinct object visible in this room photo: furniture, appliances, electronics, "
                 "decor, linen, fixtures and small items. Do not skip any. One bounding box per instance. "
                 'Answer in JSON: [{{"bbox_2d":[x1,y1,x2,y2],"label":"object name"}}, ...]')

ZERO_SHARE = 1 / 3          # per class: share of the questions about it whose answer is 0
MAX_PRESENT = 6             # items that ARE there, per question
MAX_ZEROS = 2               # missing items per question
MIN_SHARE = 0.25            # a class short of real zeros is asked less often where present, down to this share
KEEP_FLOOR = 0.3            # ... but keeps at least this part of its present questions
MAIN = ["pillow", "towel", "lamp", "chair"]
INDOOR = {"bedroom", "living", "bathroom", "kitchen"}

# never answer "X: 0" when one of these is in the photo: the boundary between them is fuzzy (LVIS vs your labels)
_SIB = [("pillow", "cushion"), ("towel", "dish towel"), ("towel", "paper towel"),          # fuzzy boundaries
        ("table", "desk"), ("oven", "microwave"), ("sink", "kitchen sink"), ("clock", "alarm clock"),
        ("bottle", "water bottle"), ("bottle", "wine bottle"), ("plant pot", "flowers"), ("vase", "flowers"),
        ("lamp", "lampshade"), ("lamp", "light bulb"), ("lamp", "chandelier"), ("lamp", "spotlight"),  # parts / kinds
        ("bed", "mattress"), ("bed", "headboard"), ("oven", "stove"), ("cabinet", "drawer"), ("cabinet", "wardrobe"),
        ("sofa", "cushion")]
# NOT siblings on purpose: sofa/chair, chair/stool, bed/blanket, painting/mirror ("chair: 0" in a room with only a
# sofa is exactly the case v2 got wrong)
SIBLINGS: dict[str, set] = defaultdict(set)
for _a, _b in _SIB:
    SIBLINGS[_a].add(_b)
    SIBLINGS[_b].add(_a)


def allocate_zeros(present: list[dict], visible: list[set], absent: list[list], area_of: dict,
                   share: float = ZERO_SHARE, max_zeros: int = MAX_ZEROS, prefer=MAIN):
    """Pick the missing items for every question.

    present[i]  {class: count} that will be asked for row i (already cut to MAX_PRESENT)
    visible[i]  every class seen in row i's photo (also the ones not asked: their count was not exact)
    absent[i]   classes verified absent in row i's photo
    Returns (zeros per row, report). Per class, zeros = share/(1-share) x its present answers, taken at random from
    the photos where it is verified absent and no sibling is visible, indoor photos first; main classes first.
    """
    n_present = Counter(c for p in present for c in p)
    need = {c: round(n * share / (1 - share)) for c, n in n_present.items()}
    pool_in, pool_out = defaultdict(list), defaultdict(list)
    for i, (p, vis, ab) in enumerate(zip(present, visible, absent)):
        if not p:
            continue
        indoor = any(area_of.get(c) in INDOOR for c in p)
        blocked = set(vis).union(*(SIBLINGS[c] for c in vis))
        for z in ab:
            if z not in blocked and z in need:
                (pool_in if indoor else pool_out)[z].append(i)
    zeros: list[list] = [[] for _ in present]
    used = Counter()
    for c in sorted(need, key=lambda c: (c not in prefer, random.random())):
        # indoor photos first (the realistic "missing from the room" case); only if a class runs short, other photos
        # that have countable items (LVIS verifies "pillow" absent mostly outside: 65 of 7k images, 22 indoor)
        rows_in, rows_out = pool_in.get(c, []), pool_out.get(c, [])
        random.shuffle(rows_in)
        random.shuffle(rows_out)
        for i in rows_in + rows_out:
            if used[c] >= need[c]:
                break
            if len(zeros[i]) < max_zeros:
                zeros[i].append(c)
                used[c] += 1
    # a class that cannot get enough real zeros is asked less often where it IS there, so its zero share stays
    # >= MIN_SHARE (else "on the list = there" comes back for it); never below KEEP_FLOOR of its present questions.
    # The photo stays (its other items are still asked) and its grounding rows are untouched.
    dropped = Counter()
    for c in n_present:
        if used[c] >= need[c]:
            continue
        cap = max(math.ceil(used[c] * (1 - MIN_SHARE) / MIN_SHARE), math.ceil(KEEP_FLOOR * n_present[c]))
        extra = n_present[c] - cap
        if extra <= 0:
            continue
        rows = [i for i, p in enumerate(present) if c in p]
        rows.sort(key=lambda i: (len(present[i]) + len(zeros[i]) <= 1, random.random()))  # keep rows non-empty
        for i in rows[:extra]:
            del present[i][c]
            dropped[c] += 1
    n_present = Counter(c for p in present for c in p)
    report = {c: {"present": n_present[c], "zero": used[c], "share": round(used[c] / max(1, used[c] + n_present[c]), 3),
                  "short": max(0, need[c] - used[c]), "present_questions_dropped": dropped[c]}
              for c in sorted(set(n_present) | set(dropped))}
    tot_p, tot_z = sum(n_present.values()), sum(used.values())
    report["_ALL"] = {"present": tot_p, "zero": tot_z, "share": round(tot_z / max(1, tot_p + tot_z), 3)}
    return zeros, report


def print_zero_report(report: dict, title: str) -> None:
    a = report["_ALL"]
    print(f"{title}: {a['present']} present answers, {a['zero']} zero answers, zero share {a['share']:.0%}")
    for c in MAIN:
        if c in report:
            r = report[c]
            flag = "  <-- below 20%, not enough verified absences" if r["share"] < 0.2 else ""
            drop = f" ({r['present_questions_dropped']} present questions dropped)" if r.get("present_questions_dropped") else ""
            print(f"  {c:8s} present {r['present']:5d}  zero {r['zero']:5d}  share {r['share']:.0%}{drop}{flag}")
    low = [c for c, r in report.items() if c != "_ALL" and r["present"] >= 50 and r["share"] < 0.2]
    if low:
        print(f"  classes with >=50 present answers but <20% zeros: {', '.join(low[:20])}")


def count_row(img: str, counts: dict) -> dict:
    names = list(counts)
    random.shuffle(names)
    answer = {"objects": [{"label": n, "detected_count": counts[n]} for n in names]}
    return {"messages": [{"role": "user", "content": "<image>" + COUNT_PROMPT.format(names=", ".join(names))},
                         {"role": "assistant", "content": json.dumps(answer)}], "images": [img]}


# ---------------------------------------------------------------- open-vocabulary label matching (new items)
SYNONYMS = {"picture": "painting", "picture frame": "painting", "wall art": "painting", "photograph": "painting",
            "poster": "painting", "frame": "painting", "photo frame": "painting", "artwork": "painting",
            "plant": "potted plant", "plant pot": "potted plant", "houseplant": "potted plant", "flowers": "vase",
            "tv": "television", "television set": "television", "bookshelf": "shelf", "bookcase": "shelf",
            "couch": "sofa", "carpet": "rug", "drapes": "curtain", "blinds": "curtain", "nightstand": "bedside table",
            "staircase": "stairs", "coffee machine": "coffee maker", "fridge": "refrigerator", "cushion": "pillow"}
# parts of the room, not objects: a model listing them is not finding "new items"
STRUCTURE = {"wall", "floor", "ceiling", "room", "corner", "light", "shadow", "baseboard"}


def norm_label(s: str) -> str:
    s = str(s).lower().strip()
    s = SYNONYMS.get(s, s)
    if s.endswith(("sses", "xes", "ches", "shes")):      # glasses, boxes, benches, dishes
        s = s[:-2]
    elif s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return SYNONYMS.get(s, s)


def same_object(a: str, b: str) -> bool:
    """Loose match for open-vocabulary labels: equal after synonyms, or the same head noun ("coffee table" ~ "table")."""
    a, b = norm_label(a), norm_label(b)
    return a == b or a.split()[-1] == b.split()[-1]


def clean_new_items(items: dict, expected: list[str]) -> dict:
    """What production code keeps from a "new items" answer: count > 0, not an expected item, not a room part."""
    return {k: v for k, v in items.items()
            if v and v > 0 and norm_label(k) not in STRUCTURE and not any(same_object(k, e) for e in expected)}
