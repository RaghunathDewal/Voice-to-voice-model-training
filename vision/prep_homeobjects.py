"""(vision/prep_homeobjects.py) HomeObjects-3K TRAIN split -> Qwen3-VL counting + grounding rows.

    python prep_homeobjects.py --root data/homeobjects-3K --out data/home                    # training rows
    python prep_homeobjects.py --root data/homeobjects-3K --out data/home_val --split val    # test rows (never trained)

HomeObjects labels 12 classes on EVERY image (YOLO style), so for those 12 a missing box is a real 0.
It has NO pillow / towel class: they are never asked here (that was the "always 0" bug).
Training uses images/train only; images/val is for the eval set (build_eval_set.py), like eval_frozen.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

from common import ZeroBudget, count_row

# HomeObjects class id -> YOUR class name (same names as prep_lvis.py)
NAMES = ["bed", "sofa", "chair", "table", "lamp", "television", "laptop", "wardrobe",
         "window", "door", "plant pot", "painting"]          # tv -> television, potted plant, photo frame -> painting
AREA_OF = {c: "living" for c in NAMES}                       # all indoor room classes: any of them fits a room

GROUND_PROMPT = 'Locate every {name} in the image. Answer in JSON: [{{"bbox_2d":[x1,y1,x2,y2],"label":"{name}"}}]'


def read_yolo(txt: str) -> list[tuple[str, list[int]]]:
    out = []
    for line in open(txt) if os.path.exists(txt) else []:
        p = line.split()
        if len(p) < 5 or int(float(p[0])) >= len(NAMES):
            continue
        x, y, w, h = (float(v) for v in p[1:5])
        out.append((NAMES[int(float(p[0]))], [round(max(0, x - w / 2) * 1000), round(max(0, y - h / 2) * 1000),
                                              round(min(1, x + w / 2) * 1000), round(min(1, y + h / 2) * 1000)]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/homeobjects-3K")
    p.add_argument("--out", default="data/home")
    p.add_argument("--max-names", type=int, default=6, help="classes asked in one count row")
    p.add_argument("--zero-ratio", type=float, default=0.4, help="see common.py")
    p.add_argument("--max-count", type=int, default=30)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    items = []
    for img in sorted(glob.glob(f"{a.root}/images/{a.split}/*.jpg")):
        boxes = read_yolo(f"{a.root}/labels/{a.split}/{os.path.splitext(os.path.basename(img))[0]}.txt")
        items.append((os.path.abspath(img), boxes, Counter(c for c, _ in boxes)))
    budget = ZeroBudget(Counter(c for _, _, n in items for c in n), a.zero_ratio)
    st = Counter()
    f_cnt = open(f"{a.out}/count_sft.jsonl", "w")
    f_gnd = open(f"{a.out}/ground_sft.jsonl", "w")
    for img, boxes, n in items:
        present = [c for c in n if n[c] <= a.max_count]
        if not present:
            continue
        if len(present) > a.max_names - 1:
            present = random.sample(present, a.max_names - 1)
        counts = {c: n[c] for c in present}
        # HomeObjects labels all 12 classes on every image: any absent one is a real missing item
        z = budget.pick([c for c in NAMES if n[c] == 0], AREA_OF, present)
        if z:
            counts[z] = 0
        f_cnt.write(json.dumps(count_row(img, counts)) + "\n")
        st["count_rows"] += 1
        for c in counts:
            st[f"{c}_{'present' if counts[c] else 'zero'}"] += 1
        for c in sorted(n):
            f_gnd.write(json.dumps({"messages": [
                {"role": "user", "content": "<image>" + GROUND_PROMPT.format(name=c)},
                {"role": "assistant", "content": json.dumps([{"bbox_2d": b, "label": c} for k, b in boxes if k == c])}],
                "images": [img]}) + "\n")
            st["ground_rows"] += 1
    zeros = sum(v for k, v in st.items() if k.endswith("_zero"))
    answers = sum(v for k, v in st.items() if k.endswith("_zero") or k.endswith("_present"))
    print(json.dumps(dict(sorted(st.items()))), f"zero share {zeros / max(1, answers):.0%}")


if __name__ == "__main__":
    main()
