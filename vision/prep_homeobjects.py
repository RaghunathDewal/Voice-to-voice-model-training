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

# HomeObjects class id -> YOUR class name (same names as prep_lvis.py)
NAMES = ["bed", "sofa", "chair", "table", "lamp", "television", "laptop", "wardrobe",
         "window", "door", "plant pot", "painting"]          # tv -> television, potted plant, photo frame -> painting
ALWAYS_ASK = ["chair", "lamp"]                               # the two main classes HomeObjects can answer

COUNT_PROMPT = ("Count the following objects in this image: {names}. "
                'Answer in JSON: {{"objects":[{{"label":...,"detected_count":...}}]}}')
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
    p.add_argument("--names-per-row", type=int, default=5, help="classes asked in one count row")
    p.add_argument("--max-count", type=int, default=30)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    st = Counter()
    f_cnt = open(f"{a.out}/count_sft.jsonl", "w")
    f_gnd = open(f"{a.out}/ground_sft.jsonl", "w")
    for img in sorted(glob.glob(f"{a.root}/images/{a.split}/*.jpg")):
        boxes = read_yolo(f"{a.root}/labels/{a.split}/{os.path.splitext(os.path.basename(img))[0]}.txt")
        n = Counter(c for c, _ in boxes)
        img = os.path.abspath(img)
        # count row: chair + lamp always, then present classes, then a few true zeros
        others = [c for c in NAMES if c not in ALWAYS_ASK]
        random.shuffle(others)
        others.sort(key=lambda c: n[c] == 0)                 # present ones first
        ask = ALWAYS_ASK + others[:a.names_per_row - len(ALWAYS_ASK)]
        if max(n[c] for c in ask) > a.max_count:
            continue
        random.shuffle(ask)
        answer = {"objects": [{"label": c, "detected_count": n[c]} for c in ask]}
        f_cnt.write(json.dumps({"messages": [
            {"role": "user", "content": "<image>" + COUNT_PROMPT.format(names=", ".join(ask))},
            {"role": "assistant", "content": json.dumps(answer)}], "images": [img]}) + "\n")
        st["count_rows"] += 1
        for c in ask:
            st[f"{c}_{'present' if n[c] else 'zero'}"] += 1
        for c in sorted(n):
            f_gnd.write(json.dumps({"messages": [
                {"role": "user", "content": "<image>" + GROUND_PROMPT.format(name=c)},
                {"role": "assistant", "content": json.dumps([{"bbox_2d": b, "label": c} for k, b in boxes if k == c])}],
                "images": [img]}) + "\n")
            st["ground_rows"] += 1
    print(json.dumps(dict(sorted(st.items()))))


if __name__ == "__main__":
    main()
