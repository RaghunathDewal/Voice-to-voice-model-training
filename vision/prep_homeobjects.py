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

from common import GROUND_PROMPT, MAX_PRESENT, ZERO_SHARE, allocate_zeros, count_row, print_zero_report

# HomeObjects class id -> YOUR class name (same names as prep_lvis.py)
NAMES = ["bed", "sofa", "chair", "table", "lamp", "television", "laptop", "wardrobe",
         "window", "door", "plant pot", "painting"]          # tv -> television, potted plant, photo frame -> painting
AREA_OF = {c: "living" for c in NAMES}                       # all indoor room classes: any of them fits a room



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
    p.add_argument("--zero-share", type=float, default=ZERO_SHARE, help="per class share of 0 answers (common.py)")
    p.add_argument("--max-count", type=int, default=30)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    imgs, present, visible, absent = [], [], [], []
    n_ground = 0
    with open(f"{a.out}/ground_sft.jsonl", "w") as f_gnd:
        for img in sorted(glob.glob(f"{a.root}/images/{a.split}/*.jpg")):
            boxes = read_yolo(f"{a.root}/labels/{a.split}/{os.path.splitext(os.path.basename(img))[0]}.txt")
            n = Counter(c for c, _ in boxes)
            img = os.path.abspath(img)
            for c in sorted(n):
                f_gnd.write(json.dumps({"messages": [
                    {"role": "user", "content": "<image>" + GROUND_PROMPT.format(name=c)},
                    {"role": "assistant", "content": json.dumps([{"bbox_2d": b, "label": c} for k, b in boxes if k == c])}],
                    "images": [img]}) + "\n")
                n_ground += 1
            counts = {c: k for c, k in n.items() if k <= a.max_count}
            if not counts:
                continue
            if len(counts) > MAX_PRESENT:
                counts = {c: counts[c] for c in random.sample(list(counts), MAX_PRESENT)}
            imgs.append(img)
            present.append(counts)
            visible.append(set(n))
            absent.append([c for c in NAMES if n[c] == 0])   # all 12 labelled on every image: absent = really absent
    zeros, report = allocate_zeros(present, visible, absent, AREA_OF, share=a.zero_share)
    with open(f"{a.out}/count_sft.jsonl", "w") as f_cnt:
        for img, counts, zs in zip(imgs, present, zeros):
            if counts or zs:
                f_cnt.write(json.dumps(count_row(img, {**counts, **{z: 0 for z in zs}})) + "\n")
    json.dump(report, open(f"{a.out}/zero_report.json", "w"), indent=1)
    print(f"{len(imgs)} counting rows, {n_ground} grounding rows ({a.split})")
    print_zero_report(report, f"HomeObjects {a.split}")


if __name__ == "__main__":
    main()
