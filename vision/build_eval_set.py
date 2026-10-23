"""(vision/build_eval_set.py) One mixed eval set from images that are NEVER trained on.

    python build_eval_set.py --out data/eval/eval_set.jsonl \
        --test data/sft/test.jsonl --home-val data/home_val/count_sft.jsonl --frozen /tmp/vl_eval/.../eval_data

Sources (each scored on its own, and together):
  frozen          your 79 hand-checked rooms (eval_frozen.json): pillow, towel, lamp, chair
  homeobjects_val HomeObjects val split: its 12 classes, real zeros (every class labelled on every image)
  lvis            LVIS images held out by mix_data.py: up to 120 hotel classes, zeros only where LVIS verified them
  + any other source in test.jsonl (e.g. your own hotel rows passed as EXTRA)
Every row: {"source", "images", "classes", "gt": {class: count}}.
"""

from __future__ import annotations

import argparse
import json
import os
import random

MAIN = ["pillow", "towel", "lamp", "chair"]


def from_row(r: dict, source: str) -> dict:
    gt = {o["label"]: o["detected_count"] for o in json.loads(r["messages"][1]["content"])["objects"]}
    return {"source": source, "images": r["images"], "classes": list(gt), "gt": gt}


def frozen_rows(d: str) -> list[dict]:
    out = []
    for name, item in json.load(open(os.path.join(d, "eval_frozen.json"))).items():
        path = os.path.join(d, "images", name)
        if not os.path.exists(path):
            continue
        full = item.get("full_counts") or {}
        gt = {c: item["baseline"].get(c, 0) for c in ("pillow", "towel", "chair")}
        # training has ONE lamp class (LVIS and HomeObjects cannot tell bedside from floor lamps)
        gt["lamp"] = sum(v for k, v in full.items() if k.endswith("lamp")) if full else item["baseline"].get("bedside lamp", 0)
        out.append({"source": "frozen", "images": [path], "classes": MAIN, "gt": {c: gt[c] for c in MAIN}})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/eval/eval_set.jsonl")
    p.add_argument("--test", default="data/sft/test.jsonl", help="held-out rows written by mix_data.py")
    p.add_argument("--home-val", default="data/home_val/count_sft.jsonl")
    p.add_argument("--frozen", default="", help="folder with eval_frozen.json + images/ (optional)")
    p.add_argument("--max-per-source", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    by_src: dict[str, list[dict]] = {}
    if a.frozen:
        by_src["frozen"] = frozen_rows(a.frozen)
    frozen_names = {os.path.basename(r["images"][0]) for r in by_src.get("frozen", [])}
    if os.path.exists(a.home_val):
        rows = [from_row(json.loads(l), "homeobjects_val") for l in open(a.home_val)]
        by_src["homeobjects_val"] = [r for r in rows if os.path.basename(r["images"][0]) not in frozen_names]
    if os.path.exists(a.test):
        for l in open(a.test):
            r = json.loads(l)
            src = r.pop("source")
            if src != "home":                            # HomeObjects is tested on its own val split above
                by_src.setdefault(src, []).append(from_row(r, src))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for src, rows in by_src.items():
            random.shuffle(rows)
            rows = rows[:a.max_per_source] if src != "frozen" else rows
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n_cls = len({c for r in rows for c in r["classes"]})
            print(f"  {src:16s} {len(rows):4d} images, {n_cls} classes asked")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
