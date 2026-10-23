"""(vision/mix_data.py) Combine the prepared row files into one train.jsonl + val.jsonl for ms-swift.

    python mix_data.py --out data/sft --ground-share 0.2 \
        --count data/home/count_sft.jsonl:1.0 data/lvis/count_sft.jsonl:1.0 my_hotel_count.jsonl:2.0 \
        --ground data/home/ground_sft.jsonl:1.0 data/lvis/ground_sft.jsonl:1.0

--total 0 (default) = every count row of every image, plus grounding rows to make --ground-share of the total.
FILE:WEIGHT = relative share of that file inside its task (count or ground). A file that runs out is used whole.
Any file in the same {"messages", "images"} format can be added (e.g. your own hotel photos, auto-labelled).
Rows of one image all go to the same split, so val images are never seen in training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zlib


def load(spec: str) -> tuple[list[dict], float]:
    path, _, w = spec.rpartition(":") if ":" in spec else (spec, "", "1")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows, float(w)


def take(specs: list[str], n: int) -> list[dict]:
    if not specs or n <= 0:
        return []
    files = [load(s) for s in specs]
    tot_w = sum(w for _, w in files)
    out = []
    for (rows, w), spec in zip(files, specs):
        k = min(len(rows), round(n * w / tot_w))
        out += random.sample(rows, k)
        print(f"  {spec}: {k} of {len(rows)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/sft")
    p.add_argument("--count", nargs="*", default=[])
    p.add_argument("--ground", nargs="*", default=[])
    p.add_argument("--total", type=int, default=0, help="rows in train + val together (0 = every count row of every image)")
    p.add_argument("--ground-share", type=float, default=0.2)
    p.add_argument("--val-share", type=float, default=0.01)
    p.add_argument("--val-max", type=int, default=300, help="cap on val rows (each eval pass costs training time)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    print("count rows:")
    n_count = round(a.total * (1 - a.ground_share)) if a.total else 10 ** 9
    rows = take(a.count, n_count)
    print("grounding rows:")
    rows += take(a.ground, round(len(rows) * a.ground_share / (1 - a.ground_share)))
    random.shuffle(rows)
    os.makedirs(a.out, exist_ok=True)
    is_val = lambda r: zlib.crc32(r["images"][0].encode()) % 10000 < a.val_share * 10000  # noqa: E731
    nv = 0
    with open(f"{a.out}/train.jsonl", "w") as ft, open(f"{a.out}/val.jsonl", "w") as fv:
        for r in rows:
            if is_val(r):
                if nv < a.val_max:                       # extra val rows are dropped, never trained on
                    fv.write(json.dumps(r) + "\n")
                    nv += 1
            else:
                ft.write(json.dumps(r) + "\n")
    nt = sum(1 for _ in open(f"{a.out}/train.jsonl"))
    imgs = {json.loads(l)["images"][0] for l in open(f"{a.out}/train.jsonl")}
    print(f"{nt} train rows on {len(imgs)} images, {nv} val rows -> {a.out}/")


if __name__ == "__main__":
    main()
