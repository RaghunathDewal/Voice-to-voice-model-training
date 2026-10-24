"""(vision/infer_room.py) Production check of one room: before inventory + after photo(s) -> missing / changed / new.

    python infer_room.py --before before.json --images after_1.jpg after_2.jpg \
        --adapter ThunderBlade7773/qwen3vl-room-lora-v3 --out report.json
    before.json: {"pillow": 4, "towel": 2, "lamp": 2, "kettle": 1, "remote control": 1}

Call 1 (fine-tuned LoRA): count every expected item in every photo. Call 2 (LoRA switched off, base model): list
objects that are NOT on the list. The model only counts; everything below is plain code you can change:
  - several photos of one room: an item's count is the MAX over the photos (COMBINE=sum if the photos show
    different, non-overlapping parts of the room)
  - status per item: ok / missing (before > 0, after 0) / fewer / more, with a tolerance per item if you want one
  - new items: what call 2 lists, minus expected items and room parts (common.clean_new_items)
A count is per photo: an item out of frame counts as 0, so take the photos the same way as the "before" ones.
"""

from __future__ import annotations

import argparse
import json
import os

from common import clean_new_items


def resolve_adapter(a: str | None) -> str | None:
    """Local folder, or an HF repo id whose last-checkpoint holds the LoRA."""
    if not a or os.path.isdir(a):
        return a
    from huggingface_hub import snapshot_download
    d = snapshot_download(a, allow_patterns=["last-checkpoint/adapter_*", "last-checkpoint/*.json"])
    return os.path.join(d, "last-checkpoint")


def compare(before: dict, after: dict, tolerance: dict) -> list[dict]:
    rows = []
    for item, b in before.items():
        n = after.get(item)
        if n is None:
            status = "unknown"                         # the model did not answer for this item
        elif b > 0 and n == 0:
            status = "missing"
        elif abs(n - b) <= tolerance.get(item, 0):
            status = "ok"
        else:
            status = "fewer" if n < b else "more"
        rows.append({"item": item, "before": b, "after": n, "status": status})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", required=True, help="JSON file {item: count} from the previous inspection")
    p.add_argument("--images", nargs="+", required=True, help="after photo(s) of the same room")
    p.add_argument("--adapter", default="ThunderBlade7773/qwen3vl-room-lora-v3")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--combine", choices=["max", "sum"], default=os.environ.get("COMBINE", "max"))
    p.add_argument("--tolerance", default="{}", help='JSON {item: allowed difference}, e.g. {"pillow": 1}')
    p.add_argument("--new-items-with-lora", action="store_true",
                   help="ask call 2 with the LoRA on (default off: fine-tuning narrows open-ended listing)")
    p.add_argument("--out", default="")
    a = p.parse_args()
    from vl_infer import RoomModel

    # item names as in training (prep_lvis.py GROUPS / prep_homeobjects.py NAMES), e.g. "remote control", "towel"
    before = {k.lower().strip(): int(v) for k, v in json.load(open(a.before)).items()}
    names = list(before)
    m = RoomModel(a.model, adapter=resolve_adapter(a.adapter))
    counted = m.count(a.images, [names] * len(a.images))
    after: dict = {}
    for pred, _ in counted:
        for item in names:
            if item in pred:
                v = pred[item]
                after[item] = v if item not in after else (max(after[item], v) if a.combine == "max" else after[item] + v)
    found = m.new_items(a.images, [names] * len(a.images), base_only=not a.new_items_with_lora)
    new: dict = {}
    for pred, _ in found:
        for k, v in clean_new_items(pred, names).items():
            new[k] = max(new.get(k, 0), v)
    report = {"items": compare(before, after, json.loads(a.tolerance)),
              "new_items": [{"item": k, "count": v} for k, v in sorted(new.items())],
              "per_photo": [{"image": img, "counts": c, "new_items": n}
                            for img, (c, _), (n, _) in zip(a.images, counted, found)]}
    alerts = [r for r in report["items"] if r["status"] != "ok"]
    print(json.dumps({"alerts": alerts, "new_items": report["new_items"]}, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
