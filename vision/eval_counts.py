"""(vision/eval_counts.py) Score the BASE model and the fine-tuned ADAPTER the same way on the mixed eval set.

    python eval_counts.py run --eval-set data/eval/eval_set.jsonl --tag base --out pred_base.json
    python eval_counts.py run --eval-set data/eval/eval_set.jsonl --tag finetuned --adapter out/.../checkpoint-N --out pred_ft.json
    python eval_counts.py report pred_base.json pred_ft.json --out eval_results.json

(run_kaggle.sh eval spreads the models over the GPUs, then writes one report.)
Everything goes through vl_infer.RoomModel (the same path production uses): same loading as training, same prompts
(common.py), same image size (IMAGE_MAX_TOKEN_NUM x 32 x 32), greedy decoding, batched. Per source (frozen / homeobjects_val / lvis / ...) and per class:
  acc             exact count right
  present         exact count right when the item is there
  missing_recall  the item is gone and the model says 0: the alert is caught
  false_alarm     the item is there but the model says 0 (lower is better)
  off1            within +-1
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from common import MAIN, clean_new_items, same_object


def score_extra(rows: list[dict]) -> dict | None:
    """New-items check on what production code would keep (count > 0, not an expected item, not a room part):
    recall = real other objects found, precision = listed objects that are real."""
    rows = [r for r in rows if "extra_pred" in r]
    if not rows:
        return None
    gt_n = hit_gt = pred_n = hit_pred = 0
    for r in rows:
        gt, pred = list(r["extra_gt"]), list(clean_new_items(r["extra_pred"], r["classes"]))
        gt_n += len(gt)
        pred_n += len(pred)
        hit_gt += sum(any(same_object(g, p) for p in pred) for g in gt)
        hit_pred += sum(any(same_object(p, g) for g in gt) for p in pred)
    return {"images": len(rows), "recall": round(100 * hit_gt / max(1, gt_n), 1),
            "precision": round(100 * hit_pred / max(1, pred_n), 1), "listed_per_image": round(pred_n / len(rows), 1),
            "real_per_image": round(gt_n / len(rows), 1)}


def run(a) -> None:
    from vl_infer import RoomModel

    rows = [json.loads(l) for l in open(a.eval_set)]
    m = RoomModel(a.model, adapter=a.adapter or None, dtype=a.dtype, quant4=a.quant4, image_tokens=a.image_tokens)
    out = []
    step = 24
    for i in range(0, len(rows), step):                # chunks: progress + partial results on disk
        chunk = rows[i:i + step]
        counted = m.count([r["images"][0] for r in chunk], [r["classes"] for r in chunk], batch=a.batch)
        res = [{**r, "reply": rep, "pred": pred} for r, (pred, rep) in zip(chunk, counted)]
        extra = [k for k, r in enumerate(chunk) if "extra_gt" in r]   # frozen rooms: full inventory known
        if extra:
            found = m.new_items([chunk[k]["images"][0] for k in extra], [chunk[k]["classes"] for k in extra],
                                batch=a.batch)
            for k, (pred, rep) in zip(extra, found):
                res[k].update(extra_reply=rep, extra_pred=pred)
        out += res
        print(f"  [{a.tag}] {len(out)}/{len(rows)}", flush=True)
        json.dump({"tag": a.tag, "adapter": a.adapter, "rows": out}, open(a.out, "w"))
    json.dump({"tag": a.tag, "adapter": a.adapter, "image_tokens": a.image_tokens, "rows": out}, open(a.out, "w"))
    print(f"[{a.tag}] done -> {a.out}")


def score(rows: list[dict], cls_filter=None) -> dict:
    pairs = [(g, r["pred"].get(c)) for r in rows for c, g in r["gt"].items() if cls_filter is None or c in cls_filter]
    pres = [(g, p) for g, p in pairs if g > 0]
    absn = [(g, p) for g, p in pairs if g == 0]
    pct = lambda v: round(100 * sum(g == p for g, p in v) / len(v), 1) if v else None  # noqa: E731
    return {"n": len(pairs), "n_present": len(pres), "n_missing": len(absn),
            "acc": pct(pairs),                                    # exact count, every question
            "present": pct(pres),                                 # exact count when the item is there
            "missing_recall": pct(absn),                          # item gone -> says 0 (alert caught)
            "false_alarm": round(100 * sum(p == 0 for _, p in pres) / len(pres), 1) if pres else None,  # there, says 0
            "off1": round(100 * sum(p is not None and abs(g - p) <= 1 for g, p in pairs) / len(pairs), 1) if pairs else None,
            "floor": round(100 * len(absn) / len(pairs), 1) if pairs else None}


COLS = [("acc", "acc"), ("present", "pres"), ("missing_recall", "miss"), ("false_alarm", "f.al"), ("off1", "off1")]


def report(a) -> None:
    runs = [json.load(open(f)) for f in a.preds]
    table = {}
    for run_ in runs:
        by_src = defaultdict(list)
        for r in run_["rows"]:
            by_src[r["source"]].append(r)
        t = {"ALL: every class": score(run_["rows"]),
             "ALL: pillow towel lamp chair": score(run_["rows"], MAIN)}
        for s_, rs in sorted(by_src.items()):
            t[s_] = score(rs)
        for c in MAIN:
            t[f"class {c}"] = score(run_["rows"], {c})
        per_cls = defaultdict(list)
        for r in run_["rows"]:
            for c in r["gt"]:
                per_cls[c].append(r)
        t["per class (all)"] = {c: score(rs, {c}) for c, rs in sorted(per_cls.items())}
        t["new items (frozen rooms)"] = score_extra(run_["rows"])
        table[run_["tag"]] = t
    json.dump({"scores": table, "preds": {r["tag"]: r["rows"] for r in runs}}, open(a.out, "w"), indent=1)
    tags = [r["tag"] for r in runs]
    f = lambda v: f"{v:6.1f}" if v is not None else "     -"  # noqa: E731
    w = 6 * len(COLS) + 3
    print("\nacc = exact count | pres = exact when the item is there | miss = item gone and the model says 0 (alert "
          "caught) | f.al = item there but the model says 0 (false alarm, lower is better) | off1 = within +-1")
    print(f"{'':30s}{'n':>6s}{'missing':>8s} " + "".join(f"{t:>{w}s}" for t in tags))
    print(f"{'':45s}" + "".join(f"{'':3s}" + "".join(f"{h:>6s}" for _, h in COLS) for _ in tags))
    for key in table[tags[0]]:
        if key in ("per class (all)", "new items (frozen rooms)"):
            continue
        s0 = table[tags[0]][key]
        line = f"{key[:30]:30s}{s0['n']:6d}{s0['n_missing']:8d} "
        for t in tags:
            line += f"{'':3s}" + "".join(f(table[t][key][k]) for k, _ in COLS)
        print(line)
    ex = {t: table[t]["new items (frozen rooms)"] for t in tags}
    if any(ex.values()):
        print("\nNEW ITEMS (not in the expected list), 79 frozen rooms with a full inventory; labels matched loosely:")
        for t, e in ex.items():
            if e:
                print(f"  {t:20s} finds {e['recall']:5.1f}% of the real other objects | {e['precision']:5.1f}% of what it "
                      f"lists is real | lists {e['listed_per_image']} per room (real: {e['real_per_image']})")
    print(f"(per-class numbers for every class: {a.out})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--eval-set", required=True)
    r.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    r.add_argument("--adapter", default="", help="LoRA checkpoint dir; empty = base model")
    r.add_argument("--tag", default="base")
    r.add_argument("--image-tokens", type=int, default=int(os.environ.get("IMAGE_MAX_TOKEN_NUM", 768)))
    r.add_argument("--batch", type=int, default=int(os.environ.get("EVAL_BATCH", 4)), help="questions per generate call")
    r.add_argument("--dtype", default=os.environ.get("DTYPE") or None, choices=["float16", "bfloat16"],
                   help="same as training (run_kaggle.sh: fp16 on T4, bf16 on A100/L4)")
    r.add_argument("--quant4", action="store_true", help="load the base in 4-bit (only if training used QUANT=1)")
    r.add_argument("--out", required=True)
    s = sub.add_parser("report")
    s.add_argument("preds", nargs="+")
    s.add_argument("--out", default="eval_results.json")
    a = p.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
