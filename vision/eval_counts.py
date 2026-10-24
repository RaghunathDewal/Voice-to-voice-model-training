"""(vision/eval_counts.py) Score the BASE model and the fine-tuned ADAPTER the same way on the mixed eval set.

    python eval_counts.py run --eval-set data/eval/eval_set.jsonl --tag base --out pred_base.json
    python eval_counts.py run --eval-set data/eval/eval_set.jsonl --tag finetuned --adapter out/.../checkpoint-N --out pred_ft.json
    python eval_counts.py report pred_base.json pred_ft.json --out eval_results.json

(run_kaggle.sh eval runs base on GPU 0 and finetuned on GPU 1 at the same time, then the report.)
Same model loading as training (DTYPE, or 4-bit with --quant4), same prompt as training, same image size (IMAGE_MAX_TOKEN_NUM x 32 x 32 pixels, what
ms-swift used in training), greedy decoding. Per source (frozen / homeobjects_val / lvis / ...) and per class:
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
import re
from collections import defaultdict

from common import COUNT_PROMPT

MAIN = ["pillow", "towel", "lamp", "chair"]


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        objs = json.loads(m.group(0)).get("objects", []) if m else []
        return {str(o["label"]).lower(): int(o["detected_count"]) for o in objs}
    except Exception:  # noqa: BLE001
        return {}


def run(a) -> None:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    rows = [json.loads(l) for l in open(a.eval_set)]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype]
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=dtype) if a.quant4 else None   # same base as training
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, quantization_config=q, dtype=dtype,
                                                            device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(a.model)
    max_pixels = a.image_tokens * 32 * 32
    out = []
    for i, r in enumerate(rows):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": r["images"][0], "max_pixels": max_pixels},
            {"type": "text", "text": COUNT_PROMPT.format(names=", ".join(r["classes"]))}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(msgs, image_patch_size=16)
        inp = processor(text=[text], images=images, do_resize=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=60 + 25 * len(r["classes"]), do_sample=False)
        reply = processor.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        out.append({**r, "reply": reply, "pred": parse(reply)})
        if (i + 1) % 25 == 0:
            print(f"  [{a.tag}] {i + 1}/{len(rows)}", flush=True)
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
        if key == "per class (all)":
            continue
        s0 = table[tags[0]][key]
        line = f"{key[:30]:30s}{s0['n']:6d}{s0['n_missing']:8d} "
        for t in tags:
            line += f"{'':3s}" + "".join(f(table[t][key][k]) for k, _ in COLS)
        print(line)
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
    r.add_argument("--dtype", default=os.environ.get("DTYPE", "float16"), choices=["float16", "bfloat16"],
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
