"""(vision/eval_counts.py) Score the BASE model and the fine-tuned ADAPTER the same way on your frozen eval set.

    python eval_counts.py --eval-dir /kaggle/input/<eval>/eval_data --adapter out/qwen3vl-room/v0-.../checkpoint-XXX
    python eval_counts.py --eval-dir ... --adapter none          # base model only

Same model loading (4-bit), same prompt as training, same image size (IMAGE_MAX_TOKEN_NUM x 32 x 32 pixels,
exactly what ms-swift used in training), greedy decoding. Reports, per class:
  acc        exact count right, all images
  present    exact count right on images that HAVE the object (the number that collapsed last time)
  absent     says 0 on images without it
  floor      what "always answer 0" would score: a model must beat this to have learned anything

Lamp: training uses one class "lamp" (any lamp: LVIS and HomeObjects cannot tell bedside from floor lamps),
so its ground truth here = every "...lamp" in full_counts, not baseline["bedside lamp"].
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

CLASSES = ["pillow", "towel", "lamp", "chair"]
COUNT_PROMPT = ("Count the following objects in this image: {names}. "
                'Answer in JSON: {{"objects":[{{"label":...,"detected_count":...}}]}}')


def ground_truth(item: dict) -> dict:
    gt = {c: item["baseline"].get(c, 0) for c in ("pillow", "towel", "chair")}
    full = item.get("full_counts") or {}
    gt["lamp"] = sum(v for k, v in full.items() if k.endswith("lamp")) if full else item["baseline"].get("bedside lamp", 0)
    return gt


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        objs = json.loads(m.group(0)).get("objects", []) if m else []
        return {o["label"]: int(o["detected_count"]) for o in objs}
    except Exception:  # noqa: BLE001
        return {}


def load(base: str, adapter: str | None):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(base, quantization_config=q, torch_dtype=torch.bfloat16,
                                                            device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def run(model, processor, eval_dir: str, frozen: dict, max_pixels: int) -> dict:
    preds = {}
    for i, name in enumerate(frozen):
        path = os.path.join(eval_dir, "images", name)
        if not os.path.exists(path):
            continue
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": path, "max_pixels": max_pixels},
            {"type": "text", "text": COUNT_PROMPT.format(names=", ".join(CLASSES))}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(msgs, image_patch_size=16)
        inp = processor(text=[text], images=images, do_resize=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False)
        reply = processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        preds[name] = {"reply": reply, "pred": parse(reply)}
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(frozen)}", flush=True)
    return preds


def score(frozen: dict, preds: dict) -> dict:
    res = {}
    for c in CLASSES + ["ALL"]:
        cs = CLASSES if c == "ALL" else [c]
        rows = [(ground_truth(frozen[n])[k], p["pred"].get(k)) for n, p in preds.items() for k in cs]
        pres = [(g, q) for g, q in rows if g > 0]
        absn = [(g, q) for g, q in rows if g == 0]
        pct = lambda r: round(100 * sum(g == q for g, q in r) / max(1, len(r)), 1)  # noqa: E731
        res[c] = {"acc": pct(rows), "present": pct(pres), "absent": pct(absn), "n_present": len(pres),
                  "floor": round(100 * len(absn) / max(1, len(rows)), 1),
                  "off_by_1": round(100 * sum(q is not None and abs(g - q) <= 1 for g, q in rows) / max(1, len(rows)), 1)}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="folder with eval_frozen.json and images/")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter", default="none", help="LoRA checkpoint dir, or 'none' for the base model only")
    p.add_argument("--image-tokens", type=int, default=int(os.environ.get("IMAGE_MAX_TOKEN_NUM", 768)))
    p.add_argument("--out", default="eval_results.json")
    a = p.parse_args()
    frozen = json.load(open(os.path.join(a.eval_dir, "eval_frozen.json")))
    processor = AutoProcessor.from_pretrained(a.model)
    max_pixels = a.image_tokens * 32 * 32
    report = {"image_tokens": a.image_tokens}
    runs = [("base", None)] + ([] if a.adapter == "none" else [("finetuned", a.adapter)])
    for tag, adapter in runs:
        print(f"===== {tag}", flush=True)
        model = load(a.model, adapter)
        preds = run(model, processor, a.eval_dir, frozen, max_pixels)
        report[tag] = {"scores": score(frozen, preds), "preds": preds}
        del model
        torch.cuda.empty_cache()
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n{'':10s}" + "".join(f"{t:>44s}" for t, _ in runs))
    print(f"{'class':10s}" + "".join(f"{'acc  present  absent  off-by-1  floor':>44s}" for _ in runs))
    for c in CLASSES + ["ALL"]:
        line = f"{c:10s}"
        for t, _ in runs:
            s = report[t]["scores"][c]
            line += f"{s['acc']:>15.1f}{s['present']:>9.1f}{s['absent']:>8.1f}{s['off_by_1']:>10.1f}{s['floor']:>7.1f}"
        print(line)
    print(f"results: {a.out}")


if __name__ == "__main__":
    main()
