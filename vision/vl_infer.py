"""(vision/vl_infer.py) The one inference path, used by the eval (eval_counts.py) and production (infer_room.py).

Same as training: the prompts of common.py, images resized by qwen_vl_utils to IMAGE_MAX_TOKEN_NUM x 32 x 32 pixels
(what ms-swift does), greedy decoding, fp16 on T4 / bf16 on newer GPUs. Questions are batched (left padding); a batch
that fails falls back to one question at a time, so a batching problem costs speed, never results.

Works for Qwen3-VL (our LoRAs) and Qwen3.5 (same processor and 32x32 image tokens): the model class comes from the
checkpoint's config, and thinking is switched off (Qwen3.5 thinks by default; Qwen3-VL ignores the switch).
"""

from __future__ import annotations

import contextlib
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from common import COUNT_PROMPT, EXTRA_PROMPT, GROUND_PROMPT

THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def default_dtype() -> str:
    ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    return os.environ.get("DTYPE") or ("bfloat16" if ok else "float16")


_ITEM = re.compile(r'\{[^{}]*?"label"\s*:\s*"([^"]+)"[^{}]*?\}', re.DOTALL)


def _items(text: str, keys: tuple) -> dict:
    """Every {"label": ..., <count key>: n} object in the answer, however it is wrapped: {"objects": [...]}, a bare
    list, a code fence, numbers as strings ("2"), or a list cut off by the token limit (complete objects are kept).
    Qwen3-VL writes {"objects": [...]}; Qwen3.5 writes a bare list in a ```json fence."""
    out = {}
    for m in _ITEM.finditer(THINK.sub("", text)):
        obj = m.group(0)
        for k in keys:
            n = re.search(r'"%s"\s*:\s*"?\s*(-?\d+)' % k, obj)
            if n:
                out[m.group(1).strip().lower()] = int(n.group(1))
                break
    return out


def parse_counts(text: str) -> dict:
    return _items(text, ("detected_count", "count", "number", "quantity"))


def parse_new_items(text: str) -> dict:
    return _items(text, ("count", "detected_count", "number", "quantity"))


def _iou(a: list, b: list) -> float:
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_boxes(text: str, iou_dup: float = 0.85) -> list[list[int]] | None:
    """Every [x1, y1, x2, y2] after a "bbox_2d" key (tolerates code fences / truncated lists); near-identical boxes
    (IoU > iou_dup, a known repeat failure of small VLMs) count once. None = no parsable answer at all."""
    text = THINK.sub("", text)
    found = re.findall(r'"bbox_2d"\s*:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',
                       text)
    if not found:
        return [] if re.search(r"\[\s*\]", text) else None
    boxes: list[list[int]] = []
    for b in ([int(float(v)) for v in f] for f in found):
        if b[2] > b[0] and b[3] > b[1] and all(_iou(b, k) <= iou_dup for k in boxes):
            boxes.append(b)
    return boxes


class RoomModel:
    def __init__(self, model: str = "Qwen/Qwen3-VL-4B-Instruct", adapter: str | None = None, dtype: str | None = None,
                 quant4: bool = False, image_tokens: int | None = None, device_map="auto"):
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype or default_dtype()]
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dt) if quant4 else None
        self.model = AutoModelForImageTextToText.from_pretrained(model, quantization_config=q, dtype=dt,
                                                                 device_map=device_map)
        self.has_adapter = bool(adapter)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self.processor.tokenizer.padding_side = "left"      # batched generation
        self.max_pixels = (image_tokens or int(os.environ.get("IMAGE_MAX_TOKEN_NUM", 768))) * 32 * 32

    def _generate(self, items: list[tuple[str, str]], max_new: int, base_only: bool) -> list[str]:
        msgs = [[{"role": "user", "content": [{"type": "image", "image": img, "max_pixels": self.max_pixels},
                                              {"type": "text", "text": prompt}]}] for img, prompt in items]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for m in msgs]
        images, _ = process_vision_info(msgs, image_patch_size=16)
        inp = self.processor(text=texts, images=images, padding=True, do_resize=False,
                             return_tensors="pt").to(self.model.device)
        off = self.model.disable_adapter() if (base_only and self.has_adapter) else contextlib.nullcontext()
        with torch.inference_mode(), off:
            gen = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return [THINK.sub("", r) for r in
                self.processor.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)]

    def generate(self, items: list[tuple[str, str]], max_new: int = 300, batch: int = 4, base_only: bool = False):
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            try:
                out += self._generate(chunk, max_new, base_only)
            except Exception as e:  # noqa: BLE001  (e.g. out of memory on a big batch)
                if len(chunk) == 1:
                    raise
                print(f"  batch of {len(chunk)} failed ({type(e).__name__}), one by one", flush=True)
                torch.cuda.empty_cache()
                out += [self._generate([it], max_new, base_only)[0] for it in chunk]
        return out

    def count(self, images: list[str], expected: list[list[str]], batch: int = 4) -> list[tuple[dict, str]]:
        """Call 1: count the expected items (0 = not there). One list of names per image."""
        items = [(img, COUNT_PROMPT.format(names=", ".join(names))) for img, names in zip(images, expected)]
        longest = max(len(n) for n in expected) if expected else 1
        replies = self.generate(items, max_new=60 + 25 * longest, batch=batch)
        return [(parse_counts(r), r) for r in replies]

    def locate(self, pairs: list[tuple[str, str]], batch: int = 4, max_new: int = 600) -> list[tuple[list | None, str]]:
        """"Locate every <item>" per (image, item): the boxes (None if the answer has none parsable)."""
        replies = self.generate([(img, GROUND_PROMPT.format(name=name)) for img, name in pairs], max_new=max_new,
                                batch=batch)
        return [(parse_boxes(r), r) for r in replies]

    def new_items(self, images: list[str], expected: list[list[str]], batch: int = 4,
                  base_only: bool = False) -> list[tuple[dict, str]]:
        """Call 2: objects NOT on the expected list. base_only=True switches the LoRA off for this call."""
        items = [(img, EXTRA_PROMPT.format(names=", ".join(names))) for img, names in zip(images, expected)]
        # 800: rooms have 10+ other objects and pretty-printed JSON is long (400 cut answers off in the bench)
        replies = self.generate(items, max_new=800, batch=batch, base_only=base_only)
        return [(parse_new_items(r), r) for r in replies]
