"""(vision/vl_infer.py) The one inference path, used by the eval (eval_counts.py) and production (infer_room.py).

Same as training: the prompts of common.py, images resized by qwen_vl_utils to IMAGE_MAX_TOKEN_NUM x 32 x 32 pixels
(what ms-swift does), greedy decoding, fp16 on T4 / bf16 on newer GPUs. Questions are batched (left padding); a batch
that fails falls back to one question at a time, so a batching problem costs speed, never results.
"""

from __future__ import annotations

import contextlib
import json
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from common import COUNT_PROMPT, EXTRA_PROMPT


def default_dtype() -> str:
    ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    return os.environ.get("DTYPE") or ("bfloat16" if ok else "float16")


def parse_counts(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        objs = json.loads(m.group(0)).get("objects", []) if m else []
        return {str(o["label"]).lower(): int(o["detected_count"]) for o in objs}
    except Exception:  # noqa: BLE001
        return {}


def parse_new_items(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        items = json.loads(m.group(0)).get("new_items", []) if m else []
        return {str(o["label"]).lower(): int(o.get("count", 1)) for o in items}
    except Exception:  # noqa: BLE001
        return {}


class RoomModel:
    def __init__(self, model: str = "Qwen/Qwen3-VL-4B-Instruct", adapter: str | None = None, dtype: str | None = None,
                 quant4: bool = False, image_tokens: int | None = None, device_map="auto"):
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype or default_dtype()]
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dt) if quant4 else None
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, quantization_config=q, dtype=dt,
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
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        images, _ = process_vision_info(msgs, image_patch_size=16)
        inp = self.processor(text=texts, images=images, padding=True, do_resize=False,
                             return_tensors="pt").to(self.model.device)
        off = self.model.disable_adapter() if (base_only and self.has_adapter) else contextlib.nullcontext()
        with torch.inference_mode(), off:
            gen = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return self.processor.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)

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

    def new_items(self, images: list[str], expected: list[list[str]], batch: int = 4,
                  base_only: bool = False) -> list[tuple[dict, str]]:
        """Call 2: objects NOT on the expected list. base_only=True switches the LoRA off for this call."""
        items = [(img, EXTRA_PROMPT.format(names=", ".join(names))) for img, names in zip(images, expected)]
        replies = self.generate(items, max_new=400, batch=batch, base_only=base_only)
        return [(parse_new_items(r), r) for r in replies]
