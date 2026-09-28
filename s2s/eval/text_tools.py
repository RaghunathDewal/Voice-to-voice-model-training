"""Experiment 2: tool-call accuracy of the thinker from TEXT input (no speech).

    python -m s2s.eval.text_tools --manifest data/manifests/hotel_eval_text.jsonl
    python -m s2s.eval.text_tools --manifest ... --lora-dir checkpoints/speech_llm_tools/lora

If this is low, speech input will not be better: fix it with LoRA (stage 3) or
a larger thinker before blaming the speech path.
"""

from __future__ import annotations

import collections

import torch
from tqdm import tqdm

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.data.hotel import score_example, tools_by_name
from s2s.models.speech_llm import greedy_generate
from s2s.models.thinker import Thinker
from s2s.utils import resolve_device, resolve_dtype, save_json


def extract_calls(text: str) -> list[dict]:
    calls = []
    while "<tool_call>" in text:
        start = text.index("<tool_call>") + len("<tool_call>")
        end = text.find("</tool_call>", start)
        if end < 0:
            break
        call = Thinker.parse_tool_call(text[start:end])
        calls.append(call or {"name": None})
        text = text[end + len("</tool_call>"):]
    return calls


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--manifest", default="data/manifests/hotel_eval_text.jsonl")
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--model", default=None, help="override thinker (e.g. checkpoints/thinker_merged)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--out", default=None, help="write per-example results JSON")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    thinker = Thinker(args.model or cfg.thinker.model, device, dtype, lora_dir=args.lora_dir, merge_lora=True,
                      attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    rows = load_manifest(args.manifest)
    if args.max:
        rows = rows[: args.max]
    per_tool = collections.defaultdict(lambda: [0, 0])
    results = []
    for r in tqdm(rows, desc="text tool eval"):
        system = Thinker.system_content(cfg.thinker.system_prompt, r.get("context"))
        ids = thinker.prompts.text_prompt_ids(system, r["text"], tools_by_name(r.get("tools")))
        emb = thinker.embed(torch.tensor([ids], device=device))
        out = thinker.tokenizer.decode(greedy_generate(thinker, emb, 120), skip_special_tokens=False)
        pred = extract_calls(out)
        ok = score_example(r, pred, out)
        name = r["tool_calls"][0]["name"] if r.get("tool_calls") else "(answer from context)"
        per_tool[name][0] += int(ok)
        per_tool[name][1] += 1
        results.append({"text": r["text"], "gold": r.get("tool_calls") or r.get("reply"), "pred": pred, "raw": out, "correct": ok})
    total = sum(v[0] for v in per_tool.values()) / max(1, sum(v[1] for v in per_tool.values()))
    print(f"\nexact tool-call accuracy: {total * 100:.1f}% on {len(results)} examples")
    for name, (c, n) in sorted(per_tool.items()):
        print(f"  {name:28s} {c}/{n} = {c / n * 100:.1f}%")
    for r in [x for x in results if not x["correct"]][:5]:
        print(f"  MISS: {r['text']}\n    gold {r['gold']}\n    got  {r['raw']!r}")
    if args.out:
        save_json(args.out, {"accuracy": total, "results": results})


if __name__ == "__main__":
    main()
