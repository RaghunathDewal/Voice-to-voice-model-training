"""Is the hotel-tuned thinker better at the hotel AND still itself everywhere else? Base vs tuned, from text.

  hotel      typed hotel requests (gold rows): exact tool call, or the right fact without a call
  variety    distinct wording of the answers that are not tool calls (1.0 = no two alike); a drop means
             the tuned model fell back to template sentences
  general    held-out general requests (s2s.prep.brain_sft): the tuned answer judged against the base
             model's own answer by the base model ("would the guest be equally well served")
  length     mean words per answer, base vs tuned

The tuned model is used only when hotel improves, general >= --min-general and variety keeps >= --min-variety
of the base's. Writes <out> (JSON, with examples) and prints a readable summary.

    python -m s2s.eval.brain_check --config configs/qwen4b_v3.yaml --base Qwen/Qwen3-4B-Instruct-2507 \\
        --tuned checkpoints/thinker_4b_hotel --hotel data/manifests/hotel5_eval_nm.jsonl \\
        --general data/manifests/brain_general_check.jsonl --out logs/brain_check.json
"""

from __future__ import annotations

import gc

import torch

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.data.hotel import row_system, score_example, tools_by_name
from s2s.eval.speech_vs_text import FAIR_JUDGE_SYSTEM
from s2s.eval.text_tools import extract_calls
from s2s.models.speech_llm import greedy_generate
from s2s.models.thinker import Thinker
from s2s.utils import resolve_device, resolve_dtype, save_json

GENERAL_JUDGE = FAIR_JUDGE_SYSTEM.replace("a hotel voice assistant", "a voice assistant")


def clean(raw: str) -> str:
    for tok in ("<|im_end|>", "<|endoftext|>"):
        raw = raw.split(tok)[0]
    return raw.strip()


def variety(texts: list[str]) -> float:
    norm = [" ".join(t.lower().split()) for t in texts if t.strip()]
    return len(set(norm)) / max(1, len(norm))


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--tuned", required=True)
    p.add_argument("--hotel", required=True, help="gold hotel rows (e.g. hotel5_eval_nm.jsonl)")
    p.add_argument("--general", required=True, help="from s2s.prep.brain_sft (text + the base model's response)")
    p.add_argument("--max-hotel", type=int, default=120)
    p.add_argument("--max-general", type=int, default=100)
    p.add_argument("--min-general", type=float, default=0.85)
    p.add_argument("--min-variety", type=float, default=0.8, help="fraction of the base model's variety")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    hotel = load_manifest(args.hotel)[: args.max_hotel]
    general = load_manifest(args.general)[: args.max_general]

    def answer(th: Thinker, system: str, text: str, tools=None, history=None, max_new: int = 120) -> str:
        ids = th.prompts.text_prompt_ids(system, text, tools, history)
        emb = th.embed(torch.tensor([ids], device=device))
        return th.tokenizer.decode(greedy_generate(th, emb, max_new), skip_special_tokens=False)

    def hotel_run(th: Thinker) -> dict:
        ok, replies, rows = 0, [], []
        for r in hotel:
            raw = answer(th, row_system(r, cfg.thinker.system_prompt), r["text"], tools_by_name(r.get("tools")),
                         r.get("history"))
            calls = extract_calls(raw)
            good = score_example(r, calls, raw)
            ok += int(good)
            if not calls:
                replies.append(clean(raw))
            rows.append({"text": r["text"], "answer": clean(raw), "calls": calls, "correct": good})
        return {"accuracy": ok / max(1, len(hotel)), "variety": variety(replies), "rows": rows,
                "words": sum(len(x.split()) for x in replies) / max(1, len(replies))}

    def load(name: str) -> Thinker:
        th = Thinker(name, device, dtype, attn_implementation=cfg.thinker.attn_implementation)
        th.model.eval()
        return th

    with torch.no_grad():
        tuned = load(args.tuned)
        res_t = hotel_run(tuned)
        gen_t = [clean(answer(tuned, cfg.thinker.system_prompt, r["text"], max_new=100)) for r in general]
        del tuned
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None
        base = load(args.base)
        res_b = hotel_run(base)
        same = []
        for r, t in zip(general, gen_t):
            b = r["response"].strip()
            if " ".join(t.split()) == " ".join(b.split()):
                same.append(True)
                continue
            v = answer(base, GENERAL_JUDGE, f"Guest message: {r['text']}\nReply A: {b}\nReply B: {t}", max_new=3)
            same.append(clean(v).upper().startswith("SAME"))

    general_same = sum(same) / max(1, len(same))
    words_b = sum(len(r["response"].split()) for r in general) / max(1, len(general))
    words_t = sum(len(x.split()) for x in gen_t) / max(1, len(gen_t))
    use = (res_t["accuracy"] > res_b["accuracy"] and general_same >= args.min_general
           and res_t["variety"] >= args.min_variety * res_b["variety"])
    summary = {
        "hotel_base": res_b["accuracy"], "hotel_tuned": res_t["accuracy"],
        "variety_base": res_b["variety"], "variety_tuned": res_t["variety"],
        "hotel_words_base": res_b["words"], "hotel_words_tuned": res_t["words"],
        "general_same": general_same, "general_words_base": words_b, "general_words_tuned": words_t,
        "use_tuned": use,
    }
    print(f"hotel (typed, {len(hotel)}):   base {res_b['accuracy'] * 100:.1f}%   tuned {res_t['accuracy'] * 100:.1f}%")
    print(f"variety of wording:  base {res_b['variety']:.2f}   tuned {res_t['variety']:.2f}   "
          f"(words per answer {res_b['words']:.1f} -> {res_t['words']:.1f})")
    print(f"general ({len(general)}): tuned answer as good as the base's in {general_same * 100:.1f}%   "
          f"(words {words_b:.1f} -> {words_t:.1f})")
    print(f"USE THE TUNED 4B: {'yes' if use else 'NO (base 4B is kept)'}")
    print("examples (general: base | tuned):")
    for r, t, s in list(zip(general, gen_t, same))[:6]:
        print(f"  [{'same' if s else 'DIFF'}] {r['text']}\n     base:  {r['response'].strip()[:160]}\n     tuned: {t[:160]}")
    print("examples (hotel: base | tuned):")
    for b, t in list(zip(res_b["rows"], res_t["rows"]))[:6]:
        print(f"  {b['text']}\n     base [{'ok' if b['correct'] else 'X'}]:  {b['answer'][:160]}\n"
              f"     tuned [{'ok' if t['correct'] else 'X'}]: {t['answer'][:160]}")
    save_json(args.out, {**summary, "general": [{"text": r["text"], "base": r["response"], "tuned": t, "same": s}
                                                for r, t, s in zip(general, gen_t, same)],
                         "hotel_base_rows": res_b["rows"], "hotel_tuned_rows": res_t["rows"]})


if __name__ == "__main__":
    main()
