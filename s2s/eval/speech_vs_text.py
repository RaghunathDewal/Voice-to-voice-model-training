"""Speech vs text with the same thinker, on any recordings (no human answer labels needed).

The adapter's job is to make the thinker answer a SPOKEN request exactly as it answers the TYPED one.
So the reference is the thinker itself, run on the reference transcript. For every row:

  text reply    thinker on the transcript (+ the dialogue so far, if the row has `history`)
  speech reply  thinker on the adapter's speech embeddings (+ the same history)
  heard         thinker asked to transcribe the speech -> word error rate vs the transcript
  judge         the thinker compares the two replies: SAME (same answer / information / action) or DIFFERENT;
                a strict judge (wording-level) and a fair one (same intent, action and key details) are both reported

Baseline (--asr-json, from s2s.eval.asr_nemotron): the same thinker answers a recogniser's TRANSCRIPT as text,
per streaming setting (--asr-keys), judged the same way: what a strong speech recogniser + the same 4B gets.
Word error rates spell out numbers first (s2s.eval.spoken_wer), so "7:45" and "seven forty five" match.

    python -m s2s.eval.speech_vs_text --config configs/qwen4b.yaml --speech-llm-dir checkpoints/speech_llm_4b_v2 \
        --manifest data/manifests/slurp_test_pk.jsonl --max 200 --name slurp --out-dir outputs/s2t

Writes <out-dir>/<name>.json (summary + every row) and <name>.md (the summary and the first 25 rows to read).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.data.hotel import row_system, tools_by_name
from s2s.models.adapter import SpeechAdapter
from s2s.models.speech_llm import assemble_inputs, greedy_generate
from s2s.models.thinker import Thinker
from s2s.eval.spoken_wer import spoken_wer
from s2s.text import ctc_greedy_decode, sentence_case
from s2s.utils import autocast_ctx, resolve_device, resolve_dtype, save_json

JUDGE_SYSTEM = (
    "You compare two replies that a hotel voice assistant gave to the same guest message. Answer with one word: "
    "SAME if both replies give the same answer, the same information and take the same action (for a tool call: "
    "the same tool with the same important values); different wording is fine. Otherwise answer DIFFERENT.")
FAIR_JUDGE_SYSTEM = (
    "You compare two replies that a hotel voice assistant gave to the same guest message. Answer with one word. "
    "SAME if a guest would be equally well served by either: the same request was understood, the same action was "
    "taken (same tool, or both politely decline, or both send the guest to the same kind of staff) and the same key "
    "details (names, numbers, times, items) are used. Different wording, length, politeness or an extra offer to "
    "help is still SAME. DIFFERENT if a different request was understood, a different action or tool was chosen, or "
    "a key detail differs.")


def clean(text: str) -> str:
    return text.replace("<|im_end|>", "").strip()


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--speech-llm-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--name", required=True, help="label for this test set in the outputs")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--max-new", type=int, default=100)
    p.add_argument("--out-dir", default="outputs/speech_vs_text")
    p.add_argument("--asr-json", default=None, help="s2s.eval.asr_nemotron output for the same clips (baseline)")
    p.add_argument("--asr-keys", default="160,560", help="which streaming settings of --asr-json to answer from")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    thinker = Thinker(cfg.thinker.model, device, dtype, attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    adapter = SpeechAdapter.load(os.path.join(args.speech_llm_dir, "adapter.pt")).to(device).eval()
    rows = [r for r in load_manifest(args.manifest) if r.get("text")][: args.max or None]
    enc = adapter.hparams.get("encoder", "mimi")
    if {r.get("encoder", "mimi") for r in rows} != {enc}:
        raise ValueError(f"the adapter needs {enc} features; extract the manifest with --encoder {enc}")
    system = cfg.thinker.system_prompt
    trail = int(cfg.train_speech_llm.max_trailing_frames) // 2
    tok = thinker.tokenizer

    def generate_text(user: str, sys_prompt: str, history=None, max_new: int = args.max_new, tools=None) -> str:
        ids = thinker.prompts.text_prompt_ids(sys_prompt, user, tools, history)
        with torch.no_grad(), autocast_ctx(device, dtype):
            out = greedy_generate(thinker, thinker.embed(torch.tensor([ids], device=device)), max_new)
        return clean(tok.decode(out, skip_special_tokens=False))

    asr, asr_keys = {}, []
    if args.asr_json:
        with open(args.asr_json, encoding="utf-8") as f:
            asr = {x["id"]: x["hyp"] for x in (json.loads(line) for line in f if line.strip())}
        asr_keys = [k for k in args.asr_keys.split(",") if k]
        rows = [r for r in rows if r.get("id") in asr]  # only clips the recogniser also transcribed

    def judge(message: str, a: str, b: str, system_prompt: str) -> bool:
        if a == b:
            return True
        verdict = generate_text(f"Guest message: {message}\nReply A: {a}\nReply B: {b}", system_prompt, max_new=3)
        return verdict.strip().upper().startswith("SAME")

    results = []
    for r in tqdm(rows, desc=f"speech vs text [{args.name}]", mininterval=30):
        lat = np.load(r["latent"]).astype(np.float32)
        t = min(lat.shape[0], int(r.get("speech_frames", lat.shape[0])) + trail)
        with torch.no_grad(), autocast_ctx(device, dtype):
            out = adapter(torch.from_numpy(lat[:t])[None].to(device))
        history = r.get("history") or None
        row_sys = row_system(r, system)           # hotel rows carry their own prompt and tools
        tools = tools_by_name(r.get("tools"))

        def from_speech(prefix, suffix, max_new):
            with torch.no_grad(), autocast_ctx(device, dtype):
                emb, _, _ = assemble_inputs(thinker, adapter, out["embeds"], torch.tensor([t]), [prefix], [suffix], None)
                return clean(tok.decode(greedy_generate(thinker, emb, max_new), skip_special_tokens=False))

        pre, suf = thinker.prompts.prompt_parts(row_sys, tools, None, history)
        speech_reply = from_speech(pre, suf, args.max_new)
        text_reply = generate_text(sentence_case(r["text"]), row_sys, history, tools=tools)
        pre, suf = thinker.prompts.prompt_parts(system, None, cfg.thinker.transcribe_instruction)
        heard = from_speech(pre, suf, 100)
        msg = sentence_case(r["text"])
        exact = text_reply == speech_reply
        row = {
            "id": r.get("id"), "text": r["text"], "history": history, "heard": heard,
            "ctc": ctc_greedy_decode(out["ctc_logits"][0].argmax(-1).tolist()),
            "text_reply": text_reply, "speech_reply": speech_reply, "exact": exact,
            "same": judge(msg, text_reply, speech_reply, JUDGE_SYSTEM),
            "same_fair": judge(msg, text_reply, speech_reply, FAIR_JUDGE_SYSTEM),
        }
        for k in asr_keys:  # baseline: the recogniser's transcript, answered as text by the same 4B
            hyp = asr[r["id"]][k]
            reply = generate_text(sentence_case(hyp), row_sys, history, tools=tools)
            row[f"asr_{k}"] = {"heard": hyp, "reply": reply, "exact": reply == text_reply,
                               "same": judge(msg, text_reply, reply, JUDGE_SYSTEM),
                               "same_fair": judge(msg, text_reply, reply, FAIR_JUDGE_SYSTEM)}
        results.append(row)

    n = len(results)
    refs = [x["text"] for x in results]
    pct = lambda xs: sum(xs) / max(1, n)  # noqa: E731
    summary = {
        "name": args.name, "manifest": args.manifest, "adapter": args.speech_llm_dir, "n": n,
        "same_answer": pct(x["same"] for x in results),
        "same_answer_fair": pct(x["same_fair"] for x in results),
        "identical_answer": pct(x["exact"] for x in results),
        "heard_wer": spoken_wer(refs, [x["heard"] for x in results]),
        "ctc_wer": spoken_wer(refs, [x["ctc"] for x in results]),
        "multi_turn_rows": sum(bool(x["history"]) for x in results),
    }
    if asr:
        for k, hyps in next(iter(asr.values())).items():  # recogniser WER for every setting it was run with
            summary[f"asr_{k}_wer"] = spoken_wer(refs, [asr[x["id"]][k] for x in results])
        for k in asr_keys:
            summary[f"asr_{k}_same"] = pct(x[f"asr_{k}"]["same"] for x in results)
            summary[f"asr_{k}_same_fair"] = pct(x[f"asr_{k}"]["same_fair"] for x in results)
            summary[f"asr_{k}_identical"] = pct(x[f"asr_{k}"]["exact"] for x in results)
    line = (f"[{args.name}] n={n}  ADAPTER same answer: strict {summary['same_answer'] * 100:.1f}% / fair "
            f"{summary['same_answer_fair'] * 100:.1f}%  identical {summary['identical_answer'] * 100:.1f}%  "
            f"heard WER {summary['heard_wer'] * 100:.1f}%  CTC WER {summary['ctc_wer'] * 100:.1f}%  "
            f"multi-turn rows {summary['multi_turn_rows']}")
    for k in asr_keys:
        line += (f"\n[{args.name}] RECOGNISER {k}ms + 4B: strict {summary[f'asr_{k}_same'] * 100:.1f}% / fair "
                 f"{summary[f'asr_{k}_same_fair'] * 100:.1f}%  identical {summary[f'asr_{k}_identical'] * 100:.1f}%  "
                 f"WER {summary[f'asr_{k}_wer'] * 100:.1f}%")
    print(line, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(os.path.join(args.out_dir, f"{args.name}.json"), {"summary": summary, "rows": results})
    verdict = lambda strict, fair: f"{'SAME' if strict else 'DIFF'}/{'SAME' if fair else 'DIFF'}"  # noqa: E731
    with open(os.path.join(args.out_dir, f"{args.name}.md"), "w", encoding="utf-8") as f:
        f.write(f"# {args.name}\n\n{line}\n\n(verdicts: strict/fair judge)\n\n")
        for x in results[:25]:
            f.write(f"- said: **{x['text']}**\n")
            if x["history"]:
                f.write(f"  - earlier turns: {x['history']}\n")
            f.write(f"  - typed reply: {x['text_reply']}\n")
            f.write(f"  - adapter [{verdict(x['same'], x['same_fair'])}] heard: {x['heard']} | reply: {x['speech_reply']}\n")
            for k in asr_keys:
                a = x[f"asr_{k}"]
                f.write(f"  - recogniser {k}ms [{verdict(a['same'], a['same_fair'])}] heard: {a['heard']} | "
                        f"reply: {a['reply']}\n")


if __name__ == "__main__":
    main()
