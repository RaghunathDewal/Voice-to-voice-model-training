"""Speech vs text with the same thinker, on any recordings (no human answer labels needed).

The adapter's job is to make the thinker answer a SPOKEN request exactly as it answers the TYPED one.
So the reference is the thinker itself, run on the reference transcript. For every row:

  text reply    thinker on the transcript (+ the dialogue so far, if the row has `history`)
  speech reply  thinker on the adapter's speech embeddings (+ the same history)
  heard         thinker asked to transcribe the speech -> word error rate vs the transcript
  judge         the thinker compares the two replies: SAME (same answer / information / action) or DIFFERENT

    python -m s2s.eval.speech_vs_text --config configs/qwen4b.yaml --speech-llm-dir checkpoints/speech_llm_4b_v2 \
        --manifest data/manifests/slurp_test_pk.jsonl --max 200 --name slurp --out-dir outputs/s2t

Writes <out-dir>/<name>.json (summary + every row) and <name>.md (the summary and the first 25 rows to read).
"""

from __future__ import annotations

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
from s2s.text import ctc_greedy_decode, sentence_case, wer
from s2s.utils import autocast_ctx, resolve_device, resolve_dtype, save_json

JUDGE_SYSTEM = (
    "You compare two replies that a hotel voice assistant gave to the same guest message. Answer with one word: "
    "SAME if both replies give the same answer, the same information and take the same action (for a tool call: "
    "the same tool with the same important values); different wording is fine. Otherwise answer DIFFERENT.")


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
        exact = text_reply == speech_reply
        verdict = "SAME" if exact else generate_text(
            f"Guest message: {sentence_case(r['text'])}\nReply A: {text_reply}\nReply B: {speech_reply}",
            JUDGE_SYSTEM, max_new=3)
        results.append({
            "id": r.get("id"), "text": r["text"], "history": history, "heard": heard,
            "ctc": ctc_greedy_decode(out["ctc_logits"][0].argmax(-1).tolist()),
            "text_reply": text_reply, "speech_reply": speech_reply,
            "same": verdict.strip().upper().startswith("SAME"), "exact": exact,
        })

    n = len(results)
    refs = [x["text"] for x in results]
    summary = {
        "name": args.name, "manifest": args.manifest, "adapter": args.speech_llm_dir, "n": n,
        "same_answer": sum(x["same"] for x in results) / max(1, n),
        "identical_answer": sum(x["exact"] for x in results) / max(1, n),
        "heard_wer": wer(refs, [x["heard"] for x in results]),
        "ctc_wer": wer(refs, [x["ctc"] for x in results]),
        "multi_turn_rows": sum(bool(x["history"]) for x in results),
    }
    line = (f"[{args.name}] n={n}  same answer as typed: {summary['same_answer'] * 100:.1f}%  "
            f"identical: {summary['identical_answer'] * 100:.1f}%  heard WER {summary['heard_wer'] * 100:.1f}%  "
            f"CTC WER {summary['ctc_wer'] * 100:.1f}%  multi-turn rows {summary['multi_turn_rows']}")
    print(line, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(os.path.join(args.out_dir, f"{args.name}.json"), {"summary": summary, "rows": results})
    with open(os.path.join(args.out_dir, f"{args.name}.md"), "w", encoding="utf-8") as f:
        f.write(f"# {args.name}\n\n{line}\n\n")
        for x in results[:25]:
            f.write(f"- **{'SAME' if x['same'] else 'DIFFERENT'}** | said: {x['text']}\n")
            if x["history"]:
                f.write(f"  - earlier turns: {x['history']}\n")
            f.write(f"  - heard: {x['heard']}\n  - typed reply: {x['text_reply']}\n  - spoken reply: {x['speech_reply']}\n")


if __name__ == "__main__":
    main()
