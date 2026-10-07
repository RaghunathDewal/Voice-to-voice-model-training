"""Experiment 3: does speech input work as well as text input?

For rows with `tool_calls` (spoken hotel requests): tool-call accuracy from
SPEECH vs from the TEXT of the same request (same thinker).
For other rows: thinker transcription WER, CTC WER, and sample replies.

    python -m s2s.eval.speech_llm --speech-llm-dir checkpoints/speech_llm_align \
        --manifest data/manifests/librispeech_test.jsonl --max 300
    python -m s2s.eval.speech_llm --speech-llm-dir checkpoints/speech_llm_tools \
        --manifest data/manifests/hotel_eval.jsonl
"""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.data.hotel import row_system, score_example, tools_by_name
from s2s.eval.text_tools import extract_calls
from s2s.models.adapter import SpeechAdapter
from s2s.models.speech_llm import assemble_inputs, greedy_generate
from s2s.models.thinker import Thinker
from s2s.text import ctc_greedy_decode, wer
from s2s.utils import autocast_ctx, resolve_device, resolve_dtype, save_json


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--speech-llm-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--max", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    lora = os.path.join(args.speech_llm_dir, "lora")
    thinker = Thinker(cfg.thinker.model, device, dtype, lora_dir=lora if os.path.isdir(lora) else None,
                      merge_lora=True, attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    adapter = SpeechAdapter.load(os.path.join(args.speech_llm_dir, "adapter.pt")).to(device).eval()
    rows = load_manifest(args.manifest)[: args.max or None]
    enc_spec = adapter.hparams.get("encoder", "mimi")
    found = {r.get("encoder", "mimi") for r in rows}
    if found != {enc_spec}:
        raise ValueError(f"adapter was trained on {enc_spec} features but {args.manifest} has {sorted(found)}")
    trail = int(cfg.train_speech_llm.max_trailing_frames) // 2

    tool_speech, tool_text, n_tool = 0, 0, 0
    refs, ctc_hyps, llm_hyps, replies, misses = [], [], [], [], []
    for r in tqdm(rows, desc="speech eval"):
        lat = np.load(r["latent"]).astype(np.float32)
        t = min(lat.shape[0], int(r.get("speech_frames", lat.shape[0])) + trail)
        lat_t = torch.from_numpy(lat[:t])[None].to(device)
        with torch.no_grad(), autocast_ctx(device, dtype):
            out = adapter(lat_t)
        ctc = ctc_greedy_decode(out["ctc_logits"][0].argmax(-1).tolist())
        system = row_system(r, cfg.thinker.system_prompt)
        tools = tools_by_name(r.get("tools"))
        lengths = torch.tensor([t])

        def run(prefix, suffix, max_new=100):
            with torch.no_grad(), autocast_ctx(device, dtype):
                emb, _, _ = assemble_inputs(thinker, adapter, out["embeds"], lengths, [prefix], [suffix], None)
                return thinker.tokenizer.decode(greedy_generate(thinker, emb, max_new), skip_special_tokens=False)

        if r.get("tools") or "tool_calls" in r:  # hotel rows (v3 rows may offer no tools)
            n_tool += 1
            history = r.get("history") or None
            pre, suf = thinker.prompts.prompt_parts(system, tools, None, history)
            speech_out = run(pre, suf, 120)
            ok_s = score_example(r, extract_calls(speech_out), speech_out)
            ids = thinker.prompts.text_prompt_ids(system, r["text"], tools, history)
            text_out = thinker.tokenizer.decode(
                greedy_generate(thinker, thinker.embed(torch.tensor([ids], device=device)), 120), skip_special_tokens=False)
            ok_t = score_example(r, extract_calls(text_out), text_out)
            tool_speech += int(ok_s)
            tool_text += int(ok_t)
            if not ok_s:
                misses.append({"text": r["text"], "ctc": ctc, "gold": r.get("tool_calls") or r.get("reply"), "speech_out": speech_out,
                               "text_ok": ok_t})
        else:
            pre, suf = thinker.prompts.prompt_parts(system, None, cfg.thinker.transcribe_instruction)
            hyp = run(pre, suf, 100).replace("<|im_end|>", "").strip()
            refs.append(r["text"])
            ctc_hyps.append(ctc)
            llm_hyps.append(hyp)
            if len(replies) < 5:
                pre, suf = thinker.prompts.prompt_parts(system, tools, None)
                replies.append({"heard": r["text"], "reply": run(pre, suf, 80)})

    result = {}
    if n_tool:
        result["tool_acc_speech"] = tool_speech / n_tool
        result["tool_acc_text"] = tool_text / n_tool
        print(f"tool-call accuracy  speech: {tool_speech / n_tool * 100:.1f}%   text: {tool_text / n_tool * 100:.1f}%   (n={n_tool})")
        for m in misses[:5]:
            print(f"  MISS: {m['text']} | ctc: {m['ctc']}\n    gold {m['gold']}\n    got  {m['speech_out']!r}")
    if refs:
        result["thinker_transcribe_wer"] = wer(refs, llm_hyps)
        result["ctc_wer"] = wer(refs, ctc_hyps)
        print(f"thinker transcription WER {result['thinker_transcribe_wer'] * 100:.2f}%   CTC WER {result['ctc_wer'] * 100:.2f}%   (n={len(refs)})")
        for rep in replies:
            print(f"  HEARD: {rep['heard']}\n  REPLY: {rep['reply']!r}")
    if args.out:
        save_json(args.out, {**result, "misses": misses, "replies": replies})


if __name__ == "__main__":
    main()
