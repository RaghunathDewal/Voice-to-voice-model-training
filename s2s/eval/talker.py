"""Experiment 4: is the talker's speech intelligible?

Generates speech for held-out texts, transcribes it with Whisper and reports
WER. The same is done for the reference audio passed through Mimi
(encode -> decode) which gives the best WER achievable with this codec + ASR.

    python -m s2s.eval.talker --talker-dir checkpoints/talker --manifest data/manifests/talker_valid.jsonl --max 50
"""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from s2s.audio import resample, save_audio
from s2s.cli_common import base_parser, config_from_args
from s2s.data.datasets import load_manifest
from s2s.models.codec import MimiCodec
from s2s.models.talker import Talker, TalkerStream
from s2s.models.thinker import Thinker
from s2s.text import wer
from s2s.train.talker import features_for_batch
from s2s.utils import load_json, resolve_device, resolve_dtype, save_json


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--talker-dir", default="checkpoints/talker")
    p.add_argument("--manifest", default="data/manifests/talker_valid.jsonl")
    p.add_argument("--max", type=int, default=50)
    p.add_argument("--asr", default="openai/whisper-small")
    p.add_argument("--out-dir", default="outputs/talker_eval")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.thinker.dtype, device)
    meta = load_json(os.path.join(args.talker_dir, "meta.json"))
    thinker = Thinker(meta["thinker"], device, dtype, attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    talker = Talker.load(os.path.join(args.talker_dir, "talker.pt")).to(device).eval()
    codec = MimiCodec(cfg.codec.model, device, talker.K)
    prefix = thinker.prompts.text_prompt_ids(cfg.thinker.system_prompt, cfg.talker.talker_prompt_user)

    # Whisper is called directly: the ASR pipeline breaks on some transformers versions (KeyError 'num_frames')
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    asr_proc = WhisperProcessor.from_pretrained(args.asr)
    asr_model = WhisperForConditionalGeneration.from_pretrained(args.asr).to(device).eval()

    @torch.no_grad()
    def transcribe(wav24: np.ndarray) -> str:
        wav16 = resample(wav24, codec.sample_rate, 16000)[: 30 * 16000]
        feats = asr_proc.feature_extractor(wav16, sampling_rate=16000, return_tensors="pt").input_features
        ids = asr_model.generate(feats.to(device), language="en", task="transcribe", max_new_tokens=200)
        return asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    rows = [r for r in load_manifest(args.manifest) if r.get("text")][: args.max]
    os.makedirs(args.out_dir, exist_ok=True)
    refs, hyps, ref_hyps, durations, results = [], [], [], [], []
    rt = cfg.runtime
    for i, r in enumerate(tqdm(rows, desc="talker eval")):
        toks, hids = features_for_batch(thinker, prefix, [thinker.prompts.ids(r["text"])], meta["layer_idx"])
        with torch.no_grad():
            stream = TalkerStream(talker, rt.talker_temperature, rt.talker_top_k)
            stream.push_text(talker.fuse(toks[0], hids[0]))
            stream.end_text()
            frames = []
            while stream.can_step():
                frames += stream.step()
        wav = codec.decode(torch.stack(frames, 1)) if frames else np.zeros(1920, dtype=np.float32)
        save_audio(os.path.join(args.out_dir, f"{i:03d}.wav"), wav, codec.sample_rate)
        hyp = transcribe(wav)
        refs.append(r["text"])
        hyps.append(hyp)
        durations.append(len(wav) / codec.sample_rate)
        if r.get("codes"):
            ref_codes = torch.from_numpy(np.load(r["codes"]).astype(np.int64)[: talker.K])
            ref_hyps.append(transcribe(codec.decode(ref_codes)))
        results.append({"text": r["text"], "asr": hyp, "seconds": durations[-1],
                        "hit_max_frames": len(frames) >= talker.max_frames})
    res = {"talker_wer": wer(refs, hyps), "mean_seconds": float(np.mean(durations)),
           "hit_max_frames": sum(x["hit_max_frames"] for x in results)}
    if ref_hyps and len(ref_hyps) == len(refs):
        res["mimi_reference_wer"] = wer(refs, ref_hyps)
    print(f"talker WER {res['talker_wer'] * 100:.2f}%" +
          (f"   (reference audio through Mimi: {res['mimi_reference_wer'] * 100:.2f}%)" if "mimi_reference_wer" in res else ""))
    print(f"mean duration {res['mean_seconds']:.2f}s; runaway generations (hit max frames): {res['hit_max_frames']}")
    for x in results[:5]:
        print(f"  TEXT: {x['text']}\n  ASR:  {x['asr']}")
    save_json(os.path.join(args.out_dir, "result.json"), {**res, "results": results})
    print(f"wavs + result.json in {args.out_dir}")


if __name__ == "__main__":
    main()
