"""Run NVIDIA PersonaPlex-7B (full duplex, Moshi-based) on recorded calls, as it would run live: the guest audio
is fed in 80 ms frames and the agent's audio + text come out frame by frame. The model is loaded once for all
calls, and every frame is timed (live use needs each frame computed in under 80 ms).

Runs in its own environment with NVIDIA's code (github.com/NVIDIA/personaplex, `pip install moshi/.`):

    python personaplex_run.py --calls calls.json --out-dir out [--voice NATF2.pt] [--cpu-offload --gpu-gib 11.5] [--dtype bf16]

Per call: <id>.wav (agent audio, same length as the input), <id>.json (text token per frame, frame times).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download
from moshi.models import LMGen, loaders
from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
from moshi.offline import _get_voice_prompt_dir, decode_tokens_to_pcm, warmup, wrap_with_system_tags

SPECIAL = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_lm(path: str, device: str, dtype: torch.dtype, offload: bool, gpu_gib: float):
    """The 7B on one GPU, or (offload) as much as fits on it with the rest in CPU memory, streamed to the GPU each
    step. Loaded on the CPU first without a second copy (NVIDIA's own offload loader holds the weights twice,
    ~34 GB, more than a Kaggle machine's RAM)."""
    if not offload:
        return loaders.get_moshi_lm(path, device=device, dtype=dtype).eval()
    from accelerate import dispatch_model, infer_auto_device_map

    lm = loaders.get_moshi_lm(path, device="cpu", dtype=dtype).eval()
    dmap = infer_auto_device_map(lm, max_memory={0: f"{gpu_gib}GiB", "cpu": "200GiB"}, dtype=dtype,
                                 no_split_module_classes=["StreamingTransformerLayer"])
    on_cpu = sum(v == "cpu" for v in dmap.values())
    print(f"[personaplex] {len(dmap) - on_cpu} parts on the GPU, {on_cpu} in CPU memory", flush=True)
    return dispatch_model(lm, device_map=dmap, main_device=device if ":" in device else f"{device}:0").eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--voice", default="NATF2.pt", help="NATF0-3 / NATM0-3 (natural), VARF0-4 / VARM0-4")
    p.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--cpu-offload", action="store_true", help="spread the 7B over the GPUs (+ CPU) when one is too small")
    p.add_argument("--gpu-gib", type=float, default=11.5, help="with --cpu-offload: model weights kept on the GPU")
    p.add_argument("--seed", type=int, default=42424242)
    args = p.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    mimi_w = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi, other_mimi = loaders.get_mimi(mimi_w, args.device), loaders.get_mimi(mimi_w, args.device)
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME))
    lm = load_lm(hf_hub_download(args.hf_repo, loaders.MOSHI_NAME), args.device, dtype, args.cpu_offload,
                 args.gpu_gib)
    lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * mimi.frame_rate), sample_rate=mimi.sample_rate,
                   device=args.device, frame_rate=mimi.frame_rate, save_voice_prompt_embeddings=False,
                   use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    warmup(mimi, other_mimi, lm_gen, args.device, frame)
    load_s = time.time() - t0
    print(f"[personaplex] loaded in {load_s:.0f}s ({args.dtype}{', offloaded' if args.cpu_offload else ''}); "
          f"GPU memory: " + ", ".join(f"cuda:{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GB"
                                      for i in range(torch.cuda.device_count())), flush=True)
    voice_path = os.path.join(_get_voice_prompt_dir(None, args.hf_repo), args.voice)

    os.makedirs(args.out_dir, exist_ok=True)
    calls = json.load(open(args.calls))
    for call in calls:
        lm_gen.load_voice_prompt_embeddings(voice_path)
        lm_gen.text_prompt_tokens = tok.encode(wrap_with_system_tags(call["prompt"])) if call.get("prompt") else None
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        sync()
        t = time.time()
        lm_gen.step_system_prompts(mimi)  # voice prompt + text prompt, before the call starts
        sync()
        prompt_s = time.time() - t
        mimi.reset_streaming()

        audio = load_audio(call["wav"], mimi.sample_rate)
        pcm, text, frame_ms = [], [], []
        frames = encode_from_sphn(mimi, _iterate_audio(audio, sample_interval_size=lm_gen._frame_size, pad=True),
                                  max_batch=1)
        while True:  # the time of a frame = encoding the guest's 80 ms + the model step + decoding the agent's 80 ms
            sync()
            t = time.time()
            enc = next(frames, None)
            if enc is None:
                break
            for c in range(enc.shape[-1]):
                tokens = lm_gen.step(enc[:, :, c: c + 1])
                if tokens is None:
                    continue
                pcm.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))
                tid = int(tokens[0, 0, 0].item())
                text.append(SPECIAL[tid] if tid in SPECIAL else tok.id_to_piece(tid).replace("▁", " "))
            sync()
            frame_ms.append((time.time() - t) * 1000)
        out = np.concatenate(pcm)[: audio.shape[-1]] if pcm else np.zeros(audio.shape[-1], np.float32)
        out = np.pad(out, (0, max(0, audio.shape[-1] - len(out))))
        sphn.write_wav(os.path.join(args.out_dir, f"{call['id']}.wav"), out.astype(np.float32), mimi.sample_rate)
        ms = np.array(frame_ms[3:] or frame_ms)
        stats = {"frame_ms_p50": float(np.median(ms)), "frame_ms_p95": float(np.percentile(ms, 95)),
                 "frame_budget_ms": 1000 / mimi.frame_rate, "prompt_s": prompt_s, "load_s": load_s,
                 "seconds": audio.shape[-1] / mimi.sample_rate, "dtype": args.dtype, "offload": args.cpu_offload,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "voice": args.voice}
        with open(os.path.join(args.out_dir, f"{call['id']}.json"), "w") as f:
            json.dump({"id": call["id"], "frame_s": 1 / mimi.frame_rate, "text": text, "stats": stats}, f)
        said = "".join(x for x in text if x not in SPECIAL.values()).strip()
        print(f"[personaplex] {call['id']}: {stats['seconds']:.0f}s of call, frame {stats['frame_ms_p50']:.0f} ms "
              f"(p95 {stats['frame_ms_p95']:.0f}) vs {stats['frame_budget_ms']:.0f} ms budget | prompt "
              f"{prompt_s:.1f}s | said: {said[:160]!r}", flush=True)


if __name__ == "__main__":
    main()
