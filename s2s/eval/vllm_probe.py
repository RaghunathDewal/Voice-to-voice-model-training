"""Can vLLM batch our thinker with SPEECH EMBEDDINGS as input (no text in between)? A two-step probe.

Step 1, in the training/runtime environment (our code, Parakeet, the adapter):

    python s2s/eval/vllm_probe.py prepare --config configs/small.yaml \
        --speech-llm-dir checkpoints/speech_llm_pk4 --talker-dir checkpoints/talker_hifi_pre --out /root/vllm_probe.pt

  builds, for a few spoken guest questions, exactly the input our runtime gives the thinker
  (system prompt + tools as embeddings, the adapter's speech embeddings, the turn boundaries) and the
  reply our current runtime generates from it (greedy), and saves both.

Step 2, in an environment with vLLM; needs only torch + vllm. On AMD (MI300X), AMD's vLLM image needs
`--security-opt seccomp=unconfined --ulimit memlock=-1:-1` or the engine hangs silently at start-up:

    docker run -d --name vllm_probe --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn -v /root/probe:/probe rocm/vllm:latest \
      bash -c "python /probe/vllm_probe.py run --data /probe/vllm_probe.pt --model /probe/thinker > /probe/result.txt 2>&1"

  First result (MI300X, thinker_merged_v3, eager mode): 6/6 replies identical to our runtime, the system
  prompt's KV reused across requests, first token 33 ms alone and 152 ms with 64 requests at once.

  1. correctness: does vLLM, given the same embeddings, produce the same replies?
  2. prefix caching: is the system prompt's KV reused across requests when the prompt is embeddings?
     (vLLM reports cached prompt tokens per request; 0 means the prompt is recomputed every turn)
  3. batching: time to first token and total time with 1 .. 64 simultaneous requests.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import torch

QUESTIONS = [
    ("af_bella", "Hi, could you send two extra towels to my room?"),
    ("am_michael", "What time is check out tomorrow?"),
    ("bf_emma", "The air conditioning in my room is not working."),
    ("am_adam", "What is the Wi-Fi password, please?"),
    ("af_sarah", "Can you tell me my reservation details?"),
    ("bm_george", "Is breakfast included, and what time does it start?"),
]
SYSTEM = ("You are the voice concierge of Lakeview Holiday Park. Be warm and brief; replies are spoken aloud.\n"
          "Guest: Priya Sharma, Lodge Heron 14, 7 to 11 October, 4 guests. Wi-Fi: LakeviewGuest / otter2026. "
          "Check-out 11:00. Breakfast 7:00-10:30 in the main lodge, included.")


# ----------------------------------------------------------------------------- step 1
def prepare(argv: list[str]) -> None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from s2s.cli_common import base_parser, config_from_args
    from s2s.data.hotel_v3 import select_tools
    from s2s.eval.load_test import synth_questions
    from s2s.audio import load_audio
    from s2s.runtime.agent import VoiceAgent

    p = base_parser(__doc__)
    p.add_argument("--speech-llm-dir", required=True)
    p.add_argument("--talker-dir", required=True)
    p.add_argument("--tools", default="order_product,create_issue")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=60)
    args = p.parse_args(argv)
    cfg = config_from_args(args)
    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    th, ad = agent.thinker, agent.adapter
    tools = select_tools(args.tools, None)
    prefix_ids, close_ids = th.prompts.prompt_parts(th.system_content(SYSTEM), tools, None)
    start, end = ad.boundary_embeddings()
    wavs = [load_audio(path, agent.codec.sample_rate) for path in synth_questions(os.path.expanduser("~/.cache/s2s_load_test"))]
    items = []
    with torch.no_grad():
        for (voice, text), wav in zip(QUESTIONS, wavs):
            speech = ad(agent.encode_input([wav])[0][None].to(agent.device))["embeds"][0]
            emb = torch.cat([th.embed(torch.tensor(prefix_ids, device=agent.device)), start[None].to(th.dtype),
                             speech.to(th.dtype), end[None].to(th.dtype),
                             th.embed(torch.tensor(close_ids, device=agent.device))], dim=0)
            out = th.model.generate(inputs_embeds=emb[None], attention_mask=torch.ones(1, emb.shape[0], device=agent.device,
                                    dtype=torch.long), max_new_tokens=args.max_new_tokens, do_sample=False,
                                    eos_token_id=sorted(th.eos_ids), pad_token_id=th.pad_id)
            reply = th.tokenizer.decode(out[0], skip_special_tokens=False).replace("<|im_end|>", "").strip()
            print(f"{voice:>10} | {text}\n           -> {reply}")
            items.append({"question": text, "embeds": emb.float().cpu(), "reply": reply})
    torch.save({"items": items, "system_tokens": len(prefix_ids), "hidden": int(items[0]["embeds"].shape[1]),
                "max_new_tokens": args.max_new_tokens}, args.out)
    print(f"saved {len(items)} prompts ({items[0]['embeds'].shape[0]} positions, system part {len(prefix_ids)}) -> {args.out}")


# ----------------------------------------------------------------------------- step 2
def make_engine(model: str, dtype: str, mem: float, eager: bool = False):
    from vllm import AsyncEngineArgs

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:  # older vLLM
        from vllm import AsyncLLMEngine as Engine
    args = AsyncEngineArgs(model=model, dtype=dtype, enable_prompt_embeds=True, enable_prefix_caching=True,
                           gpu_memory_utilization=mem, max_model_len=4096, enforce_eager=eager)
    return Engine.from_engine_args(args)


async def one(engine, emb: torch.Tensor, params, rid: str) -> dict:
    t0 = time.perf_counter()
    first, text, cached = None, "", None
    async for out in engine.generate({"prompt_embeds": emb}, params, rid):
        if first is None and out.outputs and out.outputs[0].token_ids:
            first = time.perf_counter() - t0
        text = out.outputs[0].text if out.outputs else text
        cached = getattr(out, "num_cached_tokens", cached)
    return {"ttft": first, "total": time.perf_counter() - t0, "text": text.strip(), "cached": cached}


def pct(xs, q):
    import numpy as np

    xs = [x for x in xs if x is not None]
    return f"{1000 * float(np.percentile(xs, q)):6.0f} ms" if xs else "     -   "


async def run_async(args) -> None:
    from vllm import SamplingParams

    data = torch.load(args.data)
    items = data["items"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    embs = [it["embeds"].to(dtype) for it in items]
    engine = make_engine(args.model, args.dtype, args.gpu_memory_utilization, args.enforce_eager)
    params = SamplingParams(temperature=0.0, max_tokens=data["max_new_tokens"])
    n = 0

    def rid():
        nonlocal n
        n += 1
        return f"r{n}"

    print("\n1. correctness (greedy, same embeddings as our runtime)")
    same = 0
    for it, emb in zip(items, embs):
        r = await one(engine, emb, params, rid())
        ok = r["text"] == it["reply"]
        same += ok
        print(f"   {'SAME' if ok else 'DIFF'} | ours: {it['reply']}\n          vllm: {r['text']}")
    print(f"   {same}/{len(items)} identical (small differences can come from bf16 vs fp16 kernels)")

    print(f"\n2. prefix caching with embedding prompts (system part = {data['system_tokens']} positions)")
    for i in range(3):
        r = await one(engine, embs[i % len(embs)], params, rid())
        print(f"   request {i + 1}: cached prompt tokens = {r['cached']}  (ttft {pct([r['ttft']], 50).strip()})")
    print("   > 0 on later requests: the system prompt is reused. 0 / None: recomputed every turn.")

    print("\n3. batching: N simultaneous requests (each a different spoken question)")
    for conc in args.concurrency:
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(one(engine, embs[i % len(embs)], params, rid()) for i in range(conc)))
        wall = time.perf_counter() - t0
        toks = sum(len(r["text"].split()) for r in rs)
        print(f"   {conc:>3} at once | first token p50 {pct([r['ttft'] for r in rs], 50)} p95 {pct([r['ttft'] for r in rs], 95)}"
              f" | whole reply p50 {pct([r['total'] for r in rs], 50)} | all done in {wall:5.2f} s (~{toks / wall:5.0f} words/s)")


def run(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description="vLLM probe, step 2")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True, help="the merged thinker (HF folder), e.g. checkpoints/thinker_merged_v3")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--enforce-eager", action="store_true",
                   help="skip graph compilation / CUDA graphs: starts in seconds, but runs slower than production")
    asyncio.run(run_async(p.parse_args(argv)))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "run"):
        raise SystemExit(__doc__)
    (prepare if sys.argv[1] == "prepare" else run)(sys.argv[2:])
