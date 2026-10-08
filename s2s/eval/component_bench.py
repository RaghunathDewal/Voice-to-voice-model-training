"""How many simultaneous conversations can each part of the pipeline serve on this GPU?

    python -m s2s.eval.component_bench --config configs/small.yaml \
        --speech-llm-dir checkpoints/speech_llm_pk4 --talker-dir checkpoints/talker_hifi_pre --batch 1 8 32 64 200

The thinker is measured by s2s/eval/vllm_probe.py; this covers the rest, with the real models:

  listen   Parakeet encoder + adapter on finished user turns (~2.5 s of speech each): N turns one after
           another (today's server) vs N turns in ONE batch (what batching would do).
  speak    talker + Mimi decoder: seconds of audio generated per second of compute for ONE conversation.
           A conversation needs >= 1x real time while it speaks, so 1 / (time per audio second) is roughly
           how many can speak at once without batching. Batched talker decoding does not exist yet; the
           teacher-forced pass over N replies at once shows how much a batched talker could gain.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from s2s.cli_common import base_parser, config_from_args


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeat: int = 3) -> float:
    fn()  # warm-up
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeat


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--speech-llm-dir", required=True)
    p.add_argument("--talker-dir", required=True)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64, 200])
    p.add_argument("--speak-seconds", type=float, default=3.0, help="audio generated per reply in the speak test")
    args = p.parse_args()
    cfg = config_from_args(args)

    from s2s.audio import load_audio
    from s2s.eval.load_test import synth_questions
    from s2s.models.talker import TalkerStream
    from s2s.runtime.agent import VoiceAgent
    from s2s.train.talker import features_for_batch

    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    dev, sr = agent.device, agent.codec.sample_rate
    wavs = [load_audio(path, sr) for path in synth_questions(os.path.expanduser("~/.cache/s2s_load_test"))]
    print(f"\nlisten: Parakeet + adapter per finished turn (clips {np.mean([len(w) / sr for w in wavs]):.1f} s average)")

    @torch.no_grad()
    def listen(batch: list[np.ndarray]) -> None:
        feats = agent.encode_input(batch)
        lens = [f.shape[0] for f in feats]
        x = torch.zeros(len(feats), max(lens), feats[0].shape[1], device=dev)
        for i, f in enumerate(feats):
            x[i, : f.shape[0]] = f.to(dev)
        agent.adapter(x)

    for n in args.batch:
        batch = [wavs[i % len(wavs)] for i in range(n)]
        one_by_one = timed(lambda: [listen([w]) for w in batch], dev, repeat=1)
        together = timed(lambda: listen(batch), dev)
        print(f"   {n:>4} turns | one after another {1000 * one_by_one:8.0f} ms | in one batch {1000 * together:7.0f} ms"
              f" | speed-up x{one_by_one / together:5.1f}")

    print(f"\nspeak: talker + Mimi, {args.speak_seconds:.0f} s of reply audio")
    th, talker, rt = agent.thinker, agent.talker, cfg.runtime
    text = "Sure, I have ordered two extra towels for you, and they should arrive at Lodge Heron 14 in about fifteen minutes."
    prefix = th.prompts.text_prompt_ids(cfg.thinker.system_prompt, cfg.talker.talker_prompt_user)
    toks, hids = features_for_batch(th, prefix, [th.prompts.ids(text)], agent.layer_idx)
    n_frames = int(args.speak_seconds * agent.codec.frame_rate)

    @torch.no_grad()
    def speak_one() -> None:
        from s2s.models.codec import StreamingDecoder

        stream = TalkerStream(talker, float(rt.talker_temperature), int(rt.talker_top_k), max_frames=n_frames)
        stream.push_text(talker.fuse(toks[0].to(dev), hids[0].to(dev)))
        stream.end_text()
        decoder, pending, made = StreamingDecoder(agent.codec, int(rt.decode_context_frames)), [], 0
        while stream.can_step() and made < n_frames:
            new = stream.step()
            made += len(new)
            pending += new
            if len(pending) >= int(rt.emit_every_frames):
                decoder.push(pending)
                pending = []

    t = timed(speak_one, dev, repeat=2)
    rtf = t / args.speak_seconds
    print(f"   one conversation: {1000 * t:.0f} ms for {args.speak_seconds:.0f} s of audio -> {rtf:.3f} s of compute per"
          f" audio second -> about {1 / rtf:.0f} conversations can speak at the same moment without batching")

    print("\n   batched talker (teacher-forced pass over N replies at once; an upper bound for batched decoding)")
    codes = torch.randint(0, talker.card, (talker.K, n_frames), device=dev)
    from s2s.models.talker import apply_delay

    @torch.no_grad()
    def forced(n: int) -> None:
        grids = [apply_delay(codes, talker.delay, talker.card)] * n
        talker([toks[0].to(dev)] * n, [hids[0].to(dev)] * n, grids)

    for n in args.batch:
        one = timed(lambda: forced(1), dev)
        many = timed(lambda: forced(n), dev)
        print(f"   {n:>4} replies | one {1000 * one:6.0f} ms | all {n} together {1000 * many:7.0f} ms"
              f" | x{n * one / many:5.1f} more work per second than one at a time")


if __name__ == "__main__":
    main()
