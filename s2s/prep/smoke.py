"""Build tiny random models + synthetic data so every stage can be run in minutes.

This checks that the code, the installed library versions and the GPU work
together before spending hours on real data. The outputs are meaningless.

    python -m s2s.prep.smoke --root outputs/smoke
    # then run the commands it prints (or see README "Smoke test")
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from s2s.audio import save_audio
from s2s.data.hotel import generate_examples
from s2s.utils import write_jsonl

WORDS = "hello towel room please send two pillows checkout time breakfast morning water".split()


def tiny_models(root: str, tokenizer_from: str) -> tuple[str, str]:
    from transformers import AutoTokenizer, MimiConfig, MimiModel, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(tokenizer_from)
    qwen_dir = os.path.join(root, "tiny_qwen3")
    qcfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                       tie_word_embeddings=True, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    Qwen3ForCausalLM(qcfg).save_pretrained(qwen_dir)
    tok.save_pretrained(qwen_dir)

    mimi_dir = os.path.join(root, "tiny_mimi")
    mcfg = MimiConfig(hidden_size=64, num_filters=8, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, head_dim=16, intermediate_size=128, codebook_dim=32,
                      vector_quantization_hidden_dimension=32, num_quantizers=8, codebook_size=64, upsample_groups=64)
    MimiModel(mcfg).save_pretrained(mimi_dir)
    return qwen_dir, mimi_dir


def fake_speech(text: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A deterministic 'voice': one tone per word (enough to exercise the pipeline)."""
    out = []
    for w in text.lower().split():
        f = 150 + (sum(map(ord, w)) % 40) * 10
        n = int(sr * (0.18 + 0.02 * len(w)))
        t = np.arange(n) / sr
        out.append(0.2 * np.sin(2 * np.pi * f * t) * np.hanning(n))
        out.append(np.zeros(int(sr * 0.05)))
    return (np.concatenate(out) + rng.standard_normal(sum(map(len, out))) * 0.003).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/smoke")
    p.add_argument("--tokenizer-from", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n", type=int, default=48)
    args = p.parse_args()
    root = args.root
    data = os.path.join(root, "data")
    man = os.path.join(data, "manifests")
    wav_dir = os.path.join(data, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    qwen_dir, mimi_dir = tiny_models(root, args.tokenizer_from)
    rng = np.random.default_rng(0)
    sr = 24000

    def speech_rows(prefix: str, texts: list[str], extra: list[dict] | None = None) -> list[dict]:
        rows = []
        for i, text in enumerate(texts):
            path = os.path.join(wav_dir, f"{prefix}_{i:04d}.wav")
            save_audio(path, fake_speech(text, sr, rng), sr)
            row = {"id": f"{prefix}_{i:04d}", "audio": os.path.relpath(path, man), "text": text}
            if extra:
                row.update(extra[i])
            rows.append(row)
        return rows

    texts = [" ".join(rng.choice(WORDS, size=rng.integers(3, 8))) for _ in range(args.n)]
    rows = speech_rows("ls", texts)
    for r in rows[::2]:
        r["response"] = "Sure, I can help with that."
    write_jsonl(os.path.join(man, "librispeech_train_raw.jsonl"), rows[8:])
    write_jsonl(os.path.join(man, "librispeech_dev_raw.jsonl"), rows[:8])

    hotel = generate_examples(24, seed=3)
    hotel_rows = speech_rows("hotel", [h["text"] for h in hotel], [{k: v for k, v in h.items() if k not in ("id", "text")} for h in hotel])
    write_jsonl(os.path.join(man, "hotel_train_audio.jsonl"), hotel_rows[4:])
    write_jsonl(os.path.join(man, "hotel_eval_audio.jsonl"), hotel_rows[:4])
    write_jsonl(os.path.join(man, "hotel_eval_text.jsonl"), [dict(h) for h in hotel[:4]])

    talker_rows = speech_rows("tk", texts[:24])
    write_jsonl(os.path.join(man, "talker_train_raw.jsonl"), talker_rows[4:])
    write_jsonl(os.path.join(man, "talker_valid_raw.jsonl"), talker_rows[:4])

    cfg = {
        "base": os.path.relpath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default.yaml"), root),
        "paths": {"data_dir": data, "ckpt_dir": os.path.join(root, "checkpoints")},
        "codec": {"model": mimi_dir, "extract_batch_size": 8},
        "thinker": {"model": qwen_dir, "dtype": "auto", "gradient_checkpointing": False},
        "adapter": {"d_model": 64, "n_layers": 2, "n_heads": 4},
        "talker": {"d_model": 64, "n_layers": 2, "n_heads": 4, "depth_d_model": 32, "depth_layers": 2,
                   "depth_heads": 2, "max_frames": 40},
        "train_probe": {"d_model": 64, "n_layers": 1, "n_heads": 4, "batch_size": 8, "max_steps": 20, "eval_every": 10},
        "train_speech_llm": {
            "train_manifests": [{"path": os.path.join(man, "librispeech_train.jsonl"), "weight": 1.0}],
            "valid_manifest": os.path.join(man, "librispeech_dev.jsonl"),
            "output_dir": os.path.join(root, "checkpoints", "speech_llm_align"),
            "batch_size": 4, "grad_accum": 1, "max_steps": 6, "warmup_steps": 2, "eval_every": 3,
            "eval_batches": 1, "save_every": 3, "log_every": 2, "num_workers": 0},
        "train_talker": {
            "train_manifests": [{"path": os.path.join(man, "talker_train.jsonl"), "weight": 1.0}],
            "valid_manifest": os.path.join(man, "talker_valid.jsonl"),
            "output_dir": os.path.join(root, "checkpoints", "talker"),
            "thinker_dir": os.path.join(root, "checkpoints", "thinker_merged"),
            "batch_size": 4, "grad_accum": 1, "max_steps": 6, "warmup_steps": 2, "eval_every": 3,
            "eval_batches": 1, "save_every": 3, "log_every": 2, "num_workers": 0},
        "runtime": {"speech_llm_dir": os.path.join(root, "checkpoints", "speech_llm_tools"),
                    "talker_dir": os.path.join(root, "checkpoints", "talker"), "max_new_tokens": 12},
    }
    cfg_path = os.path.join(root, "smoke.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"tiny models + synthetic data ready. config: {cfg_path}")


if __name__ == "__main__":
    main()
