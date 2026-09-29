"""Datasets and collators.

Manifest rows are JSON objects. Paths inside a manifest are resolved relative
to the manifest file's directory (so a data folder can be moved as a whole).

Speech-LLM rows (stage 2/3):
    latent         path to .npy [T, 512] float16 Mimi latents (incl. trailing silence)
    speech_frames  number of frames before the appended trailing silence
    text           transcript of the audio
    response       (optional) distilled thinker reply to the text -> "respond" task
    tool_calls     (optional) gold tool calls  -> tool task
    reply          (optional) gold spoken reply -> tool/respond task
    context        (optional) extra system context (e.g. reservation JSON)
    tools          (optional) tool set name, e.g. "hotel"

Talker rows (stage 4):
    text   text that was spoken
    codes  path to .npy [K, F] int16 Mimi codes
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from s2s.data.hotel import tools_by_name
from s2s.models.thinker import PromptBuilder
from s2s.text import ctc_encode, sentence_case
from s2s.utils import read_jsonl


def resolve(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def load_manifest(path: str) -> list[dict]:
    rows = read_jsonl(path)
    base = os.path.dirname(os.path.abspath(path))
    for r in rows:
        for key in ("latent", "codes", "audio"):
            if key in r and r[key]:
                r[key] = resolve(r[key], base)
    return rows


def load_mixture(entries: list[dict]) -> tuple[list[dict], list[float]]:
    """entries: [{path, weight}] -> rows and per-row sampling weights (each manifest gets `weight` share)."""
    rows, weights = [], []
    for e in entries:
        part = load_manifest(e["path"])
        if not part:
            continue
        rows.extend(part)
        weights.extend([float(e.get("weight", 1.0)) / len(part)] * len(part))
    if not rows:
        raise ValueError(f"no rows found in manifests {[e['path'] for e in entries]}")
    return rows, weights


def mixture_sampler(weights: list[float], num_samples: int, seed: int) -> WeightedRandomSampler:
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=g)


# ---------------------------------------------------------------- speech LLM
class SpeechLLMDataset(Dataset):
    def __init__(self, rows: list[dict], prompts: PromptBuilder, system_prompt: str, transcribe_instruction: str,
                 transcribe_prob: float, max_trailing_frames: int, max_frames: int, max_target_tokens: int,
                 train: bool = True, eot_margin: int = 4, seed: int = 0):
        self.rows = rows
        self.prompts = prompts
        self.system_prompt = system_prompt
        self.transcribe_instruction = transcribe_instruction
        self.transcribe_prob = transcribe_prob
        self.max_trailing = max_trailing_frames
        self.max_frames = max_frames
        self.max_target_tokens = max_target_tokens
        self.train = train
        self.eot_margin = eot_margin
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def task_for(self, row: dict, rng: random.Random) -> str:
        if row.get("tool_calls") or row.get("reply"):
            return "tool"
        if row.get("response") and rng.random() >= self.transcribe_prob:
            return "respond"
        return "transcribe"

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        rng = random.Random() if self.train else random.Random(self.seed * 1_000_003 + idx)
        lat = np.load(row["latent"]).astype(np.float32)
        speech_frames = int(row.get("speech_frames", lat.shape[0]))
        extra = rng.randint(0, self.max_trailing) if self.train else self.max_trailing // 2
        t = min(lat.shape[0], speech_frames + extra, self.max_frames)
        lat = lat[:t]

        task = self.task_for(row, rng)
        system = PromptBuilder.system_content(self.system_prompt, row.get("context"))
        tools = tools_by_name(row.get("tools"))
        if task == "tool":
            prefix, suffix = self.prompts.prompt_parts(system, tools, None)
            target = self.prompts.target_ids(system, tools, content=row.get("reply", ""),
                                             tool_calls=row.get("tool_calls") or None)
        elif task == "respond":
            prefix, suffix = self.prompts.prompt_parts(system, tools, None)
            target = self.prompts.target_ids(system, tools, content=row["response"])
        else:
            prefix, suffix = self.prompts.prompt_parts(system, None, self.transcribe_instruction)
            target = self.prompts.target_ids(system, None, content=sentence_case(row["text"]))
        target = target[: self.max_target_tokens]

        eot = np.full(t, -1, dtype=np.int64)
        eot[: max(0, speech_frames - self.eot_margin)] = 0
        eot[speech_frames:] = 1
        return {
            "id": row.get("id", str(idx)), "task": task, "latent": torch.from_numpy(lat),
            "prefix": prefix, "suffix": suffix, "target": target,
            "ctc": ctc_encode(row.get("text", "")), "eot": torch.from_numpy(eot), "text": row.get("text", ""),
        }


def collate_speech_llm(items: list[dict]) -> dict:
    lengths = torch.tensor([it["latent"].shape[0] for it in items])
    dim = items[0]["latent"].shape[1]
    latents = torch.zeros(len(items), int(lengths.max()), dim)
    eot = torch.full((len(items), int(lengths.max())), -1, dtype=torch.long)
    for i, it in enumerate(items):
        latents[i, : lengths[i]] = it["latent"]
        eot[i, : lengths[i]] = it["eot"]
    ctc_targets = torch.tensor([c for it in items for c in it["ctc"]], dtype=torch.long)
    ctc_lengths = torch.tensor([len(it["ctc"]) for it in items], dtype=torch.long)
    return {
        "latents": latents, "lengths": lengths, "eot": eot,
        "prefix": [it["prefix"] for it in items], "suffix": [it["suffix"] for it in items],
        "target": [it["target"] for it in items], "ctc_targets": ctc_targets, "ctc_lengths": ctc_lengths,
        "ids": [it["id"] for it in items], "tasks": [it["task"] for it in items], "texts": [it["text"] for it in items],
    }


# -------------------------------------------------------------------- talker
class TalkerDataset(Dataset):
    def __init__(self, rows: list[dict], prompts: PromptBuilder, num_codebooks: int, max_frames: int):
        self.rows = [r for r in rows if r.get("codes")]
        self.prompts = prompts
        self.K = num_codebooks
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        codes = np.load(row["codes"]).astype(np.int64)
        if codes.shape[0] < self.K:
            raise ValueError(f"{row['codes']} has {codes.shape[0]} codebooks, config needs {self.K}")
        codes = codes[: self.K]
        if codes.shape[1] > self.max_frames or codes.shape[1] < 2:
            return None
        ids = self.prompts.ids(row["text"].strip())
        if not ids:
            return None
        return {"id": row.get("id", str(idx)), "text": row["text"], "text_ids": ids, "codes": torch.from_numpy(codes)}


def collate_talker(items: list[dict | None]) -> dict | None:
    items = [it for it in items if it is not None]
    if not items:
        return None
    return {"ids": [it["id"] for it in items], "texts": [it["text"] for it in items],
            "text_ids": [it["text_ids"] for it in items], "codes": [it["codes"] for it in items]}
