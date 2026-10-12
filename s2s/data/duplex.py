"""Frame-aligned training streams for the full-duplex model (see s2s/models/duplex.py).

One call -> {prefix, guest [T, F], codes [K, T], text [T], segments}:
  * prefix: the system prompt with the tool list (chat template), as token ids
  * guest: the listener's vectors of the guest channel, one per 80 ms frame
  * codes: Mimi codes of the agent channel (silence included: being quiet is an output too)
  * text: the agent's inner-monologue token at each frame, PAD when there is none. A reply's tokens
    start at the frame the agent decides to speak (its audio follows LEAD frames later) and are spread
    over the speech at a steady pace; an interrupted reply keeps only the tokens before the cut, so
    the history holds what the guest actually heard.
  * segments: tool calls. At frame f the text token is <tool_call>; right after frame f the sequence
    gets the full call and the backend's result as ordinary token positions ("<tool_call>{...}
    </tool_call><tool_response>{...}</tool_response>"); the call is predicted, the result is given.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME = 0.08
PAD_TOKEN = "<|fim_pad|>"


def prefix_ids(tokenizer, system: str, tools: list | None) -> list[int]:
    text = tokenizer.apply_chat_template([{"role": "system", "content": system}], tools=tools, tokenize=False)
    return tokenizer(text, add_special_tokens=False).input_ids


def tool_segment(tokenizer, call: dict, result: dict) -> tuple[list[int], int]:
    """Token ids of one tool exchange and how many of them the model predicts (the call)."""
    call_text = "<tool_call>\n" + json.dumps({"name": call["name"], "arguments": call["arguments"]}) + "\n</tool_call>"
    res_text = "<tool_response>\n" + json.dumps(result) + "\n</tool_response>"
    call_ids = tokenizer(call_text, add_special_tokens=False).input_ids
    res_ids = tokenizer(res_text, add_special_tokens=False).input_ids
    return call_ids + res_ids, len(call_ids) - 1  # positions 0..n-2 predict tokens 1..n-1 of the call


def text_stream(tokenizer, timeline: dict, n_frames: int, pad_id: int, call_id: int,
                pace: float = 0.85) -> tuple[list[int], list[tuple[int, list[int], int]]]:
    text = [pad_id] * n_frames
    for seg in timeline["agent"]:
        ids = tokenizer(seg["text"], add_special_tokens=False).input_ids
        if not ids:
            continue
        n_full = max(1, int(math.ceil(seg["full"] / FRAME)))
        rate = max(1.0, pace * n_full / len(ids))
        cut = int(math.ceil(seg["audio_end"] / FRAME)) if seg.get("cut") else n_frames
        pos = seg["frame"]
        for i, tok in enumerate(ids):
            f = seg["frame"] + int(i * rate)
            f = max(f, pos)
            while f < n_frames and text[f] != pad_id:  # never overwrite (rare overlaps)
                f += 1
            if f >= min(cut, n_frames):
                break
            text[f] = tok
            pos = f + 1
    segments = []
    for t in timeline["tools"]:
        f = t["frame"]
        if f >= n_frames:
            continue
        text[f] = call_id
        ids, n_pred = tool_segment(tokenizer, t["call"], t["result"])
        segments.append((f, ids, n_pred))
    return text, segments


class DuplexDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, manifest_dir: str, max_frames: int = 625):
        self.rows, self.tok, self.root, self.max_frames = rows, tokenizer, manifest_dir, max_frames
        self.pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        self.call_id = tokenizer.convert_tokens_to_ids("<tool_call>")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        d = os.path.join(self.root, r["dir"])
        guest = np.load(os.path.join(d, "guest.npy"))
        codes = np.load(os.path.join(d, "agent.npy")).astype(np.int64)
        with open(os.path.join(d, "timeline.json")) as f:
            timeline = json.load(f)
        t = min(guest.shape[0], codes.shape[1], self.max_frames)
        text, segments = text_stream(self.tok, timeline, t, self.pad_id, self.call_id)
        return {"id": r["id"], "prefix": torch.tensor(prefix_ids(self.tok, r["system"], r.get("tools"))),
                "guest": torch.from_numpy(guest[:t].astype(np.float32)), "codes": torch.from_numpy(codes[:, :t]),
                "text": torch.tensor(text), "segments": segments, "timeline": timeline}


def collate_duplex(items: list[dict]) -> list[dict]:
    return items  # variable-length streams; the model assembles each sequence itself
