"""Frame-by-frame inference for the full-duplex experiment model.

Every 80 ms: take the guest's newest listener vector, run ONE position of the LLM (KV cache), pick the
text token (greedy) and sample the agent's 8 Mimi codes. When the text token is <tool_call> the model
writes the call as ordinary tokens, the backend runs it, its result is fed back, and the frame loop
continues. The agent's audio is decoded on the fly by the streaming Mimi decoder.
"""

from __future__ import annotations

import json
import os
import re
import time

import torch

from s2s.data.duplex import PAD_TOKEN, prefix_ids
from s2s.models.duplex import DuplexModel, unwrap_llm


def load_duplex(ckpt_dir: str, device: torch.device, dtype: torch.dtype | None = None, merge_lora: bool = True):
    from s2s.config import load_config
    from s2s.train.duplex import build_model
    from s2s.utils import resolve_dtype

    cfg = load_config(os.path.join(ckpt_dir, "config.yaml"), [f"train_duplex.init_from={ckpt_dir}",
                                                               "train_duplex.gradient_checkpointing=false"])
    dtype = dtype or resolve_dtype(cfg.train_duplex.get("dtype", "auto"), device)
    model, tok = build_model(cfg, device, dtype)
    if merge_lora and hasattr(model.llm, "merge_and_unload"):  # same outputs, about half the GPU kernels per frame
        model.llm = model.llm.merge_and_unload()
    model.eval()
    return model, tok, cfg, dtype


def sample(logits: torch.Tensor, temperature: float, top_k: int, gen: torch.Generator | None) -> torch.Tensor:
    """One code, kept on the GPU (no CPU sync)."""
    if temperature <= 0:
        return logits.argmax()
    logits = logits.float() / temperature
    if top_k:
        v, i = logits.topk(min(top_k, logits.shape[-1]))
        return i[torch.multinomial(torch.softmax(v, -1), 1, generator=gen)[0]]
    return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)[0]


class DuplexSession:
    def __init__(self, model: DuplexModel, tokenizer, system: str, tools: list | None, backend=None,
                 audio_temperature: float = 0.6, audio_top_k: int = 30, seed: int = 0, max_tool_tokens: int = 160):
        self.m, self.tok, self.backend = model, tokenizer, backend
        self.device = model.in_scale.device
        self.backbone, self.lm_head, _ = unwrap_llm(model.llm)
        self.pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        self.call_id = tokenizer.convert_tokens_to_ids("<tool_call>")
        self.call_end = tokenizer.convert_tokens_to_ids("</tool_call>")
        self.temp, self.top_k, self.max_tool_tokens = audio_temperature, audio_top_k, max_tool_tokens
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.cache = None
        self.llm_dtype = next(self.backbone.parameters()).dtype
        self.prev_text = self.pad_id
        self.prev_col = torch.full((model.K,), model.card, dtype=torch.long, device=self.device)
        self.grid: list[torch.Tensor] = []
        self.texts: list[int] = []
        self.tool_log: list[dict] = []
        self.t = 0
        self.timing = {"llm": 0.0, "depth": 0.0, "frames": 0}
        self._run(self._embed(prefix_ids(tokenizer, system, tools)))

    @torch.no_grad()
    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """Feed positions x [N, H]; returns the last hidden state [H] (in the LLM's dtype)."""
        from transformers import DynamicCache

        if self.cache is None:
            self.cache = DynamicCache()
        out = self.backbone(inputs_embeds=x[None].to(self.llm_dtype), past_key_values=self.cache, use_cache=True)
        return out.last_hidden_state[0, -1]

    def _embed(self, ids: list[int]) -> torch.Tensor:
        return self.m.text_embed(torch.tensor(ids, device=self.device, dtype=torch.long))

    @torch.no_grad()
    def _depth_sample(self, h: torch.Tensor, text_id: int) -> torch.Tensor:
        """The 8 codes of this grid column, one depth-transformer position each (KV cache, like training)."""
        m = self.m
        h = h.float()
        x0 = m.depth_in(h[None]) + m.depth_text(self._embed([text_id]).float())   # [1, d]
        cache, x, codes = m.depth.new_cache(), x0, []
        for j in range(m.K):
            out = m.depth(x[:, None], cache=cache, start_pos=j)[:, -1]
            c = sample(m.heads[j](out)[0], self.temp, self.top_k, self.gen)
            codes.append(c)
            if j + 1 < m.K:
                x = x0 + m.depth_code_emb[j](c.view(1))
        return torch.stack(codes)

    @torch.no_grad()
    def step(self, guest_vec: torch.Tensor) -> dict:
        """One 80 ms frame. Returns {"text": id, "frame": [K] codes of a finished frame or None, "tool": ...}."""
        m = self.m
        x = m.frame_inputs(guest_vec[None].to(self.device).float(), torch.tensor([self.prev_text], device=self.device),
                           self.prev_col[None]).float()
        t0 = time.perf_counter()
        h = self._run(x)
        text_id = int(self.m.text_logits(h[None])[0].argmax())   # the one CPU sync per frame
        t1 = time.perf_counter()
        col = self._depth_sample(h, text_id)
        self.timing["llm"] += t1 - t0
        self.timing["depth"] += time.perf_counter() - t1
        self.timing["frames"] += 1
        if self.t < m.delay:
            col[1:] = -1
        self.grid.append(col)
        self.texts.append(text_id)
        self.prev_text = text_id
        self.prev_col = col.masked_fill(col < 0, m.card)
        tool = None
        if text_id == self.call_id:
            tool = self._tool_call()
        frame = None
        if self.t >= m.delay:  # frame t - delay is complete now
            prev = self.grid[self.t - m.delay]
            frame = torch.cat([prev[:1], col[1:]]) if m.delay else col
        self.t += 1
        return {"text": text_id, "frame": frame, "tool": tool}

    @torch.no_grad()
    def _tool_call(self) -> dict:
        ids, pending = [self.call_id], [self.call_id]
        for _ in range(self.max_tool_tokens):
            h = self._run(self._embed(pending))
            nxt = int(self.m.text_logits(h[None])[0].argmax())
            ids.append(nxt)
            pending = [nxt]
            if nxt == self.call_end:
                break
        text = self.tok.decode(ids)
        match = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", text, re.S)
        call, result = None, {"error": "could not parse the tool call"}
        if match:
            try:
                call = json.loads(match.group(1))
                result = self.backend.execute(call) if self.backend else {"success": True}
            except json.JSONDecodeError:
                pass
        res_ids = self.tok("<tool_response>\n" + json.dumps(result) + "\n</tool_response>",
                           add_special_tokens=False).input_ids
        self._run(self._embed(pending + res_ids))  # the last call token (not fed yet) and the result
        self.prev_text = res_ids[-1]
        entry = {"frame": self.t, "call": call, "raw": text, "result": result}
        self.tool_log.append(entry)
        return entry


def run_offline(session: DuplexSession, guest_feats: torch.Tensor, codec=None) -> dict:
    """Feed a whole guest channel frame by frame; returns the agent's text per frame, tool calls and audio."""
    import numpy as np

    from s2s.models.codec import StreamingDecoder

    dec = StreamingDecoder(codec) if codec is not None else None
    wav, t0 = [], time.time()
    for t in range(guest_feats.shape[0]):
        out = session.step(guest_feats[t])
        if dec is not None and out["frame"] is not None:
            wav.append(dec.push([out["frame"].clamp(min=0)]))
    secs = time.time() - t0
    return {"texts": session.texts, "tools": session.tool_log,
            "audio": np.concatenate(wav) if wav else np.zeros(0, np.float32),
            "rtf": secs / max(1e-6, guest_feats.shape[0] * 0.08)}
