"""Batched serving of the speech thinker (adapter v3 / v4: Qwen3-4B + merged speech LoRA): many calls on one GPU.

The single-call agent (runtime/agent.py) runs one conversation at a time. Each reply word costs one full pass of
the 4B, and the time of that pass is mostly reading the 4B's weights from GPU memory; that cost is the same for 1
or 30 conversations stepped together. So this engine steps every conversation that is currently answering in ONE
pass, and fits the new turns (prefill) of the others in between ("continuous batching").

    BatchedThinker   the model side. A preallocated KV cache of `slots` conversations x `max_tokens` positions
                     (one slot per call, rows never mixed) and a batched forward of the Qwen3 decoder that reads
                     and writes each row at its own positions. The weights are the HF model's own modules
                     (same numbers as the normal path; checked in tests/test_integration.py).
    BatchScheduler   a thread that owns the GPU: takes turns submitted by calls (text ids and speech features),
                     runs the adapter, prefills a limited number of tokens per step (so answering calls are never
                     held up for long), then one decode step for every call that is answering, and reports each
                     token back with its time.

Not here (yet): the listener (each call's Nemotron stream; the load test uses stored features), the voice
(Kokoro), prompt-lookup speculation and CUDA graphs of the single-call path.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


class BatchedThinker:
    def __init__(self, model, slots: int, max_tokens: int, device: torch.device, dtype: torch.dtype):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        self.model = model
        self.base = model.model
        self.layers = list(self.base.layers)
        cfg = model.config
        self.hq, self.hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.slots, self.max_tokens = int(slots), int(max_tokens)
        self.device, self.dtype = device, dtype
        shape = (self.slots, self.hkv, self.max_tokens, self.dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in self.layers]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in self.layers]
        self.lengths = [0] * self.slots
        self.free = list(range(self.slots))
        self._lock = threading.Lock()  # calls open / close from their own threads
        self._rope = apply_rotary_pos_emb
        self._gqa = True

    @property
    def cache_gb(self) -> float:
        return 2 * len(self.layers) * self.k[0].numel() * self.k[0].element_size() / 2**30

    # ------------------------------------------------------------------ slots
    def alloc(self) -> int:
        with self._lock:
            if not self.free:
                raise RuntimeError(f"all {self.slots} conversation slots are in use")
            slot = self.free.pop(0)
            self.lengths[slot] = 0
            return slot

    def release(self, slot: int) -> None:
        with self._lock:
            self.lengths[slot] = 0
            self.free.append(slot)
            self.free.sort()  # low slots first: answering calls stay close together (a view, not a copy)

    def crop(self, slot: int, n: int) -> None:
        """Forget positions from n on (later writes overwrite them; the mask never reads past a row's length)."""
        self.lengths[slot] = min(self.lengths[slot], int(n))

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, slots: list[int], x: torch.Tensor, all_positions: bool = False) -> torch.Tensor:
        """x [B, T, H] input embeddings, row b continuing conversation slots[b] at its current length.
        Returns logits at the last position [B, V] (all positions: [B, T, V]). Rows are appended in place."""
        b, t, _ = x.shape
        for s in slots:
            if self.lengths[s] + t > self.max_tokens:
                raise RuntimeError(f"conversation in slot {s} is longer than max_tokens={self.max_tokens}")
        start = torch.tensor([self.lengths[s] for s in slots], device=self.device)
        pos = start[:, None] + torch.arange(t, device=self.device)[None]             # [B, T]
        rows = torch.tensor(slots, device=self.device)
        ridx = rows[:, None].expand(b, t)
        span = int(pos.max()) + 1
        keys = torch.arange(span, device=self.device)
        mask = keys[None, None, None, :] <= pos[:, None, :, None]                   # [B, 1, T, span]
        x = x.to(self.dtype)
        cos, sin = self.base.rotary_emb(x, pos)
        contiguous = b == 1 or slots == list(range(slots[0], slots[0] + b))
        for i, layer in enumerate(self.layers):
            at = layer.self_attn
            h = layer.input_layernorm(x)
            q = at.q_norm(at.q_proj(h).view(b, t, self.hq, self.dim)).transpose(1, 2)
            k = at.k_norm(at.k_proj(h).view(b, t, self.hkv, self.dim)).transpose(1, 2)
            v = at.v_proj(h).view(b, t, self.hkv, self.dim).transpose(1, 2)
            q, k = self._rope(q, k, cos, sin)
            kc, vc = self.k[i], self.v[i]
            kc[ridx, :, pos] = k.transpose(1, 2)                                    # [B, T, Hkv, D] at each row's positions
            vc[ridx, :, pos] = v.transpose(1, 2)
            if contiguous:  # a view, no copy
                kb, vb = kc[slots[0]: slots[0] + b, :, :span], vc[slots[0]: slots[0] + b, :, :span]
            else:
                kb, vb = kc[:, :, :span].index_select(0, rows), vc[:, :, :span].index_select(0, rows)
            o = self._attend(q, kb, vb, mask, at.scaling)
            x = x + at.o_proj(o.transpose(1, 2).reshape(b, t, self.hq * self.dim))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        for s in slots:
            self.lengths[s] += t
        x = self.base.norm(x if all_positions else x[:, -1:])
        logits = self.model.lm_head(x)
        return logits if all_positions else logits[:, -1]

    def _attend(self, q, k, v, mask, scale):
        if self._gqa:
            try:
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)
            except TypeError:  # torch < 2.5
                self._gqa = False
        rep = self.hq // self.hkv
        return F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                              attn_mask=mask, scale=scale)


# ===================================================================== scheduler
@dataclass
class Job:
    """One turn of one call: what to feed (pieces in order), then (optionally) generate a reply."""
    conv: "Conversation"
    pieces: list                      # ("ids", [token ids]) | ("speech", latents [T, D]) | ("embeds", [n, H])
    generate: bool = True
    max_new: int = 200
    temperature: float = 0.0
    on_token: Callable[[int, float], None] | None = None   # (token id, perf_counter time), on the engine thread
    on_done: Callable[[dict], None] | None = None          # stats, on the engine thread
    submitted: float = field(default_factory=time.perf_counter)


@dataclass
class Conversation:
    slot: int
    pending_ids: list = field(default_factory=list)  # fed before the next job (the <|im_end|> after a reply)
    job: Job | None = None
    embeds: torch.Tensor | None = None               # the job's input still to prefill
    next_token: int | None = None
    n_new: int = 0
    stats: dict = field(default_factory=dict)


class BatchScheduler(threading.Thread):
    def __init__(self, engine: BatchedThinker, embed: Callable[[list[int]], torch.Tensor], adapter=None,
                 boundary: tuple[torch.Tensor, torch.Tensor] | None = None, eos_ids: set | None = None,
                 im_end_id: int | None = None, prefill_budget: int = 1024, seed: int = 0):
        super().__init__(daemon=True)
        self.engine, self.embed, self.adapter = engine, embed, adapter
        self.boundary = boundary
        self.eos_ids, self.im_end_id = set(eos_ids or []), im_end_id
        self.prefill_budget = int(prefill_budget)
        self.inbox: queue.Queue = queue.Queue()
        self.prefilling: list[Conversation] = []
        self.answering: list[Conversation] = []
        self.gen = torch.Generator(device=engine.device).manual_seed(seed)
        self.stop_flag = False
        self.error: BaseException | None = None
        self.stats = {"decode_steps": 0, "decode_rows": 0, "max_batch": 0, "decode_s": 0.0, "prefill_s": 0.0,
                      "prefill_tokens": 0}

    # ---------------------------------------------------------------- calls
    def open(self) -> Conversation:
        return Conversation(slot=self.engine.alloc())

    def close(self, conv: Conversation) -> None:
        self.inbox.put(("close", conv))

    def submit(self, job: Job) -> None:
        self.inbox.put(("job", job))

    def shutdown(self) -> None:
        self.stop_flag = True
        self.inbox.put(("stop", None))

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        try:
            while not self.stop_flag:
                self._take(block=not (self.prefilling or self.answering))
                self._prefill_step()
                self._decode_step()
        except BaseException as e:  # noqa: BLE001 - report to the caller, never hang it
            self.error = e
            for c in self.prefilling + self.answering:
                if c.job and c.job.on_done:
                    c.job.on_done({"error": repr(e)})

    def _take(self, block: bool) -> None:
        while True:
            try:
                kind, item = self.inbox.get(block=block, timeout=0.5 if block else None)
            except queue.Empty:
                return
            block = False
            if kind == "stop":
                return
            if kind == "close":  # a call that hung up, maybe mid-reply: stop its work before its slot is reused
                for lst in (self.prefilling, self.answering):
                    if item in lst:
                        lst.remove(item)
                item.job = None
                self.engine.release(item.slot)
            elif kind == "job":
                self._start(item)

    @torch.no_grad()
    def _start(self, job: Job) -> None:
        conv = job.conv
        parts, ids = [], list(conv.pending_ids)
        conv.pending_ids = []

        def flush():
            if ids:
                parts.append(self.embed(list(ids)))
                ids.clear()

        for kind, value in job.pieces:
            if kind == "ids":
                ids.extend(value)
                continue
            flush()
            if kind == "speech":  # the adapter's vectors between the start / end of speech markers
                start, end = self.boundary
                emb = self.adapter(value[None].to(self.engine.device))["embeds"][0]
                parts.extend([start[None], emb, end[None]])
            else:
                parts.append(value)
        flush()
        conv.job, conv.n_new, conv.next_token = job, 0, None
        conv.embeds = torch.cat([p.to(self.engine.dtype) for p in parts], 0) if parts else None
        conv.stats = {"queued_ms": 0.0, "prefill_tokens": 0 if conv.embeds is None else int(conv.embeds.shape[0])}
        self.prefilling.append(conv)

    @torch.no_grad()
    def _prefill_step(self) -> None:
        budget = self.prefill_budget
        while self.prefilling and budget > 0:
            conv = self.prefilling[0]
            if conv.stats["queued_ms"] == 0.0:
                conv.stats["queued_ms"] = (time.perf_counter() - conv.job.submitted) * 1000
            t0 = time.perf_counter()
            logits = None
            if conv.embeds is not None and conv.embeds.shape[0]:
                n = min(budget, conv.embeds.shape[0])
                logits = self.engine.forward([conv.slot], conv.embeds[None, :n])[0]
                conv.embeds = conv.embeds[n:]
                budget -= n
                self.stats["prefill_tokens"] += n
            self.stats["prefill_s"] += time.perf_counter() - t0
            if conv.embeds is not None and conv.embeds.shape[0]:
                return  # the rest of this prefill next step: answering calls go first
            self.prefilling.pop(0)
            conv.stats["prefill_ms"] = (time.perf_counter() - conv.job.submitted) * 1000
            if conv.job.generate and logits is not None:
                self._emit(conv, self._sample(logits[None], [conv.job.temperature])[0])
            else:
                self._finish(conv, "fed")

    @torch.no_grad()
    def _decode_step(self) -> None:
        if not self.answering:
            return
        convs = sorted(self.answering, key=lambda c: c.slot)
        toks = torch.tensor([c.next_token for c in convs], device=self.engine.device)
        t0 = time.perf_counter()
        logits = self.engine.forward([c.slot for c in convs], self.embed(toks.tolist())[:, None])
        picks = self._sample(logits, [c.job.temperature for c in convs])
        self.stats["decode_s"] += time.perf_counter() - t0
        self.stats["decode_steps"] += 1
        self.stats["decode_rows"] += len(convs)
        self.stats["max_batch"] = max(self.stats["max_batch"], len(convs))
        for c, tok in zip(convs, picks):
            self._emit(c, tok)

    def _sample(self, logits: torch.Tensor, temps: list[float]) -> list[int]:
        greedy = logits.argmax(-1)
        if all(t <= 0 for t in temps):
            return greedy.tolist()
        out = []
        for row, t, g in zip(logits, temps, greedy.tolist()):
            if t <= 0:
                out.append(g)
            else:
                p = torch.softmax(row.float() / t, -1)
                out.append(int(torch.multinomial(p, 1, generator=self.gen)))
        return out

    def _emit(self, conv: Conversation, tok: int) -> None:
        now = time.perf_counter()
        if conv in self.answering:
            self.answering.remove(conv)
        if tok in self.eos_ids:
            if self.im_end_id is not None:
                conv.pending_ids = [self.im_end_id]  # keep the cache in the chat format (fed with the next job)
            self._finish(conv, "eos")
            return
        if conv.job.on_token:
            conv.job.on_token(tok, now)
        conv.n_new += 1
        if conv.n_new >= conv.job.max_new:
            conv.pending_ids = [tok] + ([self.im_end_id] if self.im_end_id is not None else [])
            self._finish(conv, "max_new")
            return
        conv.next_token = tok
        self.answering.append(conv)

    def _finish(self, conv: Conversation, reason: str) -> None:
        job, conv.job = conv.job, None
        conv.stats.update(reason=reason, new_tokens=conv.n_new, length=self.engine.lengths[conv.slot])
        if job and job.on_done:
            job.on_done(dict(conv.stats))
