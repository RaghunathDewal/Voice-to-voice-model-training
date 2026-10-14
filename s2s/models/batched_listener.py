"""Many live Nemotron streams encoded together (one listener per call, one GPU pass for all of them).

Each call keeps its own StreamingListener: audio buffer, mel position, attention K/V cache (sliding window of
`config.sliding_window - 1` frames) and convolution padding caches. Every 160 ms each call has one new chunk;
calls whose caches have the same shape are stacked along the batch dimension, encoded in one pass, and their
caches split back. In steady state (after the first ~5.6 s of a call, when its window is full) all calls share
one shape, so they form one batch; calls still filling their window run in small groups of equal length.

Exact: a call gets the same vectors as when encoded alone (checked in tests/test_integration.py). Two calls are
only stacked when their attention masks and position encodings are identical: same cache length, and when
the window is full, the same position within a chunk (cumulative frames modulo 2).
"""

from __future__ import annotations

import copy
from collections import defaultdict

import torch

from s2s.models.streaming_listener import StreamingListener


class BatchedListener:
    def __init__(self, nemotron, lookahead: int = 1):
        self.nem, self.look = nemotron, int(lookahead)
        self.enc, self.device, self.dtype = nemotron.enc, nemotron.device, nemotron.dtype
        self.window = int(getattr(self.enc.config, "sliding_window", 71))
        self.stats = {"passes": 0, "chunks": 0, "max_batch": 0, "seconds": 0.0}

    def new_stream(self) -> StreamingListener:
        return StreamingListener(self.nem, self.look)

    # ------------------------------------------------------------------ steps
    @torch.no_grad()
    def step(self, streams: list[StreamingListener]) -> dict[int, list[torch.Tensor]]:
        """Encode every chunk that is ready in `streams`. Returns {id(stream): [new frames [D]]}."""
        import time

        out: dict[int, list[torch.Tensor]] = {id(s): [] for s in streams}
        t0 = time.perf_counter()
        while True:
            groups: dict[tuple, list] = defaultdict(list)
            for s in streams:
                feats = s._next_chunk()
                if feats is not None:
                    groups[self._key(s, feats)].append((s, feats))
            if not groups:
                break
            for items in groups.values():
                for s, frames in self._run(items):
                    out[id(s)].extend(frames)
        self.stats["seconds"] += time.perf_counter() - t0
        return out

    def _key(self, s: StreamingListener, feats: torch.Tensor) -> tuple:
        if s.kv is None:
            return ("first", tuple(feats.shape))
        cum = int(s.kv.layers[0].cumulative_length)
        if cum > self.window:  # window full: only the phase within a 2-frame chunk matters
            return ("full", cum % 2, tuple(feats.shape))
        return ("filling", cum, tuple(feats.shape))

    def _run(self, items: list) -> list:
        b = len(items)
        feats = torch.cat([f for _, f in items], 0).to(self.device, self.dtype)
        first = items[0][0].kv is None
        if b == 1:
            s = items[0][0]
            kv, pad, cum0 = s.kv, s.pad, None
        elif first:
            kv = pad = cum0 = None
        else:
            cum0 = [int(s.kv.layers[0].cumulative_length) for s, _ in items]
            kv, pad = _stack_kv([s.kv for s, _ in items]), _stack_pad([s.pad for s, _ in items])
        before = int(kv.layers[0].cumulative_length) if kv is not None else 0
        o = self.enc(input_features=feats, past_key_values=kv, padding_cache=pad, use_cache=True,
                     num_lookahead_tokens=self.look, output_attention_mask=False)
        self.stats["passes"] += 1
        self.stats["chunks"] += b
        self.stats["max_batch"] = max(self.stats["max_batch"], b)
        res = []
        added = int(o.past_key_values.layers[0].cumulative_length) - before
        hidden = o.last_hidden_state.float().cpu()
        for j, (s, _) in enumerate(items):
            if b == 1:
                s.kv, s.pad = o.past_key_values, o.padding_cache
            else:
                s.kv = _slice_kv(o.past_key_values, j, (cum0[j] if cum0 else 0) + added)
                s.pad = _slice_pad(o.padding_cache, j)
            frames = list(hidden[j])
            s.done += len(frames)
            res.append((s, frames))
        return res


# --------------------------------------------------------------- cache surgery
def _stack_kv(caches):
    m = copy.copy(caches[0])
    m.layers = []
    for i, layer in enumerate(caches[0].layers):
        nl = copy.copy(layer)
        nl.keys = torch.cat([c.layers[i].keys for c in caches], 0)
        nl.values = torch.cat([c.layers[i].values for c in caches], 0)
        m.layers.append(nl)
    return m


def _slice_kv(m, j: int, cumulative: int):
    c = copy.copy(m)
    c.layers = []
    for layer in m.layers:
        nl = copy.copy(layer)
        nl.keys, nl.values = layer.keys[j: j + 1], layer.values[j: j + 1]
        nl.cumulative_length = cumulative
        c.layers.append(nl)
    return c


def _stack_pad(pads):
    m = copy.copy(pads[0])
    m.layers = {}
    for key, layer in pads[0].layers.items():
        nl = copy.copy(layer)
        nl.cache = torch.cat([p.layers[key].cache for p in pads], 0)
        m.layers[key] = nl
    return m


def _slice_pad(m, j: int):
    c = copy.copy(m)
    c.layers = {}
    for key, layer in m.layers.items():
        nl = copy.copy(layer)
        nl.cache = layer.cache[j: j + 1].clone()  # the next pass writes into it in place
        c.layers[key] = nl
    return c
