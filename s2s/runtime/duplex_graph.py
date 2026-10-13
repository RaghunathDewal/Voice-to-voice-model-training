"""The duplex model's whole 80 ms frame replayed from ONE CUDA graph (DuplexSession(..., graphs=True)).

Why: a frame is the listener-vector fusion, one position of the 0.6B LLM, the text pick and 8 depth-transformer
steps with sampling, launched from Python as many hundreds of tiny GPU kernels. On a T4 the launches cost more
than the maths (1.53x real time), and a duplex model that is slower than real time falls further behind every
frame. The graph records the frame once and replays it with a single launch.

What it needs (as runtime/cuda_graph.py for the 4B):
  - the LLM's KV cache is a StaticCache of `max_tokens` positions (one conversation at a time: a new session
    resets it); prompts and tool calls (many tokens) run normally into the same cache;
  - the depth transformer runs WITHOUT a KV cache: its 8 positions are a fixed buffer and step j reads position
    j of a causal pass over all 8 (later positions hold stale values the causal mask never lets it see);
  - sampling uses the Gumbel-max trick: argmax(logits / T + Gumbel noise) over the top-k is a sample from the
    softmax over the top-k. The noise is drawn outside the graph (one kernel) into a fixed buffer.
The graph is checked against the normal path when it is recorded (zero noise = greedy codes, same text token, close hidden
state); if anything differs or fails, the session uses the normal path and says so.
"""

from __future__ import annotations

import time

import torch


class DuplexGraph:
    def __init__(self, model, max_tokens: int, temperature: float, top_k: int):
        from transformers import StaticCache

        from s2s.models.duplex import unwrap_llm

        self.m = model
        self.backbone = unwrap_llm(model.llm)[0]
        cfg = self.backbone.config
        self.device = model.in_scale.device
        self.dtype = next(self.backbone.parameters()).dtype
        self.max_tokens, self.temp = int(max_tokens), float(temperature)
        self.top_k = min(int(top_k), model.card) if top_k else model.card
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.cache = StaticCache(config=cfg, max_cache_len=self.max_tokens)
        self.cache.early_initialization(batch_size=1, num_heads=cfg.num_key_value_heads, head_dim=head_dim,
                                        dtype=self.dtype, device=self.device)
        self.n = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        feat_dim = model.feat_proj.in_features
        d, K = model.depth_in.out_features, model.K
        # fixed input / output buffers of the recorded frame
        self.vec = torch.zeros(feat_dim, device=self.device)
        self.prev_text = torch.zeros(1, dtype=torch.long, device=self.device)
        self.prev_col = torch.full((K,), model.card, dtype=torch.long, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.noise = torch.zeros(K, self.top_k, device=self.device)
        self.out_text = torch.zeros((), dtype=torch.long, device=self.device)
        self.out_col = torch.zeros(K, dtype=torch.long, device=self.device)
        self.out_h = torch.zeros(cfg.hidden_size, dtype=self.dtype, device=self.device)
        self.d = d

    # ------------------------------------------------------------------ cache
    def reset(self) -> None:
        self.cache.reset()
        self.n = 0

    def _check(self, n: int) -> None:
        if self.n + n > self.max_tokens:
            raise RuntimeError(f"the call is longer than duplex graph_max_tokens={self.max_tokens} "
                               f"(~{self.max_tokens * 0.08 / 60:.1f} min of frames)")

    @torch.no_grad()
    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Normal (eager) path into the static cache: x [N, H] -> last hidden state [H]."""
        n = int(x.shape[0])
        self._check(n)
        pos = torch.arange(self.n, self.n + n, device=self.device)
        out = self.backbone(inputs_embeds=x[None].to(self.dtype), past_key_values=self.cache, cache_position=pos,
                            use_cache=True)
        self.n += n
        return out.last_hidden_state[0, -1]

    # ------------------------------------------------------------------ the frame
    def _pick(self, logits: torch.Tensor, j: int) -> torch.Tensor:
        v, i = logits.float().topk(self.top_k)  # (recorded once: the temperature is baked into the graph)
        return i[(v / max(self.temp, 1e-6) + self.noise[j]).argmax()]

    def _frame(self) -> None:
        m = self.m
        x = m.frame_inputs(self.vec[None], self.prev_text, self.prev_col[None]).float()
        h = self.backbone(inputs_embeds=x[None].to(self.dtype), past_key_values=self.cache, cache_position=self.pos,
                          use_cache=True).last_hidden_state[0, -1]
        text = m.text_logits(h[None])[0].argmax()
        x0 = m.depth_in(h.float()[None]) + m.depth_text(m.text_embed(text.view(1)).float())  # [1, d]
        seq = x0[:, None].repeat(1, m.K, 1)  # [1, K, d]: position j is filled once code j-1 is known
        codes = []
        for j in range(m.K):
            out = m.depth(seq)[0, j]
            c = self._pick(m.heads[j](out[None])[0], j)
            codes.append(c)
            if j + 1 < m.K:
                seq[0, j + 1] = x0[0] + m.depth_code_emb[j](c.view(1))[0]
        self.out_text.copy_(text)
        self.out_col.copy_(torch.stack(codes))
        self.out_h.copy_(h)

    def _gumbel(self, gen: torch.Generator | None, greedy: bool) -> None:
        if greedy or self.temp <= 0:
            self.noise.zero_()
        else:
            self.noise.uniform_(1e-9, 1.0, generator=gen)
            self.noise.log_().neg_().log_().neg_()

    @torch.no_grad()
    def frame(self, vec: torch.Tensor, prev_text: int, prev_col: torch.Tensor, gen: torch.Generator | None,
              greedy: bool = False):
        """One frame: (text id tensor, [K] codes, h). Replayed from the graph when recorded."""
        self._check(1)
        self.vec.copy_(vec.reshape(-1).float())
        self.prev_text.fill_(int(prev_text))
        self.prev_col.copy_(prev_col)
        self.pos.fill_(self.n)
        self._gumbel(gen, greedy)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._frame()
        self.n += 1
        return self.out_text.clone(), self.out_col.clone(), self.out_h.clone()

    # ------------------------------------------------------------------ recording
    @torch.no_grad()
    def capture(self, prefix: torch.Tensor, reference) -> bool:
        """Record the frame and check it against the normal path. prefix: [N, H] prompt embeddings;
        reference(vec, prev_text, prev_col) -> (text id, [K] greedy codes, h) from the normal path after the same
        prompt. Returns True when the graph is in use."""
        if self.device.type != "cuda":
            print("[duplex graphs] not a CUDA/ROCm device: graphs off", flush=True)
            return False
        t0 = time.time()
        try:
            m = self.m
            vec = torch.randn(self.vec.shape[0], device=self.device)
            col = torch.full((m.K,), m.card, dtype=torch.long, device=self.device)
            ref_text, ref_col, ref_h = reference(vec, m.pad_id, col)
            # record on a side stream after warm-up runs (as torch.cuda.graphs requires)
            self.reset()
            self.run(prefix)
            self.vec.copy_(vec)
            self.prev_text.fill_(m.pad_id)
            self.prev_col.copy_(col)
            self.pos.fill_(self.n)
            self.noise.zero_()
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._frame()
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._frame()
            self.graph = graph
            # check: the same prompt and frame again, replayed, against the normal path
            self.reset()
            self.run(prefix)
            text, codes, h = self.frame(vec, m.pad_id, col, None, greedy=True)
            diff = float((h.float() - ref_h.float()).abs().max())
            scale = float(ref_h.float().abs().max()) or 1.0
            same_codes = int((codes == ref_col).sum())
            if int(text) != int(ref_text) or diff > 0.05 * scale or same_codes < m.K - 2:
                raise RuntimeError(f"graph frame differs from the normal path (text {int(text)} vs {int(ref_text)}, "
                                   f"hidden diff {diff:.3g} of {scale:.3g}, {same_codes}/{m.K} codes equal)")
            # timing: replayed frames
            torch.cuda.synchronize(self.device)
            t = time.time()
            for _ in range(20):
                self.frame(vec, m.pad_id, col, None)
            torch.cuda.synchronize(self.device)
            ms = (time.time() - t) * 50
            self.reset()
            print(f"[duplex graphs] frame graph recorded in {time.time() - t0:.1f}s, checked against the normal path "
                  f"(hidden diff {diff:.2g}, {same_codes}/{m.K} codes equal): {ms:.1f} ms per 80 ms frame "
                  f"(LLM + text + 8 depth steps)", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 - never break the session: fall back to the normal path
            self.graph = None
            self.reset()
            print(f"[duplex graphs] graphs off ({type(e).__name__}: {e}); using the normal path", flush=True)
            return False
