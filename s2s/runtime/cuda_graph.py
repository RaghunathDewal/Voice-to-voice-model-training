"""The thinker with a fixed-size KV cache, and its one-token reply step replayed from a CUDA graph
(runtime.cuda_graphs: true). HIP graphs on AMD GPUs: the same torch.cuda.CUDAGraph API under ROCm.

Why: a reply token is one forward pass of the 4B, launched from Python as hundreds of small GPU kernels. On a T4
(and even more on fast GPUs) launching them costs about as much as running them. A CUDA graph records the whole
step once and replays it with a single launch.

What it needs: tensors at fixed addresses. So the KV cache is a StaticCache of `max_tokens` positions, allocated
once and shared by the conversations (one at a time: a new session clears it), and the step's input (the token
embedding, its position) is copied into fixed buffers before each replay. Prompts / spoken turns (many tokens at
once) run normally into the same cache. The graph is checked against the normal path (same tokens, same logits)
when it is recorded; if anything differs or fails, the agent falls back to the normal path and says so.
"""

from __future__ import annotations

import time

import torch


class StaticThinker:
    def __init__(self, model, hidden_idx: list[int], need_hidden: bool, max_tokens: int, device: torch.device,
                 dtype: torch.dtype):
        from transformers import StaticCache

        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.model, self.hidden_idx, self.need_hidden = model, list(hidden_idx), need_hidden
        self.max_tokens, self.device, self.dtype = int(max_tokens), device, dtype
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.cache = StaticCache(config=cfg, max_cache_len=self.max_tokens)
        self.cache.early_initialization(batch_size=1, num_heads=cfg.num_key_value_heads, head_dim=head_dim,
                                        dtype=dtype, device=device)
        self.n = 0
        self.owner = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.x = torch.zeros(1, 1, cfg.hidden_size, dtype=dtype, device=device)  # the step's input embedding
        self.pos = torch.zeros(1, dtype=torch.long, device=device)  # the step's cache position
        self.out_logits: torch.Tensor | None = None
        self.out_hidden: list[torch.Tensor] = []

    # ------------------------------------------------------------------ sessions
    def reset(self, owner) -> None:
        """A new conversation takes the cache (the previous one can no longer be continued)."""
        self.cache.reset()
        self.n, self.owner = 0, owner

    def crop(self, owner, n: int) -> None:
        """Forget positions from n on (later steps overwrite them; the causal mask never looks past the current
        position, so what is left there is never read)."""
        self._check(owner, 0)
        self.n = min(self.n, int(n))

    def _check(self, owner, n: int) -> None:
        if owner is not self.owner:
            raise RuntimeError("runtime.cuda_graphs: another conversation has taken the thinker cache "
                               "(one conversation at a time with CUDA graphs)")
        if self.n + n > self.max_tokens:
            raise RuntimeError(f"the conversation is longer than runtime.graph_max_tokens={self.max_tokens}")

    # ------------------------------------------------------------------ forward
    def _run(self, embeds: torch.Tensor, positions: torch.Tensor):
        return self.model(inputs_embeds=embeds, past_key_values=self.cache, cache_position=positions, use_cache=True,
                          output_hidden_states=self.need_hidden, logits_to_keep=1)

    def _outputs(self, out) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = torch.stack([out.hidden_states[j][0, -1] for j in self.hidden_idx], 0) if self.need_hidden else None
        return out.logits[0, -1], hidden

    @torch.no_grad()
    def forward(self, owner, embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """embeds [n, H] -> (logits at the last position [V], selected hidden states [n_sel, H] or None)."""
        n = int(embeds.shape[0])
        self._check(owner, n)
        if n == 1 and self.graph is not None:
            self.x.copy_(embeds.reshape(self.x.shape).to(self.dtype))
            self.pos.fill_(self.n)
            self.graph.replay()
            self.n += 1
            hidden = torch.stack([h[0, -1] for h in self.out_hidden], 0) if self.need_hidden else None
            return self.out_logits[0, -1].clone(), (hidden.clone() if hidden is not None else None)
        positions = torch.arange(self.n, self.n + n, device=self.device)
        out = self._run(embeds[None].to(self.dtype), positions)
        self.n += n
        return self._outputs(out)

    # ------------------------------------------------------------------ recording
    @torch.no_grad()
    def capture(self, embed_fn, probe_ids: list[int]) -> bool:
        """Record the one-token step and check it against the normal path. Returns True when it is in use."""
        if self.device.type != "cuda":
            print("[graphs] not a CUDA/ROCm device: CUDA graphs off")
            return False
        t0 = time.time()
        try:
            owner = object()
            self.reset(owner)
            # the reference: the normal (dynamic cache) path on the same tokens
            prompt, step = embed_fn(probe_ids[:-1]), embed_fn(probe_ids[-1:])
            ref = self.model(inputs_embeds=prompt[None].to(self.dtype), use_cache=True)
            ref = self.model(inputs_embeds=step[None].to(self.dtype), past_key_values=ref.past_key_values,
                             use_cache=True, logits_to_keep=1).logits[0, -1].float()
            # record on a side stream after a few warm-up runs (as torch.cuda.graphs requires)
            self.forward(owner, prompt)
            self.x.copy_(step.reshape(self.x.shape).to(self.dtype))
            self.pos.fill_(self.n)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._run(self.x, self.pos)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run(self.x, self.pos)
            self.out_logits = out.logits
            self.out_hidden = [out.hidden_states[j] for j in self.hidden_idx] if self.need_hidden else []
            self.graph = graph
            # check: the same prompt again, the step replayed from the graph, against the reference
            self.reset(owner)
            self.forward(owner, prompt)
            got = self.forward(owner, step)[0].float()
            diff = float((got - ref).abs().max())
            scale = float(ref.abs().max()) or 1.0
            same_top = int(got.argmax()) == int(ref.argmax())
            if diff > 0.05 * scale or not same_top:
                raise RuntimeError(f"graph output differs from the normal path (max diff {diff:.3g} of {scale:.3g}, "
                                   f"same top token: {same_top})")
            # timing: replayed step vs the normal path
            self.reset(owner)
            self.forward(owner, prompt)
            torch.cuda.synchronize(self.device)
            t = time.time()
            for _ in range(10):
                self.forward(owner, step)
            torch.cuda.synchronize(self.device)
            graph_ms = (time.time() - t) * 100
            past = self.model(inputs_embeds=prompt[None].to(self.dtype), use_cache=True).past_key_values
            torch.cuda.synchronize(self.device)
            t = time.time()
            for _ in range(10):
                past = self.model(inputs_embeds=step[None].to(self.dtype), past_key_values=past, use_cache=True,
                                  output_hidden_states=True, logits_to_keep=1).past_key_values
            torch.cuda.synchronize(self.device)
            eager_ms = (time.time() - t) * 100
            self.reset(None)
            print(f"[graphs] CUDA graph recorded in {time.time() - t0:.1f}s, checked against the normal path "
                  f"(max diff {diff:.2g}): one reply token {graph_ms:.1f} ms (was {eager_ms:.1f} ms)", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 - never break the agent: fall back to the normal path
            self.graph = None
            self.reset(None)
            print(f"[graphs] CUDA graphs off ({type(e).__name__}: {e}); using the normal path", flush=True)
            return False
