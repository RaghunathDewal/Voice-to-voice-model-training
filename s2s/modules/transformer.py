"""A small causal transformer (pre-norm, RoPE, SDPA) with an optional KV cache.

Used by the speech adapter, the temporal talker and the depth transformer.
Batches are right-padded; because attention is causal, real positions never
attend to padding, so no padding mask is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float, dtype: torch.dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_base: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, start_pos: int, cache: dict | None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        positions = torch.arange(start_pos, start_pos + t, device=x.device)
        cos, sin = rope_cos_sin(positions, self.head_dim, self.rope_base, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        t_k = k.shape[2]
        if t == 1:
            mask, is_causal = None, False
        elif t_k == t:
            mask, is_causal = None, True
        else:  # chunk appended to a cache: query i may see keys up to (t_k - t + i)
            qi = torch.arange(t, device=x.device)[:, None] + (t_k - t)
            kj = torch.arange(t_k, device=x.device)[None, :]
            mask, is_causal = kj <= qi, False
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float, rope_base: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout, rope_base)
        self.norm2 = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, start_pos: int, cache: dict | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), start_pos, cache))
        return x + self.drop(self.ff(self.norm2(x)))


class CausalTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(d_model, n_heads, ff_mult, dropout, rope_base) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.gradient_checkpointing = False

    def new_cache(self) -> list[dict]:
        return [{} for _ in self.layers]

    def forward(self, x: torch.Tensor, cache: list[dict] | None = None, start_pos: int = 0) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.gradient_checkpointing and self.training and cache is None:
                x = torch.utils.checkpoint.checkpoint(layer, x, start_pos, None, use_reentrant=False)
            else:
                x = layer(x, start_pos, layer_cache)
        return self.norm(x)
