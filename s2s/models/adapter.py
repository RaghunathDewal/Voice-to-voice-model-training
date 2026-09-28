"""Speech adapter: Mimi latents (12.5 Hz) -> Qwen input embeddings.

    latents [B,T,512] -> LayerNorm -> Linear -> causal transformer -> hidden [B,T,d]
        hidden -> Linear -> RMSNorm * scale -> speech embeddings [B,T,H_llm]
        hidden -> CTC head (upsampled x4 -> 50 Hz, character vocab)   [aux: transcripts]
        hidden -> end-of-turn head (per frame logit)                  [aux: endpointing]

Everything is causal, so embeddings for frame t never change when later audio
arrives. That is what makes incremental prefill of the LLM possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from s2s.modules.transformer import CausalTransformer, RMSNorm
from s2s.text import CTC_VOCAB


class SpeechAdapter(nn.Module):
    def __init__(self, latent_dim: int, llm_dim: int, d_model: int = 1024, n_layers: int = 4,
                 n_heads: int = 16, ff_mult: int = 4, dropout: float = 0.1, ctc_upsample: int = 4):
        super().__init__()
        self.hparams = dict(latent_dim=latent_dim, llm_dim=llm_dim, d_model=d_model, n_layers=n_layers,
                            n_heads=n_heads, ff_mult=ff_mult, dropout=dropout, ctc_upsample=ctc_upsample)
        self.in_norm = nn.LayerNorm(latent_dim)
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.encoder = CausalTransformer(d_model, n_layers, n_heads, ff_mult, dropout)
        self.out_proj = nn.Linear(d_model, llm_dim)
        self.out_norm = RMSNorm(llm_dim)
        # scale so speech embeddings have the same RMS as Qwen text embeddings (set by init_scale)
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.speech_start = nn.Parameter(torch.randn(llm_dim) * 0.02)
        self.speech_end = nn.Parameter(torch.randn(llm_dim) * 0.02)
        self.ctc_upsample = ctc_upsample
        self.ctc_head = nn.Linear(d_model, len(CTC_VOCAB) * ctc_upsample)
        self.eot_head = nn.Linear(d_model, 1)

    @classmethod
    def from_config(cls, cfg, latent_dim: int, llm_dim: int) -> "SpeechAdapter":
        return cls(latent_dim=latent_dim, llm_dim=llm_dim, d_model=cfg.d_model, n_layers=cfg.n_layers,
                   n_heads=cfg.n_heads, ff_mult=cfg.ff_mult, dropout=cfg.dropout, ctc_upsample=cfg.ctc_upsample)

    @torch.no_grad()
    def init_scale(self, text_embedding_rms: float) -> None:
        self.out_scale.fill_(text_embedding_rms)
        unit = torch.randn_like(self.speech_start)
        self.speech_start.copy_(unit / unit.pow(2).mean().sqrt() * text_embedding_rms)
        unit = torch.randn_like(self.speech_end)
        self.speech_end.copy_(unit / unit.pow(2).mean().sqrt() * text_embedding_rms)

    def boundary_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.speech_start, self.speech_end

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self.in_proj(self.in_norm(latents)))
        embeds = self.out_norm(self.out_proj(h)) * self.out_scale
        b, t, _ = h.shape
        ctc_logits = self.ctc_head(h).view(b, t * self.ctc_upsample, len(CTC_VOCAB))
        eot_logits = self.eot_head(h).squeeze(-1)
        return {"hidden": h, "embeds": embeds, "ctc_logits": ctc_logits, "eot_logits": eot_logits}

    def save(self, path: str) -> None:
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "SpeechAdapter":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model
