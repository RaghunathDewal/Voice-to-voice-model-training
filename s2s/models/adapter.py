"""Speech adapter: frozen-encoder features (12.5 Hz) -> Qwen input embeddings.

The input encoder is Mimi (512-d latents) or any encoder from s2s/models/encoders.py
(e.g. Parakeet, 1024-d); its spec is stored in the checkpoint (hparams["encoder"]),
so the runtime always encodes speech with the encoder the adapter was trained on.

    latents [B,T,D] -> LayerNorm -> Linear -> causal transformer -> hidden [B,T,d]
        hidden -> Linear -> RMSNorm * scale -> speech embeddings [B,T,H_llm]
        hidden -> CTC head (upsampled x4 -> 50 Hz, character vocab)   [aux: transcripts]
        hidden -> end-of-turn head (per frame logit)                  [aux: endpointing]

Everything is causal, so embeddings for frame t never change when later audio
arrives. That is what makes incremental prefill of the LLM possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.modules.transformer import CausalTransformer, RMSNorm
from s2s.text import CTC_VOCAB


class SpeechAdapter(nn.Module):
    def __init__(self, latent_dim: int, llm_dim: int, d_model: int = 1024, n_layers: int = 4,
                 n_heads: int = 16, ff_mult: int = 4, dropout: float = 0.1, ctc_upsample: int = 4,
                 encoder: str = "mimi", token_vocab_size: int = 0):
        super().__init__()
        self.hparams = dict(latent_dim=latent_dim, llm_dim=llm_dim, d_model=d_model, n_layers=n_layers,
                            n_heads=n_heads, ff_mult=ff_mult, dropout=dropout, ctc_upsample=ctc_upsample,
                            encoder=encoder, token_vocab_size=token_vocab_size)
        self.encoder_spec = encoder
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
        # word-level CTC tied to the thinker's own (frozen) token embeddings: each output vector is scored by
        # its cosine similarity to the embeddings of a fixed set of thinker tokens (`tok_vocab`), plus a learned
        # blank. Pulls the speech vectors to where the thinker reads those words, and gives word confidences.
        self.token_vocab_size = token_vocab_size
        if token_vocab_size:
            self.register_buffer("tok_vocab", torch.zeros(token_vocab_size, dtype=torch.long))
            self.tok_blank = nn.Parameter(torch.randn(llm_dim) * 0.02)
            self.tok_scale = nn.Parameter(torch.tensor(10.0))

    @classmethod
    def from_config(cls, cfg, latent_dim: int, llm_dim: int) -> "SpeechAdapter":
        return cls(latent_dim=latent_dim, llm_dim=llm_dim, d_model=cfg.d_model, n_layers=cfg.n_layers,
                   n_heads=cfg.n_heads, ff_mult=cfg.ff_mult, dropout=cfg.dropout, ctc_upsample=cfg.ctc_upsample,
                   encoder=str(cfg.get("encoder", "mimi")), token_vocab_size=int(cfg.get("token_vocab_size", 0) or 0))

    @torch.no_grad()
    def init_scale(self, text_embedding_rms: float) -> None:
        self.out_scale.fill_(text_embedding_rms)
        unit = torch.randn_like(self.speech_start)
        self.speech_start.copy_(unit / unit.pow(2).mean().sqrt() * text_embedding_rms)
        unit = torch.randn_like(self.speech_end)
        self.speech_end.copy_(unit / unit.pow(2).mean().sqrt() * text_embedding_rms)

    def set_token_vocab(self, token_ids) -> None:
        ids = torch.as_tensor(sorted(set(int(i) for i in token_ids)), dtype=torch.long)
        if ids.numel() != self.token_vocab_size:
            raise ValueError(f"{ids.numel()} token ids for a head of size {self.token_vocab_size}")
        self.tok_vocab.copy_(ids.to(self.tok_vocab.device))

    def token_logits(self, embeds: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """embeds [B, T, llm_dim] (the vectors given to the thinker), table = the thinker's input embedding
        matrix -> [B, T, 1 + V] logits; index 0 is blank, index i >= 1 is thinker token tok_vocab[i - 1]."""
        words = F.normalize(table[self.tok_vocab].float(), dim=-1)
        allv = torch.cat([F.normalize(self.tok_blank.float(), dim=-1)[None], words], 0)
        return self.tok_scale * (F.normalize(embeds.float(), dim=-1) @ allv.T)

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
    def load(cls, path: str, map_location="cpu", llm_dim: int | None = None) -> "SpeechAdapter":
        """llm_dim: the thinker width to build for. When it differs from the checkpoint's (another thinker),
        every weight that still fits is loaded and the thinker-facing layers start fresh; their names
        are in `model.fresh_params` (empty for a normal load)."""
        from s2s.utils import load_matching

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp = dict(ckpt["hparams"])
        if llm_dim is not None and llm_dim != hp["llm_dim"]:
            hp["llm_dim"] = llm_dim
            model = cls(**hp)
            model.fresh_params = load_matching(model, ckpt["state_dict"])
            return model
        model = cls(**hp)
        model.load_state_dict(ckpt["state_dict"])
        model.fresh_params = []
        return model
