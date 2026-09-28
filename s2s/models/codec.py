"""Frozen Mimi codec wrapper (Hugging Face `transformers` implementation).

* `encode_latents`: continuous encoder output *before* quantisation
  (12.5 Hz, `hidden_size`=512). This is the speech INPUT to the adapter.
* `encode_codes`: discrete RVQ codes [B, K, T] (targets for the talker).
* `decode`: codes -> 24 kHz waveform.

Mimi's encoder is causal, so right-padding a batch does not change the
latents of the real frames (checked in tests/test_codec.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch


class MimiCodec:
    def __init__(self, name_or_path: str, device: torch.device, num_codebooks: int = 8):
        from transformers import MimiModel

        self.model = MimiModel.from_pretrained(name_or_path).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        cfg = self.model.config
        self.sample_rate = int(cfg.sampling_rate)
        self.frame_rate = float(cfg.frame_rate)
        self.hop = int(round(self.sample_rate / self.frame_rate))
        self.latent_dim = int(cfg.hidden_size)
        self.codebook_size = int(cfg.codebook_size)
        if num_codebooks > cfg.num_quantizers:
            raise ValueError(f"num_codebooks={num_codebooks} > Mimi num_quantizers={cfg.num_quantizers}")
        self.num_codebooks = num_codebooks

    def num_frames(self, num_samples: int) -> int:
        return int(math.ceil(num_samples / self.hop))

    def _batch(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        lengths = [len(w) for w in wavs]
        # pad to a whole number of frames so every real frame is fully computed
        max_len = self.num_frames(max(lengths)) * self.hop
        x = torch.zeros(len(wavs), 1, max_len, dtype=torch.float32)
        for i, w in enumerate(wavs):
            x[i, 0, : len(w)] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        return x.to(self.device), lengths

    @torch.no_grad()
    def encode_latents(self, wavs: list[np.ndarray]) -> list[torch.Tensor]:
        """Returns a list of [T_i, latent_dim] float tensors (one per input)."""
        x, lengths = self._batch(wavs)
        m = self.model
        emb = m.encoder(x)
        out = m.encoder_transformer(emb.transpose(1, 2), return_dict=True)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        lat = m.downsample(hidden.transpose(1, 2))  # [B, D, T]
        lat = lat.transpose(1, 2).float()
        return [lat[i, : self.num_frames(n)] for i, n in enumerate(lengths)]

    @torch.no_grad()
    def encode_codes(self, wavs: list[np.ndarray]) -> list[torch.Tensor]:
        """Returns a list of [K, T_i] long tensors."""
        x, lengths = self._batch(wavs)
        out = self.model.encode(x, num_quantizers=self.num_codebooks, return_dict=True)
        codes = out.audio_codes
        return [codes[i, :, : self.num_frames(n)] for i, n in enumerate(lengths)]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """codes: [K, T] or [B, K, T] -> waveform(s) as numpy float32."""
        squeeze = codes.dim() == 2
        if squeeze:
            codes = codes[None]
        out = self.model.decode(codes.to(self.device).long(), return_dict=True)
        wav = out.audio_values[:, 0, : codes.shape[-1] * self.hop].float().cpu().numpy()
        return wav[0] if squeeze else wav


class StreamingDecoder:
    """Decodes a growing stream of Mimi frames chunk by chunk.

    The HF Mimi decoder has no convolution state cache, so each chunk is
    decoded together with `context_frames` of left context and only the new
    samples are returned. Mimi's decoder is causal, so the result matches a
    full decode exactly when the context covers the whole history, and closely
    otherwise (attention only sees `context_frames` of history).
    """

    def __init__(self, codec: MimiCodec, context_frames: int = 25):
        self.codec = codec
        self.context = context_frames
        self.frames: list[torch.Tensor] = []  # each [K]

    def push(self, new_frames: list[torch.Tensor]) -> np.ndarray:
        if not new_frames:
            return np.zeros(0, dtype=np.float32)
        self.frames.extend(f.detach().long().cpu() for f in new_frames)
        n_new = len(new_frames)
        start = max(0, len(self.frames) - n_new - self.context)
        codes = torch.stack(self.frames[start:], dim=1)  # [K, T]
        wav = self.codec.decode(codes)
        return wav[-n_new * self.codec.hop:]
