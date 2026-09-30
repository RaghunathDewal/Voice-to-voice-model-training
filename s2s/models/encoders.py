"""Frozen speech encoders behind one interface, for the encoder bake-off.

Every encoder returns features at 12.5 Hz (one frame per 80 ms, like Mimi), so
the same adapter / CTC probe / end-of-turn logic works on all of them. Encoders
that run at 50 Hz are brought to 12.5 Hz by stacking 4 consecutive frames.

    enc = build_encoder("whisper:openai/whisper-small", device)
    feats = enc.encode([wav_16k_a, wav_16k_b])      # list of [T, enc.dim] float32 tensors

Specs:
    mimi[:kyutai/mimi]                               causal (streams), 512-d
    whisper:<hf id>                                  whole utterance (padded to 30 s), 4 x d_model
    parakeet:<hf id>        e.g. nvidia/parakeet-ctc-0.6b      FastConformer, 12.5 Hz, whole utterance
    moonshine:<hf id>       e.g. moonshine-ai/moonshine-streaming-small   sliding window, streams, 4 x d
"""

from __future__ import annotations

import math

import numpy as np
import torch

FRAME_RATE = 12.5


def n_frames(n_samples: int, sample_rate: int) -> int:
    return max(1, math.ceil(n_samples * FRAME_RATE / sample_rate))


def stack_to_12_5hz(h: torch.Tensor, t: int, factor: int) -> torch.Tensor:
    """[T50, D] -> [t, factor * D]; pads with the last frame if the encoder output is short."""
    need = t * factor
    if h.shape[0] < need:
        h = torch.cat([h, h[-1:].expand(need - h.shape[0], -1)], dim=0)
    return h[:need].reshape(t, factor * h.shape[1])


class SpeechEncoder:
    name: str = ""
    sample_rate: int = 16000
    dim: int = 0
    streaming: bool = False
    params_m: float = 0.0

    def encode(self, wavs: list[np.ndarray]) -> list[torch.Tensor]:
        raise NotImplementedError


class MimiEncoder(SpeechEncoder):
    streaming = True

    def __init__(self, model: str, device: torch.device):
        from s2s.models.codec import MimiCodec

        self.codec = MimiCodec(model, device, 8)
        self.name, self.sample_rate, self.dim = f"mimi:{model}", self.codec.sample_rate, 512
        self.params_m = sum(p.numel() for p in self.codec.model.encoder.parameters()) / 1e6 + \
            sum(p.numel() for p in self.codec.model.encoder_transformer.parameters()) / 1e6

    @torch.no_grad()
    def encode(self, wavs):
        return [x.float().cpu() for x in self.codec.encode_latents(wavs)]


class WhisperEncoder(SpeechEncoder):
    def __init__(self, model: str, device: torch.device, dtype: torch.dtype):
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.fe = WhisperFeatureExtractor.from_pretrained(model)
        full = WhisperModel.from_pretrained(model, dtype=dtype)
        self.enc = full.encoder.to(device).eval()
        del full
        self.device, self.dtype = device, dtype
        self.name, self.dim = f"whisper:{model}", 4 * self.enc.config.d_model
        self.params_m = sum(p.numel() for p in self.enc.parameters()) / 1e6

    @torch.no_grad()
    def encode(self, wavs):
        wavs = [w[: 30 * 16000] for w in wavs]  # Whisper's window is 30 s
        feats = self.fe(wavs, sampling_rate=16000, return_tensors="pt").input_features
        h = self.enc(feats.to(self.device, self.dtype)).last_hidden_state.float()
        return [stack_to_12_5hz(h[i], min(n_frames(len(w), 16000), 375), 4).cpu() for i, w in enumerate(wavs)]


class ParakeetEncoder(SpeechEncoder):
    def __init__(self, model: str, device: torch.device, dtype: torch.dtype):
        from transformers import AutoFeatureExtractor, ParakeetForCTC

        self.fe = AutoFeatureExtractor.from_pretrained(model)
        full = ParakeetForCTC.from_pretrained(model, dtype=dtype)
        self.enc = full.encoder.to(device).eval()
        del full
        self.device, self.dtype = device, dtype
        self.name = f"parakeet:{model}"
        self.dim = int(self.enc.config.hidden_size)
        self.params_m = sum(p.numel() for p in self.enc.parameters()) / 1e6

    @torch.no_grad()
    def encode(self, wavs):
        inp = self.fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        h = self.enc(input_features=inp["input_features"].to(self.device, self.dtype),
                     attention_mask=inp["attention_mask"].to(self.device)).last_hidden_state.float()
        out = []
        for i, w in enumerate(wavs):
            t = n_frames(len(w), 16000)
            x = h[i, :t]
            if x.shape[0] < t:
                x = torch.cat([x, x[-1:].expand(t - x.shape[0], -1)], dim=0)
            out.append(x.cpu())
        return out


class MoonshineEncoder(SpeechEncoder):
    streaming = True

    def __init__(self, model: str, device: torch.device, dtype: torch.dtype):
        from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

        self.proc = AutoProcessor.from_pretrained(model)
        full = MoonshineStreamingForConditionalGeneration.from_pretrained(model, dtype=dtype)
        self.enc = full.get_encoder().to(device).eval()
        del full
        self.device, self.dtype = device, dtype
        self.name = f"moonshine:{model}"
        self.dim = 4 * int(self.enc.config.hidden_size)
        self.params_m = sum(p.numel() for p in self.enc.parameters()) / 1e6

    @torch.no_grad()
    def encode(self, wavs):
        inp = self.proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        h = self.enc(input_values=inp["input_values"].to(self.device, self.dtype),
                     attention_mask=inp["attention_mask"].to(self.device)).last_hidden_state.float()
        return [stack_to_12_5hz(h[i], n_frames(len(w), 16000), 4).cpu() for i, w in enumerate(wavs)]


def build_encoder(spec: str, device: torch.device) -> SpeechEncoder:
    kind, _, model = spec.partition(":")
    half = torch.float16 if device.type == "cuda" else torch.float32
    if kind == "mimi":
        return MimiEncoder(model or "kyutai/mimi", device)
    if kind == "whisper":
        return WhisperEncoder(model or "openai/whisper-small", device, half)
    if kind == "parakeet":  # fp32: FastConformer is not validated in fp16, and T4 has no fast bf16
        return ParakeetEncoder(model or "nvidia/parakeet-ctc-0.6b", device, torch.float32)
    if kind == "moonshine":
        return MoonshineEncoder(model or "moonshine-ai/moonshine-streaming-small", device, half)
    raise ValueError(f"unknown encoder spec {spec!r}")
