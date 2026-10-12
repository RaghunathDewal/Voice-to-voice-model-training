"""Live Nemotron listener vectors from a growing 16 kHz stream, with the encoder's own streaming caches
(used by the duplex demo and the listener bake-off)."""

from __future__ import annotations

import numpy as np
import torch

HOP = 1280  # 80 ms at 16 kHz


class StreamingListener:
    """Live Nemotron vectors from a growing 16 kHz stream, with the encoder's own streaming caches.

    Cache-aware streaming (as on the model card): the first chunk is 1 + 8 mel frames, every later chunk
    160 ms (2 frames: the current one and its one-frame look-ahead); the attention K/V cache and the
    convolution padding cache carry the left context, so the vectors are the ones the model was trained on
    (one masked pass over the whole call)."""

    def __init__(self, nemotron, lookahead: int = 1):
        self.enc, self.proc, self.device, self.dtype = nemotron.enc, nemotron.proc, nemotron.device, nemotron.dtype
        self.proc.set_num_lookahead_tokens(lookahead)
        self.look = lookahead
        fe = self.proc.feature_extractor
        self.hop, self.half_fft = int(fe.hop_length), int(fe.n_fft) // 2
        self.buf = np.zeros(0, np.float32)
        self.offset = 0          # samples dropped from the front of buf
        self.mel = 0             # mel frames consumed so far
        self.kv = self.pad = None
        self.done = 0            # 80 ms frames released

    def push(self, x16k: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, x16k.astype(np.float32)])

    def _next_chunk(self):
        total = self.offset + len(self.buf)
        if self.kv is None:
            n = self.proc.num_samples_first_audio_chunk
            if total < n:
                return None
            feats = self.proc(self.buf[:n], sampling_rate=16000, is_streaming=True, is_first_audio_chunk=True,
                              return_tensors="pt")["input_features"][:, : self.proc.num_mel_frames_first_audio_chunk]
            self.mel = self.proc.num_mel_frames_first_audio_chunk
            return feats
        start = self.mel * self.hop - self.half_fft
        end = start + self.proc.num_samples_per_audio_chunk
        if end > total:
            return None
        feats = self.proc(self.buf[start - self.offset: end - self.offset], sampling_rate=16000, is_streaming=True,
                          is_first_audio_chunk=False, return_tensors="pt")["input_features"]
        self.mel += self.proc.num_mel_frames_per_audio_chunk
        drop = max(0, self.mel * self.hop - self.half_fft - self.offset - 4096)  # keep a little before the next start
        if drop:
            self.buf, self.offset = self.buf[drop:], self.offset + drop
        return feats

    @torch.no_grad()
    def new_frames(self) -> list[torch.Tensor]:
        out = []
        while (feats := self._next_chunk()) is not None:
            o = self.enc(input_features=feats.to(self.device, self.dtype), past_key_values=self.kv,
                         padding_cache=self.pad, use_cache=True, num_lookahead_tokens=self.look,
                         output_attention_mask=False)
            self.kv, self.pad = o.past_key_values, o.padding_cache
            out.extend(o.last_hidden_state[0].float().cpu())
        self.done += len(out)
        return out

    def ready(self) -> int:
        """Frames that could be released now (for the lag estimate)."""
        return self.done + max(0, (self.offset + len(self.buf) - max(0, self.mel * self.hop - self.half_fft)) // HOP)
