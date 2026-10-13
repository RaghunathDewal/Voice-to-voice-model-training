"""Nemotron's own transcript from the listener features we already have (hybrid input, `hybrid.enabled`).

The v3 features are the Nemotron streaming encoder's output (last_hidden_state, 12.5 Hz). The model's text
decoder (an RNN-T: projector -> LSTM prediction network -> joint) sits on top of exactly those, so the transcript
can be made from stored features (training rows, test sets) or from the live turn, without the audio and without
running the encoder again. Greedy decoding, batched: each 80 ms frame emits up to `max_symbols_per_step` word
pieces; rows that emit "blank" keep their prediction-network state (the decoder cache masks them).
"""

from __future__ import annotations

import torch


class NemotronTranscriber:
    def __init__(self, projector, decoder, joint, cache_cls, config, decode_fn, device: torch.device):
        self.projector, self.decoder, self.joint = projector, decoder, joint
        self.config = config
        blank_id, max_symbols = config.blank_token_id, config.max_symbols_per_step
        self.cache_cls, self.blank, self.max_symbols = cache_cls, int(blank_id), int(max_symbols)
        self.decode_fn, self.device = decode_fn, device

    @classmethod
    def from_model(cls, model, device: torch.device, decode_fn=None) -> "NemotronTranscriber":
        """From a loaded NemotronAsrStreamingForRNNT (its encoder is not needed and can be dropped)."""
        import importlib

        mod = importlib.import_module(type(model).__module__)
        cache_cls = next(getattr(mod, n) for n in dir(mod) if n.endswith("RNNTDecoderCache"))
        parts = [m.to(device).eval() for m in (model.encoder_projector, model.decoder, model.joint)]
        return cls(*parts, cache_cls, model.config, decode_fn, device)

    @classmethod
    def load(cls, name: str, device: torch.device) -> "NemotronTranscriber":
        from transformers import AutoModelForRNNT, AutoProcessor

        proc = AutoProcessor.from_pretrained(name)
        model = AutoModelForRNNT.from_pretrained(name, dtype=torch.float32)
        model.encoder = None  # only the text decoder is kept (~25M parameters)
        return cls.from_model(model, device, lambda ids: proc.batch_decode(ids, skip_special_tokens=True))

    @torch.no_grad()
    def ids(self, feats: list[torch.Tensor]) -> list[list[int]]:
        """feats: list of [T_i, D] encoder outputs -> word-piece ids per row."""
        if not feats:
            return []
        lengths = torch.tensor([f.shape[0] for f in feats], device=self.device)
        x = torch.zeros(len(feats), int(lengths.max()), feats[0].shape[1], device=self.device)
        for i, f in enumerate(feats):
            x[i, : f.shape[0]] = torch.as_tensor(f, dtype=torch.float32, device=self.device)
        p = next(self.projector.parameters())
        enc = self.projector(x.to(p.dtype))  # [B, T, H]
        cache = self.cache_cls(self.config)
        blank = torch.full((len(feats), 1), self.blank, dtype=torch.long, device=self.device)
        dec = self.decoder(blank, cache=cache)  # [B, 1, H]
        out: list[list[int]] = [[] for _ in feats]
        for t in range(enc.shape[1]):
            active = t < lengths
            for _ in range(self.max_symbols):
                k = self.joint(decoder_hidden_states=dec[:, 0], encoder_hidden_states=enc[:, t]).argmax(-1)
                emit = active & (k != self.blank)
                if not bool(emit.any()):
                    break
                for i in emit.nonzero().flatten().tolist():
                    out[i].append(int(k[i]))
                dec = self.decoder(torch.where(emit, k, blank[:, 0])[:, None], cache=cache)
        return out

    def __call__(self, feats: list[torch.Tensor]) -> list[str]:
        ids = self.ids(feats)
        return [s.strip() for s in self.decode_fn(ids)] if self.decode_fn else [" ".join(map(str, x)) for x in ids]


def transcript_ids(prompts, text: str, template: str) -> list[int]:
    """The text tokens that carry the transcript into the 4B's input, right after the speech (inside the guest's
    turn, before it is closed). Same in training, evaluation and the live agent."""
    return prompts.ids(template.format(text=text.strip())) if text and text.strip() else []
