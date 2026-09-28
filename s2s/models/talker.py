"""Talker: thinker text features -> Mimi codes, streamed.

Input per spoken text token i (from the thinker):
    token embedding e(t_i)  and  hidden states of the position that *produced* t_i
    (selected layers, learned softmax weighting)  -> fusion -> text item [d]

Temporal talker: a causal transformer over an interleaved sequence

    [first_text_chunk text items][audio_chunk audio slots][text_chunk text][audio_chunk audio]...

After the last text token an END_OF_TEXT item is inserted. Audio slot g takes
the embedding of grid frame g-1 (BOS for g=0) and its output state is used to
predict grid frame g.

Codebook grid with acoustic delay d (Moshi-style): codebook 1 (semantic) of
frame f sits at grid column f, codebooks 2..K of frame f sit at column f+d.
Codebook 1 emits EOA (end of audio) at column F. PAD fills unused cells and is
never a training target.

Depth transformer: for one grid column, predicts codebooks 1..K in order
(position 0 input = projected temporal state, position j input adds the
embedding of the code at j-1).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.modules.transformer import CausalTransformer, RMSNorm


# ----------------------------------------------------------------- grid utils
def apply_delay(codes: torch.Tensor, delay: int, card: int) -> torch.Tensor:
    """codes [K, F] -> grid [K, F + max(delay, 1)] with EOA / PAD tokens."""
    k, f = codes.shape
    eoa, pad = card, card + 1
    g = f + max(delay, 1)
    grid = torch.full((k, g), pad, dtype=torch.long, device=codes.device)
    grid[0, :f] = codes[0]
    grid[0, f] = eoa
    if k > 1:
        grid[1:, delay: delay + f] = codes[1:]
    return grid


def undo_delay(grid: torch.Tensor, delay: int, n_frames: int) -> torch.Tensor:
    k = grid.shape[0]
    out = torch.empty((k, n_frames), dtype=torch.long, device=grid.device)
    out[0] = grid[0, :n_frames]
    if k > 1:
        out[1:] = grid[1:, delay: delay + n_frames]
    return out


def build_schedule(n_text: int, n_audio: int, first: int, text_chunk: int, audio_chunk: int) -> list[tuple[str, int]]:
    """Interleaving order used identically in training and streaming inference."""
    order: list[tuple[str, int]] = []
    t = a = c = 0
    while a < n_audio:
        t_end = min(n_text, first + c * text_chunk)
        while t < t_end:
            order.append(("t", t))
            t += 1
        a_end = min(n_audio, a + audio_chunk)
        while a < a_end:
            order.append(("a", a))
            a += 1
        c += 1
    return order


# ---------------------------------------------------------------------- model
class Talker(nn.Module):
    def __init__(self, llm_dim: int, n_hidden_layers: int, num_codebooks: int = 8, card: int = 2048,
                 d_model: int = 1024, n_layers: int = 12, n_heads: int = 16, ff_mult: int = 4,
                 dropout: float = 0.1, depth_d_model: int = 512, depth_layers: int = 6, depth_heads: int = 8,
                 first_text_chunk: int = 8, text_chunk: int = 4, audio_chunk: int = 10,
                 acoustic_delay: int = 1, semantic_loss_weight: float = 1.0, max_frames: int = 500):
        super().__init__()
        self.hparams = dict(
            llm_dim=llm_dim, n_hidden_layers=n_hidden_layers, num_codebooks=num_codebooks, card=card,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
            depth_d_model=depth_d_model, depth_layers=depth_layers, depth_heads=depth_heads,
            first_text_chunk=first_text_chunk, text_chunk=text_chunk, audio_chunk=audio_chunk,
            acoustic_delay=acoustic_delay, semantic_loss_weight=semantic_loss_weight, max_frames=max_frames,
        )
        self.max_frames = max_frames
        self.K = num_codebooks
        self.card = card
        self.EOA, self.PAD = card, card + 1
        self.delay = acoustic_delay
        self.first_text_chunk, self.text_chunk, self.audio_chunk = first_text_chunk, text_chunk, audio_chunk
        self.semantic_loss_weight = semantic_loss_weight

        # fusion of thinker features
        self.layer_logits = nn.Parameter(torch.zeros(n_hidden_layers))
        self.tok_norm = RMSNorm(llm_dim)
        self.tok_proj = nn.Linear(llm_dim, d_model)
        self.hid_norm = RMSNorm(llm_dim)
        self.hid_proj = nn.Linear(llm_dim, d_model)
        self.text_type = nn.Parameter(torch.randn(d_model) * 0.02)
        self.end_of_text = nn.Parameter(torch.randn(d_model) * 0.02)

        # temporal transformer
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)
        self.audio_type = nn.Parameter(torch.randn(d_model) * 0.02)
        self.code_emb = nn.ModuleList([nn.Embedding(card + 2, d_model) for _ in range(num_codebooks)])
        self.temporal = CausalTransformer(d_model, n_layers, n_heads, ff_mult, dropout)

        # depth transformer
        self.depth_in = nn.Linear(d_model, depth_d_model)
        self.depth_code_emb = nn.ModuleList([nn.Embedding(card + 2, depth_d_model) for _ in range(num_codebooks - 1)])
        self.depth = CausalTransformer(depth_d_model, depth_layers, depth_heads, ff_mult, dropout)
        self.heads = nn.ModuleList([nn.Linear(depth_d_model, card + 1) for _ in range(num_codebooks)])

    @classmethod
    def from_config(cls, cfg, llm_dim: int, n_hidden_layers: int, num_codebooks: int, card: int) -> "Talker":
        return cls(llm_dim=llm_dim, n_hidden_layers=n_hidden_layers, num_codebooks=num_codebooks, card=card,
                   d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, ff_mult=cfg.ff_mult,
                   dropout=cfg.dropout, depth_d_model=cfg.depth_d_model, depth_layers=cfg.depth_layers,
                   depth_heads=cfg.depth_heads, first_text_chunk=cfg.first_text_chunk,
                   text_chunk=cfg.text_chunk, audio_chunk=cfg.audio_chunk,
                   acoustic_delay=cfg.acoustic_delay, semantic_loss_weight=cfg.semantic_loss_weight,
                   max_frames=cfg.max_frames)

    # ---------------------------------------------------------------- pieces
    def fuse(self, tok_emb: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """tok_emb [N, H], hidden [N, L_sel, H] -> text items [N, d]."""
        w = torch.softmax(self.layer_logits.float(), dim=0).to(hidden.dtype)
        h = (hidden * w[None, :, None]).sum(1)
        return self.tok_proj(self.tok_norm(tok_emb)) + self.hid_proj(self.hid_norm(h)) + self.text_type

    def frame_embedding(self, col: torch.Tensor) -> torch.Tensor:
        """col [..., K] codes of one grid column -> [..., d]."""
        return sum(self.code_emb[k](col[..., k]) for k in range(self.K)) + self.audio_type

    def audio_inputs(self, grid: torch.Tensor) -> torch.Tensor:
        """grid [K, G] -> temporal inputs for audio slots [G, d] (slot g sees column g-1)."""
        prev = self.frame_embedding(grid[:, :-1].t())
        return torch.cat([(self.bos + self.audio_type)[None], prev], dim=0)

    def depth_logits(self, h: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """Teacher-forced depth pass. h [N, d], cols [N, K] -> logits [N, K, card+1]."""
        x0 = self.depth_in(h)
        inputs = [x0] + [x0 + self.depth_code_emb[j - 1](cols[:, j - 1]) for j in range(1, self.K)]
        out = self.depth(torch.stack(inputs, dim=1))
        return torch.stack([self.heads[j](out[:, j]) for j in range(self.K)], dim=1)

    # --------------------------------------------------------------- training
    def forward(self, tok_embs: list[torch.Tensor], hiddens: list[torch.Tensor], grids: list[torch.Tensor],
                return_logits: bool = False) -> dict:
        device = self.bos.device
        seqs, audio_positions = [], []
        for tok, hid, grid in zip(tok_embs, hiddens, grids):
            text_items = torch.cat([self.fuse(tok, hid), self.end_of_text[None]], dim=0)
            audio_in = self.audio_inputs(grid)
            order = build_schedule(text_items.shape[0], grid.shape[1], self.first_text_chunk,
                                   self.text_chunk, self.audio_chunk)
            n_text = text_items.shape[0]
            index = torch.tensor([i if kind == "t" else n_text + i for kind, i in order], device=device)
            seqs.append(torch.cat([text_items, audio_in], dim=0)[index])
            audio_positions.append(torch.tensor([p for p, (kind, _) in enumerate(order) if kind == "a"], device=device))
        max_len = max(s.shape[0] for s in seqs)
        batch = torch.zeros(len(seqs), max_len, seqs[0].shape[-1], device=device, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            batch[i, : s.shape[0]] = s
        states = self.temporal(batch)

        h = torch.cat([states[i, pos] for i, pos in enumerate(audio_positions)], dim=0)
        cols = torch.cat([g.t() for g in grids], dim=0)  # [N, K]
        logits = self.depth_logits(h, cols)
        targets = cols.masked_fill(cols == self.PAD, -100)
        losses, accs = [], []
        for j in range(self.K):
            lj = F.cross_entropy(logits[:, j].float(), targets[:, j], ignore_index=-100)
            losses.append(lj)
            valid = targets[:, j] != -100
            acc = (logits[:, j].argmax(-1)[valid] == targets[:, j][valid]).float().mean() if valid.any() else lj.new_zeros(())
            accs.append(acc.detach())
        w = torch.tensor([self.semantic_loss_weight] + [1.0] * (self.K - 1), device=device)
        loss = (torch.stack(losses) * w).sum() / w.sum()
        out = {"loss": loss, "loss_per_codebook": torch.stack(losses).detach(), "acc_per_codebook": torch.stack(accs)}
        if return_logits:
            out["logits"] = logits
        return out

    def save(self, path: str) -> None:
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "Talker":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator | None = None) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits.float() / temperature
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


class TalkerStream:
    """Incremental generation. Push text items as the thinker produces them;
    call `step()` while `can_step()`; completed Mimi frames come back as [K] tensors.

    `pick(j, logits, allowed_mask) -> code` can override sampling (used for
    teacher-forced consistency tests).
    """

    def __init__(self, talker: Talker, temperature: float = 0.8, top_k: int = 250, max_frames: int | None = None,
                 pick: Callable[[int, torch.Tensor, torch.Tensor], int] | None = None,
                 generator: torch.Generator | None = None):
        self.m = talker
        self.temperature, self.top_k = temperature, top_k
        self.max_frames = max_frames if max_frames is not None else talker.max_frames
        self.pick = pick
        self.generator = generator
        self.cache = talker.temporal.new_cache()
        self.pos = 0
        self.items: list[torch.Tensor] = []
        self.fed = 0
        self.text_done = False
        self.chunk = 0
        self.slot_in_chunk = 0
        self.cols: list[torch.Tensor] = []
        self.eoa_at: int | None = None
        self.finished = False
        self.emitted = 0

    # -------------------------------------------------------------- text in
    def push_text(self, items: torch.Tensor) -> None:
        for x in items:
            self.items.append(x)

    def end_text(self) -> None:
        if not self.text_done:
            self.items.append(self.m.end_of_text)
            self.text_done = True

    def _bound(self) -> int:
        return self.m.first_text_chunk + self.chunk * self.m.text_chunk

    def can_step(self) -> bool:
        if self.finished:
            return False
        if self.slot_in_chunk > 0 or self.text_done:
            return True
        return len(self.items) >= self._bound()

    # ------------------------------------------------------------ generation
    def _run_temporal(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m.temporal(x[None], cache=self.cache, start_pos=self.pos)
        self.pos += x.shape[0]
        return out[0]

    @torch.no_grad()
    def step(self) -> list[torch.Tensor]:
        """Generate one grid column. Returns newly completed frames (possibly empty)."""
        if not self.can_step():
            return []
        m = self.m
        if self.slot_in_chunk == 0:
            end = min(self._bound(), len(self.items))
            if end > self.fed:
                self._run_temporal(torch.stack(self.items[self.fed:end]))
                self.fed = end
        g = len(self.cols)
        if g == 0:
            x = (m.bos + m.audio_type)[None]
        else:
            x = m.frame_embedding(self.cols[-1])[None]
        h = self._run_temporal(x)[-1]

        col = self._depth_sample(h, g)
        self.cols.append(col)
        if self.eoa_at is None and int(col[0]) == m.EOA:
            self.eoa_at = g
        self.slot_in_chunk += 1
        if self.slot_in_chunk == m.audio_chunk:
            self.chunk += 1
            self.slot_in_chunk = 0
        if self.eoa_at is not None and g >= self.eoa_at + max(m.delay, 1) - 1:
            self.finished = True

        new_frames = []
        while True:
            f = self.emitted
            if f + m.delay >= len(self.cols):
                break
            if self.eoa_at is not None and f >= self.eoa_at:
                break
            frame = torch.empty(m.K, dtype=torch.long, device=col.device)
            frame[0] = self.cols[f][0]
            if m.K > 1:
                frame[1:] = self.cols[f + m.delay][1:]
            new_frames.append(frame)
            self.emitted += 1
        return new_frames

    def _depth_sample(self, h: torch.Tensor, g: int) -> torch.Tensor:
        m = self.m
        x0 = m.depth_in(h[None])  # [1, dd]
        inputs = [x0]
        codes: list[int] = []
        n_frames = self.eoa_at
        for j in range(m.K):
            out = m.depth(torch.stack(inputs, dim=1))[:, -1]
            logits = m.heads[j](out)[0]
            allowed = torch.ones(m.card + 1, dtype=torch.bool, device=logits.device)
            if j == 0:
                if self.eoa_at is not None:
                    code = m.PAD
                elif g >= self.max_frames:
                    code = m.EOA
                else:
                    if g == 0:
                        allowed[m.EOA] = False
                    code = self._choose(j, logits, allowed)
                    if code == m.EOA:
                        n_frames = g
            else:
                real = g >= m.delay and (n_frames is None or g < n_frames + m.delay)
                if real:
                    allowed[m.EOA] = False
                    code = self._choose(j, logits, allowed)
                else:
                    code = m.PAD
            codes.append(code)
            if j + 1 < m.K:
                inputs.append(x0 + m.depth_code_emb[j](torch.tensor([code], device=h.device)))
        return torch.tensor(codes, dtype=torch.long, device=h.device)

    def _choose(self, j: int, logits: torch.Tensor, allowed: torch.Tensor) -> int:
        if self.pick is not None:
            return int(self.pick(j, logits, allowed))
        masked = logits.masked_fill(~allowed, float("-inf"))
        return sample_logits(masked, self.temperature, self.top_k, self.generator)
