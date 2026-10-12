"""Full-duplex speech model (experiment): a small LLM that listens and speaks every 80 ms.

Sequence for one call:  [system prompt + tools] [frame 0] [frame 1] ... [frame f] [tool call + result] [frame f+1] ...

Frame t input (one position):
    listener vector of the guest at t (Nemotron, frozen, computed outside)
  + the agent's own Mimi codes of the previous grid column (what it just said)
  + the agent's previous text token (inner monologue; PAD when silent)
Frame t outputs, from the LLM state h_t:
    text token (LLM's own head; PAD = nothing to say this frame; <tool_call> = call a tool now)
    audio: a small depth transformer predicts the 8 Mimi codes of grid column t, conditioned on h_t and
    the text token. Moshi-style acoustic delay: column t holds codebook 1 of frame t and codebooks 2-8
    of frame t-1. Silence is predicted like any other sound, so "when to talk" is learned from data.
Tool calls are ordinary token positions between two frames: the model writes the call, the runtime
appends the result, and the frame stream continues.

Trained parts: LoRA on the LLM, the input fusion, the depth transformer and the code heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.modules.transformer import CausalTransformer, RMSNorm


def unwrap_llm(llm):
    m = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
    return m.model, m.lm_head, m.get_input_embeddings()


def delay_grid(codes: torch.Tensor, delay: int) -> torch.Tensor:
    """codes [K, T] -> grid [K, T]; codebooks 2..K shifted right by `delay`, -1 where empty."""
    grid = torch.full_like(codes, -1)
    grid[0] = codes[0]
    if delay:
        grid[1:, delay:] = codes[1:, :-delay]
    else:
        grid[1:] = codes[1:]
    return grid


def undelay(grid: torch.Tensor, delay: int) -> torch.Tensor:
    """grid [K, G] -> codes [K, G - delay] (frames whose every codebook is known)."""
    n = grid.shape[1] - delay
    out = grid[:, :n].clone()
    if delay:
        out[1:] = grid[1:, delay: delay + n]
    return out


class DuplexModel(nn.Module):
    def __init__(self, llm, feat_dim: int, pad_id: int, num_codebooks: int = 8, card: int = 2048,
                 depth_d_model: int = 512, depth_layers: int = 4, depth_heads: int = 8, delay: int = 1,
                 pad_weight: float = 0.5, semantic_weight: float = 2.0, embed_rms: float | None = None):
        super().__init__()
        self.llm = llm
        backbone, _, emb = unwrap_llm(llm)
        H = backbone.config.hidden_size
        self.hparams = dict(feat_dim=feat_dim, pad_id=pad_id, num_codebooks=num_codebooks, card=card,
                            depth_d_model=depth_d_model, depth_layers=depth_layers, depth_heads=depth_heads,
                            delay=delay, pad_weight=pad_weight, semantic_weight=semantic_weight)
        self.K, self.card, self.delay, self.pad_id = num_codebooks, card, delay, pad_id
        self.pad_weight, self.semantic_weight = pad_weight, semantic_weight
        if embed_rms is None:
            embed_rms = float(emb.weight.float().pow(2).mean().sqrt())
        # frame input fusion: listener vector + own previous codes, normalised to the text embedding scale
        self.feat_norm = RMSNorm(feat_dim)
        self.feat_proj = nn.Linear(feat_dim, H)
        self.code_emb = nn.ModuleList([nn.Embedding(card + 1, H) for _ in range(num_codebooks)])  # card = none
        self.in_norm = RMSNorm(H)
        self.in_scale = nn.Parameter(torch.tensor(float(embed_rms)))
        self.frame_type = nn.Parameter(torch.zeros(H))
        # depth transformer: h_t (+ the chosen text token) -> 8 codes
        self.depth_in = nn.Linear(H, depth_d_model)
        self.depth_text = nn.Linear(H, depth_d_model)
        self.depth_code_emb = nn.ModuleList([nn.Embedding(card, depth_d_model) for _ in range(num_codebooks - 1)])
        self.depth = CausalTransformer(depth_d_model, depth_layers, depth_heads, 4, 0.0)
        self.heads = nn.ModuleList([nn.Linear(depth_d_model, card) for _ in range(num_codebooks)])

    # ------------------------------------------------------------------ pieces
    def text_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return unwrap_llm(self.llm)[2](ids)

    def frame_inputs(self, guest: torch.Tensor, prev_text: torch.Tensor, prev_cols: torch.Tensor) -> torch.Tensor:
        """guest [T, F], prev_text [T], prev_cols [T, K] (card = none) -> [T, H]."""
        x = self.feat_proj(self.feat_norm(guest))
        for k in range(self.K):
            x = x + self.code_emb[k](prev_cols[:, k])
        x = self.in_norm(x) * self.in_scale
        return x + self.text_embed(prev_text).to(x.dtype) + self.frame_type

    def depth_logits(self, h: torch.Tensor, text_ids: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """Teacher-forced: h [N, H], text_ids [N], cols [N, K] (-1 allowed) -> [N, K, card]."""
        x0 = self.depth_in(h) + self.depth_text(self.text_embed(text_ids).to(h.dtype))
        c = cols.clamp(min=0)
        inputs = [x0] + [x0 + self.depth_code_emb[j - 1](c[:, j - 1]) for j in range(1, self.K)]
        out = self.depth(torch.stack(inputs, dim=1))
        return torch.stack([self.heads[j](out[:, j]) for j in range(self.K)], dim=1)

    @staticmethod
    def prev_columns(grid: torch.Tensor, card: int) -> torch.Tensor:
        """grid [K, T] -> inputs [T, K]: column t-1 for frame t (none for t = 0 and for empty cells)."""
        prev = torch.full_like(grid, card)
        prev[:, 1:] = grid[:, :-1]
        return prev.masked_fill(prev < 0, card).t()

    def assemble(self, item: dict, device) -> dict:
        """Build one call's input embeddings and the index of every frame / tool-segment position."""
        text = item["text"].to(device)
        grid = delay_grid(item["codes"].to(device), self.delay)
        prev_text = torch.cat([text.new_full((1,), self.pad_id), text[:-1]])
        seg_at = {f: (ids, n) for f, ids, n in item["segments"]}
        for f, (ids, _) in seg_at.items():  # the frame after a tool exchange sees its last token
            if f + 1 < len(prev_text):
                prev_text[f + 1] = ids[-1]
        frames = self.frame_inputs(item["guest"].to(device), prev_text, self.prev_columns(grid, self.card))
        prefix = self.text_embed(item["prefix"].to(device)).to(frames.dtype)
        pieces, frame_pos, seg_pos, seg_tgt = [prefix], [], [], []
        pos, start = prefix.shape[0], 0
        stops = sorted(seg_at)
        if not stops or stops[-1] != len(text) - 1:
            stops.append(len(text) - 1)
        for f in stops:
            pieces.append(frames[start: f + 1])
            frame_pos.extend(range(pos, pos + f + 1 - start))
            pos += f + 1 - start
            start = f + 1
            if f in seg_at:
                ids, n_pred = seg_at[f]
                ids_t = torch.tensor(ids, device=device)
                pieces.append(self.text_embed(ids_t).to(frames.dtype))
                tgt = torch.full((len(ids),), -100, device=device)
                tgt[:n_pred] = ids_t[1: n_pred + 1]
                seg_pos.extend(range(pos, pos + len(ids)))
                seg_tgt.append(tgt)
                pos += len(ids)
        x = torch.cat(pieces, dim=0)
        return {"x": x, "frame_pos": torch.tensor(frame_pos, device=device),
                "seg_pos": torch.tensor(seg_pos, device=device, dtype=torch.long),
                "seg_tgt": torch.cat(seg_tgt) if seg_tgt else torch.zeros(0, dtype=torch.long, device=device),
                "text": text, "grid": grid}

    # ------------------------------------------------------------------ training
    def forward(self, items: list[dict]) -> dict:
        device = self.in_scale.device
        backbone, lm_head, _ = unwrap_llm(self.llm)
        seqs = [self.assemble(it, device) for it in items]
        L = max(s["x"].shape[0] for s in seqs)
        H = seqs[0]["x"].shape[1]
        x = torch.zeros(len(seqs), L, H, device=device, dtype=seqs[0]["x"].dtype)
        mask = torch.zeros(len(seqs), L, device=device, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : s["x"].shape[0]] = s["x"]
            mask[i, : s["x"].shape[0]] = 1
        hidden = backbone(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state

        h_frames = torch.cat([hidden[i, s["frame_pos"]] for i, s in enumerate(seqs)])
        texts = torch.cat([s["text"] for s in seqs])
        cols = torch.cat([s["grid"].t() for s in seqs])
        # text: every frame (PAD down-weighted) + the tool calls
        logits = lm_head(h_frames).float()
        w = torch.where(texts == self.pad_id, self.pad_weight, 1.0)
        text_loss = (F.cross_entropy(logits, texts, reduction="none") * w).sum() / w.sum()
        h_seg = [hidden[i, s["seg_pos"]] for i, s in enumerate(seqs) if s["seg_pos"].numel()]
        seg_tgt = torch.cat([s["seg_tgt"] for s in seqs])
        tool_loss = F.cross_entropy(lm_head(torch.cat(h_seg)).float(), seg_tgt, ignore_index=-100) \
            if h_seg and (seg_tgt != -100).any() else logits.new_zeros(())
        # audio
        alog = self.depth_logits(h_frames, texts, cols).float()
        tgt = cols.masked_fill(cols < 0, -100)
        per_cb = torch.stack([F.cross_entropy(alog[:, j], tgt[:, j], ignore_index=-100) for j in range(self.K)])
        cw = torch.tensor([self.semantic_weight] + [1.0] * (self.K - 1), device=device)
        audio_loss = (per_cb * cw).sum() / cw.sum()
        loss = text_loss + audio_loss + (tool_loss if h_seg else 0.0)
        with torch.no_grad():
            pred = logits.argmax(-1)
            talk = texts != self.pad_id
            stats = {
                "text_acc": (pred[talk] == texts[talk]).float().mean() if talk.any() else logits.new_zeros(()),
                "pad_acc": (pred[~talk] == self.pad_id).float().mean() if (~talk).any() else logits.new_zeros(()),
                "c0_acc": (alog[:, 0].argmax(-1) == cols[:, 0]).float().mean(),
                "c_acc": ((alog.argmax(-1) == cols) | (cols < 0)).float().mean(),
            }
        return {"loss": loss, "text": text_loss.detach(), "audio": audio_loss.detach(),
                "tool": tool_loss.detach() if torch.is_tensor(tool_loss) else torch.zeros(()), **stats}

    # ------------------------------------------------------------------ io
    def trainable_state(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def save(self, path: str, extra: dict) -> None:
        torch.save({"hparams": self.hparams, "state_dict": self.trainable_state(), **extra}, path)
