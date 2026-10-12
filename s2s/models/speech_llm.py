"""Glue between the speech adapter and the thinker: input assembly and losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from s2s.models.adapter import SpeechAdapter
from s2s.models.thinker import Thinker


def assemble_inputs(thinker: Thinker, adapter: SpeechAdapter, speech: torch.Tensor, lengths: torch.Tensor,
                    prefix: list[list[int]], suffix: list[list[int]], target: list[list[int]] | None):
    """speech: [B, T, H] adapter embeddings. Returns inputs_embeds, attention_mask, labels (or None)."""
    device = thinker.device
    dtype = thinker.dtype
    start, end = adapter.boundary_embeddings()
    seqs, labels = [], []
    for i in range(speech.shape[0]):
        tgt = target[i] if target is not None else []
        pre = thinker.embed(torch.tensor(prefix[i], device=device))
        post = thinker.embed(torch.tensor(list(suffix[i]) + list(tgt), device=device))
        sp = speech[i, : int(lengths[i])]
        seq = torch.cat([pre.to(dtype), start[None].to(dtype), sp.to(dtype), end[None].to(dtype), post.to(dtype)], 0)
        seqs.append(seq)
        n_ctx = seq.shape[0] - len(tgt)
        lab = torch.full((seq.shape[0],), -100, dtype=torch.long, device=device)
        if tgt:
            lab[n_ctx:] = torch.tensor(tgt, device=device)
        labels.append(lab)
    max_len = max(s.shape[0] for s in seqs)
    b = len(seqs)
    embeds = torch.zeros(b, max_len, seqs[0].shape[-1], dtype=dtype, device=device)
    mask = torch.zeros(b, max_len, dtype=torch.long, device=device)
    lab_out = torch.full((b, max_len), -100, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        embeds[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = 1
        lab_out[i, : s.shape[0]] = labels[i]
    return embeds, mask, (lab_out if target is not None else None)


def _causal_lm(thinker: Thinker):
    return thinker.model.get_base_model() if hasattr(thinker.model, "get_base_model") else thinker.model


def target_logits(thinker: Thinker, embeds: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor):
    """LM logits only where there is a label: (logits [N, V], targets [N], row index [N]), rows in order.

    Running the LM head on every position (prompt + speech + target) would
    materialise [B, S, 151k] logits; for ~1000-token sequences that alone is
    several GB. Only the target positions are needed.
    """
    causal_lm = _causal_lm(thinker)
    hidden = causal_lm.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
    shifted = labels[:, 1:]
    keep = shifted != -100
    rows = torch.nonzero(keep, as_tuple=True)[0]
    return causal_lm.lm_head(hidden[:, :-1][keep]), shifted[keep], rows


def target_lm_loss(thinker: Thinker, embeds: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross entropy computed only where there is a label."""
    logits, targets, _ = target_logits(thinker, embeds, mask, labels)
    return F.cross_entropy(logits.float(), targets)


@torch.no_grad()
def teacher_logits(thinker: Thinker, teacher: list[list[int]], target: list[list[int]]) -> torch.Tensor:
    """The thinker reading the TEXT version of the same turn (prompt ids ending with the assistant header),
    teacher-forced on the same target: logits [N, V] for the target tokens of these rows, rows in order."""
    device = thinker.device
    seqs = [list(p) + list(t) for p, t in zip(teacher, target)]
    n = max(len(x) for x in seqs)
    ids = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
    labels = torch.full((len(seqs), n), -100, dtype=torch.long, device=device)
    for i, (x, t) in enumerate(zip(seqs, target)):
        ids[i, : len(x)] = torch.tensor(x, device=device)
        mask[i, : len(x)] = 1
        if t:
            labels[i, len(x) - len(t): len(x)] = torch.tensor(t, device=device)
    logits, _, _ = target_logits(thinker, thinker.embed(ids).to(thinker.dtype), mask, labels)
    return logits


def kd_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """KL(teacher || student) per target token, averaged (BLSP-KD: the thinker's next-token distribution
    from speech should match its distribution from the text of the same turn)."""
    t = temperature
    log_s = F.log_softmax(student.float() / t, dim=-1)
    log_t = F.log_softmax(teacher.float() / t, dim=-1)
    return (log_t.exp() * (log_t - log_s)).sum(-1).mean() * t * t


def token_ctc_loss(adapter: SpeechAdapter, embeds: torch.Tensor, lengths: torch.Tensor,
                   tok_targets: list[list[int]], table: torch.Tensor) -> torch.Tensor:
    """CTC over thinker tokens with the head tied to the thinker's embedding table (see SpeechAdapter).
    tok_targets are indices into the head (1 + position in tok_vocab); empty = only blanks (unclear audio)."""
    return token_ctc_from_logits(adapter.token_logits(embeds, table), lengths, tok_targets)


def token_ctc_from_logits(logits: torch.Tensor, lengths: torch.Tensor, tok_targets: list[list[int]]) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # [T, B, 1+V]
    flat = torch.tensor([i for t in tok_targets for i in t], dtype=torch.long, device=logits.device)
    tlen = torch.tensor([len(t) for t in tok_targets], dtype=torch.long, device=logits.device)
    return F.ctc_loss(logp, flat, lengths.to(logits.device), tlen, blank=0, zero_infinity=True)


def speech_llm_losses(thinker: Thinker, adapter: SpeechAdapter, batch: dict, ctc_weight: float,
                      eot_weight: float, kd_weight: float = 0.0, tok_weight: float = 0.0,
                      ce_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """lm (cross entropy on the target), ctc (characters), eot, and when enabled:
    kd  (KL to the thinker reading the text of the same turn, rows that have `teacher` ids)
    tok (word-level CTC tied to the thinker's embeddings, adapters with a token head)."""
    device = thinker.device
    latents = batch["latents"].to(device)
    lengths = batch["lengths"].to(device)
    out = adapter(latents)
    embeds, mask, labels = assemble_inputs(thinker, adapter, out["embeds"], lengths,
                                           batch["prefix"], batch["suffix"], batch["target"])
    logits, targets, rows = target_logits(thinker, embeds, mask, labels)
    lm_loss = F.cross_entropy(logits.float(), targets)
    zero = lm_loss.new_zeros(())

    kd = zero
    teacher = batch.get("teacher") or [None] * len(batch["target"])
    kd_rows = [i for i, t in enumerate(teacher) if t is not None and batch["target"][i]]
    if kd_weight and kd_rows:
        t_logits = teacher_logits(thinker, [teacher[i] for i in kd_rows], [batch["target"][i] for i in kd_rows])
        sel = torch.isin(rows, torch.tensor(kd_rows, device=rows.device))
        kd = kd_loss(logits[sel], t_logits)

    tok = zero
    if tok_weight and getattr(adapter, "token_vocab_size", 0) and batch.get("tok") is not None:
        tok = token_ctc_loss(adapter, out["embeds"], lengths, batch["tok"], thinker.embed_tokens().weight)

    logp = F.log_softmax(out["ctc_logits"].float(), dim=-1).transpose(0, 1)  # [T', B, V]
    ctc_loss = F.ctc_loss(logp, batch["ctc_targets"].to(device), lengths * adapter.ctc_upsample,
                          batch["ctc_lengths"].to(device), blank=0, zero_infinity=True)

    eot = batch["eot"].to(device)
    valid = eot >= 0
    if valid.any():
        eot_loss = F.binary_cross_entropy_with_logits(out["eot_logits"].float()[valid], eot[valid].float())
    else:
        eot_loss = zero
    total = ce_weight * lm_loss + ctc_weight * ctc_loss + eot_weight * eot_loss + kd_weight * kd + tok_weight * tok
    return {"loss": total, "lm": lm_loss.detach(), "ctc": ctc_loss.detach(), "eot": eot_loss.detach(),
            "kd": kd.detach(), "tok": tok.detach(),
            "ctc_logits": out["ctc_logits"].detach(), "eot_logits": out["eot_logits"].detach()}


@torch.no_grad()
def greedy_generate(thinker: Thinker, embeds: torch.Tensor, max_new_tokens: int) -> list[int]:
    """Plain greedy decoding from a single [1, S, H] prompt embedding (used in evaluation)."""
    model = thinker.model
    out = model(inputs_embeds=embeds, use_cache=True)
    past = out.past_key_values
    tokens: list[int] = []
    next_id = int(out.logits[0, -1].argmax())
    for _ in range(max_new_tokens):
        if next_id in thinker.eos_ids:
            break
        tokens.append(next_id)
        emb = thinker.embed(torch.tensor([[next_id]], device=thinker.device))
        out = model(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = int(out.logits[0, -1].argmax())
    return tokens


@torch.no_grad()
def talker_features(thinker: Thinker, prefix_ids: list[int], responses: list[list[int]],
                    layer_idx: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Teacher-forced thinker features for the talker.

    For response token i the paired hidden state is the one at the position
    that *predicted* it (prefix_len - 1 + i), exactly as in streaming inference.
    Returns per-sample token embeddings [L, H] and hidden states [L, n_layers, H].
    """
    device = thinker.device
    p = len(prefix_ids)
    seqs = [prefix_ids + r for r in responses]
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), thinker.pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=device)
        mask[i, : len(s)] = 1
    causal_lm = thinker.model.get_base_model() if hasattr(thinker.model, "get_base_model") else thinker.model
    out = causal_lm.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[j] for j in layer_idx], dim=2)  # [B, S, n_sel, H]
    embeds = thinker.embed(ids)
    tok_embs, hiddens = [], []
    for i, r in enumerate(responses):
        n = len(r)
        tok_embs.append(embeds[i, p: p + n])
        hiddens.append(hs[i, p - 1: p - 1 + n])
    return tok_embs, hiddens
