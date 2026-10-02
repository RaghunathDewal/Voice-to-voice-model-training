"""Fast unit tests that need no downloads."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from s2s.config import load_config, save_config
from s2s.data.hotel import HotelBackend, calls_match, generate_examples, make_reservation, validate_call
from s2s.modules.transformer import CausalTransformer
from s2s.models.adapter import SpeechAdapter
from s2s.models.talker import Talker, TalkerStream, apply_delay, build_schedule, undo_delay
from s2s.runtime.endpoint import Endpointer
from s2s.text import ctc_encode, ctc_greedy_decode, normalize_for_ctc, wer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_inheritance_and_overrides(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "small.yaml"),
                      ["train_talker.max_steps=7", "thinker.lora.r=4", "train_speech_llm.train_manifests=[{path: a, weight: 2}]"])
    assert cfg.thinker.model == "Qwen/Qwen3-0.6B"          # from small.yaml
    assert cfg.codec.model == "kyutai/mimi"                 # inherited from default.yaml
    assert cfg.train_talker.max_steps == 7 and cfg.thinker.lora.r == 4
    assert cfg.train_speech_llm.train_manifests[0]["weight"] == 2
    save_config(cfg, str(tmp_path / "c.yaml"))
    again = load_config(str(tmp_path / "c.yaml"))
    assert again.to_dict() == cfg.to_dict()


def test_ctc_text_roundtrip():
    text = "Hello, World! It's 5 o'clock"
    ids = ctc_encode(text)
    alignment = [x for i in ids for x in (i, i, 0)]  # a CTC path: repeats + blanks
    assert ctc_greedy_decode(alignment) == normalize_for_ctc(text) == "hello world it's o'clock"
    assert wer(["a b c"], ["a b c"]) == 0.0


def test_transformer_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = CausalTransformer(32, 2, 4, dropout=0.0).eval()
    x = torch.randn(1, 11, 32)
    full = m(x)
    cache = m.new_cache()
    outs, pos = [], 0
    for n in (3, 1, 4, 1, 2):  # mixed chunk sizes
        outs.append(m(x[:, pos:pos + n], cache=cache, start_pos=pos))
        pos += n
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-5)


def test_adapter_is_causal():
    torch.manual_seed(0)
    a = SpeechAdapter(16, 24, d_model=32, n_layers=2, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 20, 16)
    short, full = a(x[:, :12]), a(x)
    assert torch.allclose(short["embeds"], full["embeds"][:, :12], atol=1e-5)
    assert torch.allclose(short["eot_logits"], full["eot_logits"][:, :12], atol=1e-5)
    assert full["ctc_logits"].shape[1] == 20 * a.ctc_upsample


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_delay_roundtrip(delay):
    codes = torch.randint(0, 50, (8, 17))
    grid = apply_delay(codes, delay, 50)
    assert grid.shape[1] == 17 + max(delay, 1)
    assert int(grid[0, 17]) == 50  # EOA
    assert torch.equal(undo_delay(grid, delay, 17), codes)


def test_schedule_covers_everything():
    order = build_schedule(n_text=9, n_audio=23, first=4, text_chunk=2, audio_chunk=5)
    assert [i for k, i in order if k == "a"] == list(range(23))
    assert [i for k, i in order if k == "t"] == list(range(9))
    assert order[:4] == [("t", 0), ("t", 1), ("t", 2), ("t", 3)]


@pytest.mark.parametrize("delay", [0, 1, 2])
@pytest.mark.parametrize("gradual", [False, True])
def test_talker_streaming_equals_training(delay, gradual):
    """Teacher-forced streaming generation must reproduce the training logits exactly."""
    torch.manual_seed(0)
    t = Talker(llm_dim=32, n_hidden_layers=2, num_codebooks=4, card=16, d_model=32, n_layers=2, n_heads=4,
               dropout=0.0, depth_d_model=16, depth_layers=2, depth_heads=2, first_text_chunk=3, text_chunk=2,
               audio_chunk=4, acoustic_delay=delay).eval()
    n_tok, n_frames = 7, 13
    tok, hid = torch.randn(n_tok, 32), torch.randn(n_tok, 2, 32)
    codes = torch.randint(0, 16, (4, n_frames))
    grid = apply_delay(codes, delay, 16)
    ref = t([tok], [hid], [grid], return_logits=True)["logits"]
    seen = {}

    def pick(j, logits, allowed):
        g = len(stream.cols)
        seen[(g, j)] = logits.clone()
        return int(grid[j, g])

    stream = TalkerStream(t, pick=pick)
    items = t.fuse(tok, hid)
    frames, i = [], 0
    if not gradual:
        stream.push_text(items)
        stream.end_text()
    while not stream.finished:
        if stream.can_step():
            frames += stream.step()
        elif i < n_tok:
            stream.push_text(items[i:i + 1])
            i += 1
        else:
            stream.end_text()
    assert len(stream.cols) == grid.shape[1]
    assert max((seen[k] - ref[k[0], k[1]]).abs().max().item() for k in seen) < 1e-4
    assert torch.equal(torch.stack(frames, 1), codes)


def test_talker_sampling_terminates():
    torch.manual_seed(0)
    t = Talker(llm_dim=16, n_hidden_layers=1, num_codebooks=3, card=8, d_model=16, n_layers=1, n_heads=2,
               dropout=0.0, depth_d_model=8, depth_layers=1, depth_heads=2, max_frames=12).eval()
    s = TalkerStream(t, temperature=1.0, top_k=0)
    s.push_text(t.fuse(torch.randn(3, 16), torch.randn(3, 1, 16)))
    s.end_text()
    frames = []
    while s.can_step():
        frames += s.step()
    assert s.finished and len(frames) <= 12
    assert all(int(f.max()) < 8 for f in frames)  # never EOA/PAD in emitted frames


def test_endpointer():
    ep = Endpointer(frame_ms=80, energy_threshold_db=-40, min_speech_ms=160, min_silence_ms=160, max_silence_ms=400)
    loud, quiet = np.full(1920, 0.1, np.float32), np.zeros(1920, np.float32)
    assert not ep.update(quiet, 1.0)                 # silence before speech never ends a turn
    assert not ep.update(loud, 0.0) and not ep.update(loud, 0.0)
    assert not ep.update(quiet, 0.1) and not ep.update(quiet, 0.1)   # EOT head says "not done"
    assert ep.update(quiet, 0.9) and ep.reason == "eot_head"
    ep.reset()
    for _ in range(3):
        ep.update(loud, 0.0)
    assert [ep.update(quiet, 0.0) for _ in range(5)][-1] and ep.reason == "max_silence"


def test_hotel_tools():
    rows = generate_examples(50, seed=0)
    backend = HotelBackend(make_reservation())
    for r in rows:
        for call in r["tool_calls"]:
            assert validate_call(call) is None, call
            assert "error" not in backend.execute(call)
    assert calls_match([{"name": "create_issue", "arguments": {"category": "noise", "description": "x"}}],
                       [{"name": "create_issue", "arguments": {"category": "noise", "description": "y"}}])
    assert not calls_match([{"name": "order_product", "arguments": {"product": "towel", "quantity": 1}}],
                           [{"name": "order_product", "arguments": {"product": "towel", "quantity": 2}}])


def test_hotel_v2_examples_are_valid_and_varied():
    from s2s.data.hotel import validate_call
    from s2s.data.hotel_v2 import generate_examples_v2

    rows = generate_examples_v2(2000, seed=3)
    assert all(validate_call(c) is None for r in rows for c in r["tool_calls"])
    assert all(r["tool_calls"] or (r.get("reply") and r.get("answer_contains")) for r in rows)
    assert len({r["text"] for r in rows}) > 1200                       # many distinct phrasings
    assert 0.2 < sum(1 for r in rows if r.get("history")) / len(rows) < 0.4
    assert any("reservation" in r["text"].lower() and not r["tool_calls"] for r in rows)


def test_reply_filter_drops_generic_handovers():
    from s2s.prep.reply_texts import is_generic

    assert is_generic("I can't do that myself, but the front desk will be happy to help you with that.")
    assert is_generic("I'm sorry to hear that. Please call the front desk right away.")
    assert is_generic("Okay.")
    assert not is_generic("Jonathan Pine is the night manager.")
    assert not is_generic("Sure, I've ordered two towels for you; it should arrive in about 15 minutes.")
