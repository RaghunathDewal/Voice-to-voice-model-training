"""Warm-starting the adapter / talker for a thinker of another width, and the training time budget."""

import torch

from s2s.config import Config
from s2s.models.adapter import SpeechAdapter
from s2s.models.talker import Talker
from s2s.train.common import apply_time_budget


def test_adapter_warm_start_other_width(tmp_path):
    a = SpeechAdapter(latent_dim=16, llm_dim=32, d_model=24, n_layers=1, n_heads=4)
    a.save(tmp_path / "adapter.pt")
    same = SpeechAdapter.load(tmp_path / "adapter.pt", llm_dim=32)
    assert same.fresh_params == []
    b = SpeechAdapter.load(tmp_path / "adapter.pt", llm_dim=48)
    assert b.hparams["llm_dim"] == 48
    assert set(b.fresh_params) == {"out_proj.weight", "out_proj.bias", "out_norm.weight", "speech_start", "speech_end"}
    assert torch.equal(b.in_proj.weight, a.in_proj.weight)
    assert b(torch.randn(1, 5, 16))["embeds"].shape == (1, 5, 48)


def test_talker_warm_start_other_width(tmp_path):
    t = Talker(llm_dim=32, n_hidden_layers=3, num_codebooks=2, card=8, d_model=24, n_layers=1, n_heads=4,
               depth_d_model=16, depth_layers=1, depth_heads=4)
    t.save(tmp_path / "talker.pt")
    u = Talker.load(tmp_path / "talker.pt", llm_dim=40)
    # only the thinker-facing weights; the projections' biases are talker-sized and are kept
    assert set(u.fresh_params) == {"tok_norm.weight", "tok_proj.weight", "hid_norm.weight", "hid_proj.weight"}
    assert torch.equal(u.temporal.state_dict()[next(iter(u.temporal.state_dict()))],
                       t.temporal.state_dict()[next(iter(t.temporal.state_dict()))])
    assert Talker.load(tmp_path / "talker.pt").fresh_params == []


def test_time_budget_shrinks_steps_and_stops():
    tc = Config({"max_minutes": 1, "max_steps": 10_000, "log_every": 10})
    dev = torch.device("cpu")
    apply_time_budget(tc, 30, t0=0.0 + __import__("time").time() - 30.0, device=dev, log=print)  # 1 s/step
    assert 52 <= tc.max_steps <= 54  # 90% of 60 s at ~1 s/step
    apply_time_budget(tc, 40, t0=__import__("time").time() - 61.0, device=dev, log=print)
    assert tc.max_steps == 40
    off = Config({"max_minutes": 0, "max_steps": 7, "log_every": 10})
    apply_time_budget(off, 30, t0=0.0, device=dev, log=print)
    assert off.max_steps == 7
