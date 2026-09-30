import torch

from s2s.eval.encoder_bakeoff import parse_tests, take_hours
from s2s.models.encoders import n_frames, stack_to_12_5hz


def test_frame_rate_helpers():
    assert n_frames(16000 * 3, 16000) == 38 and n_frames(24000 * 3, 24000) == 38  # same grid as Mimi
    h = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)          # 10 frames at 50 Hz
    out = stack_to_12_5hz(h, 3, 4)                                         # needs 12 -> pads with last frame
    assert out.shape == (3, 8) and torch.equal(out[0], h[:4].reshape(-1)) and torch.equal(out[2, -2:], h[-1])


def test_bakeoff_arg_helpers():
    assert parse_tests(["a=x.jsonl", "b=y.jsonl:aug"]) == [("a", "x.jsonl", False), ("b", "y.jsonl", True)]
    rows = [{"duration": 1800}, {"duration": 1800}, {"duration": 1800}]
    assert len(take_hours(rows, 1.0)) == 2 and len(take_hours(rows, 0)) == 3
