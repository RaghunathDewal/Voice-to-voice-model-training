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


def test_adapter_checkpoint_remembers_its_input_encoder(tmp_path):
    from s2s.models.adapter import SpeechAdapter

    a = SpeechAdapter(1024, 16, d_model=32, n_layers=1, n_heads=2, encoder="parakeet:nvidia/parakeet-ctc-0.6b")
    a.save(str(tmp_path / "a.pt"))
    assert SpeechAdapter.load(str(tmp_path / "a.pt")).hparams["encoder"] == "parakeet:nvidia/parakeet-ctc-0.6b"
    # checkpoints written before the encoder field existed are Mimi adapters
    ckpt = torch.load(tmp_path / "a.pt", weights_only=False)
    del ckpt["hparams"]["encoder"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert SpeechAdapter.load(str(tmp_path / "old.pt")).hparams["encoder"] == "mimi"
