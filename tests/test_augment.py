import numpy as np

from s2s.augment import augment


def test_augment_is_deterministic_same_length_and_bounded():
    sr = 24000
    t = np.arange(sr * 2) / sr
    wav = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    outs = [augment(wav, sr, f"clip{i}") for i in range(20)]
    assert all(o.shape == wav.shape and o.dtype == np.float32 and np.abs(o).max() <= 1.0 for o in outs)
    assert np.array_equal(augment(wav, sr, "clip3"), outs[3])
    assert sum(not np.allclose(o / (np.abs(o).max() + 1e-9), wav / 0.3, atol=1e-3) for o in outs) >= 15
