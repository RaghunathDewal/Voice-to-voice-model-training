"""How good can our voice sound at best? Real speech through Mimi with 8, 16 and 32 codebooks.

The talker predicts Mimi codes, so its audio can never be better than real speech encoded and decoded
by Mimi with the same number of codebooks. Listen to the files this writes: if the 8-codebook copy is
already rough and the 32-codebook copy is smooth, train the talker with more codebooks
(codec.num_codebooks); if even 32 sounds rough, the codec itself is the limit.

    python -m s2s.eval.codec_ceiling --manifest data/manifests/hifi92_valid_raw.jsonl --max 4 --out-dir /root/codec_ab
    python -m s2s.eval.codec_ceiling --wav some_clip.wav --out-dir /root/codec_ab

Writes <name>_original.wav and <name>_mimi<K>.wav, and prints the share of energy above 5 kHz
(clarity of s, f, t sounds) for each.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from s2s.audio import load_audio, save_audio
from s2s.data.datasets import load_manifest
from s2s.models.codec import MimiCodec


def high_band_share(wav: np.ndarray, sr: int, cutoff: float = 5000.0) -> float:
    spec = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    return float(spec[freqs >= cutoff].sum() / (spec.sum() + 1e-12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=None, help="raw manifest with 'audio' paths (relative to the manifest)")
    p.add_argument("--wav", nargs="*", default=[])
    p.add_argument("--max", type=int, default=4)
    p.add_argument("--codebooks", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--codec", default="kyutai/mimi")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    paths = list(args.wav)
    if args.manifest:
        base = os.path.dirname(os.path.abspath(args.manifest))
        paths += [os.path.join(base, r["audio"]) for r in load_manifest(args.manifest)[: args.max]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codecs = {k: MimiCodec(args.codec, device, k) for k in args.codebooks}
    sr = next(iter(codecs.values())).sample_rate
    os.makedirs(args.out_dir, exist_ok=True)
    shares: dict[str, list[float]] = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        wav = load_audio(path, sr)
        save_audio(os.path.join(args.out_dir, f"{name}_original.wav"), wav, sr)
        shares.setdefault("original", []).append(high_band_share(wav, sr))
        for k, codec in codecs.items():
            out = codec.decode(codec.encode_codes([wav])[0])[: len(wav)]
            save_audio(os.path.join(args.out_dir, f"{name}_mimi{k}.wav"), out, sr)
            shares.setdefault(f"mimi{k}", []).append(high_band_share(out, sr))
    print(f"{len(paths)} clips -> {args.out_dir}")
    for label, v in shares.items():
        print(f"  {label:>9}: energy above 5 kHz {100 * np.mean(v):.1f}%")


if __name__ == "__main__":
    main()
