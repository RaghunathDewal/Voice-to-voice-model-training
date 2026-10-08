"""Check live turns saved by `ws_live --save-turns DIR`: what did the server actually hear?

    python -m s2s.eval.check_turns --dir ~/turns

For every saved turn it prints the length, why the turn ended, the input level, our adapter's CTC
transcript and a reference transcript from Whisper. If Whisper also gets only a fragment, the audio
was cut short or is poor (turn detection, microphone, echo); if Whisper hears the full sentence and
our adapter does not, the adapter is at fault.

Audio quality per turn:
  snr   speech level minus the noise floor between words (dB). Under ~15 dB recognition degrades
        fast, and turning the volume up does not help: the noise is amplified with the speech.
  clip  share of samples at full scale (distortion from speaking too loud / too close).
  bw    frequency below which 99% of the energy lies. ~3.5 kHz = narrowband (Bluetooth headset in
        call mode, phone line); a clean wideband mic is 6-8 kHz.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from s2s.audio import resample


def quality(wav: np.ndarray, sr: int) -> dict:
    """SNR (90th minus 10th percentile of 20 ms frame levels), clipping share, 99% energy bandwidth."""
    hop = int(0.02 * sr)
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop) if len(wav) >= hop else wav[None]
    db = 10 * np.log10(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    n = 1024
    segs = wav[: len(wav) // n * n].reshape(-1, n) if len(wav) >= n else np.pad(wav, (0, n - len(wav)))[None]
    psd = np.mean(np.abs(np.fft.rfft(segs * np.hanning(n), axis=1)) ** 2, axis=0)
    cum = np.cumsum(psd) / (psd.sum() + 1e-12)
    bw = float(np.searchsorted(cum, 0.99)) * sr / n
    return {"snr": float(np.percentile(db, 90) - np.percentile(db, 10)),
            "clip": float(np.mean(np.abs(wav) >= 0.99)), "bw": bw}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--asr", default="openai/whisper-small")
    p.add_argument("--try-rates", action="store_true",
                   help="also transcribe each turn as if the browser had sent 44.1/48/16 kHz audio under a wrong "
                        "label (a sample-rate mismatch makes speech too fast or slow for every recogniser)")
    args = p.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = WhisperProcessor.from_pretrained(args.asr)
    model = WhisperForConditionalGeneration.from_pretrained(args.asr).to(device).eval()
    lengths, short = [], 0
    for wav_path in sorted(glob.glob(os.path.join(os.path.expanduser(args.dir), "turn_*.wav"))):
        wav, sr = sf.read(wav_path, dtype="float32")
        info_path = wav_path[:-4] + ".json"
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        feats = proc.feature_extractor(resample(wav, sr, 16000), sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats.to(device), language="en", task="transcribe", max_new_tokens=120)
        ref = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        alt = []
        if args.try_rates:  # reinterpret the samples at other true rates: speed changes by true/assumed
            for ratio, label in [(44100 / 48000, "was 44.1k sent as 48k"), (48000 / 44100, "was 48k sent as 44.1k"),
                                 (16000 / 48000, "was 16k sent as 48k"), (48000 / 16000, "was 48k sent as 16k")]:
                w = resample(wav, int(round(sr * ratio)), 16000)
                f2 = proc.feature_extractor(w, sampling_rate=16000, return_tensors="pt").input_features
                with torch.no_grad():
                    i2 = model.generate(f2.to(device), language="en", task="transcribe", max_new_tokens=120)
                alt.append(f"   if {label}: {proc.batch_decode(i2, skip_special_tokens=True)[0].strip()}")
        sec = len(wav) / sr
        lengths.append(sec)
        short += sec < 1.0
        q = quality(wav, sr)
        print(f"{os.path.basename(wav_path)}  {sec:4.1f}s  end={info.get('end')}  level={info.get('rms_db')} dB  "
              f"snr={q['snr']:.0f} dB  clip={100 * q['clip']:.1f}%  bw={q['bw'] / 1000:.1f} kHz")
        print(f"   whisper: {ref}\n   adapter: {info.get('ctc', '')}\n   reply:   {info.get('reply', '')}")
        for line in alt:
            print(line)
    if lengths:
        print(f"\n{len(lengths)} turns, median length {np.median(lengths):.1f}s, {short} shorter than 1 s")


if __name__ == "__main__":
    main()
