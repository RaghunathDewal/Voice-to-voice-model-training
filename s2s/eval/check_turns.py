"""Check live turns saved by `ws_live --save-turns DIR`: what did the server actually hear?

    python -m s2s.eval.check_turns --dir ~/turns

For every saved turn it prints the length, why the turn ended, the input level, our adapter's CTC
transcript and a reference transcript from Whisper. If Whisper also gets only a fragment, the audio
was cut short or is poor (turn detection, microphone, echo); if Whisper hears the full sentence and
our adapter does not, the adapter is at fault.
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--asr", default="openai/whisper-small")
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
        sec = len(wav) / sr
        lengths.append(sec)
        short += sec < 1.0
        print(f"{os.path.basename(wav_path)}  {sec:4.1f}s  end={info.get('end')}  level={info.get('rms_db')} dB")
        print(f"   whisper: {ref}\n   adapter: {info.get('ctc', '')}\n   reply:   {info.get('reply', '')}")
    if lengths:
        print(f"\n{len(lengths)} turns, median length {np.median(lengths):.1f}s, {short} shorter than 1 s")


if __name__ == "__main__":
    main()
