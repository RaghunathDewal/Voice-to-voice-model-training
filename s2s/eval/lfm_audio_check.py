"""Where does LFM's reply audio go wrong: the model, our generation loop, or the decoder used by the live UI?

    python s2s/eval/lfm_audio_check.py --out ~/lfm_check                          # base model
    python s2s/eval/lfm_audio_check.py --model-dir ~/lfm_ft/model --out ~/lfm_check_ft
    python s2s/eval/lfm_audio_check.py --prompt-file ~/ghb_prompt.txt --out ~/lfm_check_ghb

For a few spoken questions (voiced by the base model's TTS) it generates a reply with
  stock   liquid-audio's own generate_interleaved, short default system prompt (Liquid's demo setup)
  hybrid  our generate_hybrid (used by the eval and the UI), same prompt
  prompt  our generate_hybrid with --prompt-file + tools (only with --prompt-file)
and decodes every reply two ways:
  detok   processor.decode on the whole reply (LFM2.5's detokenizer; the eval uses this)
  mimi    Mimi streaming decoder, frame by frame (the live UI uses this)
Each reply is saved as a WAV, its loudness printed, and Whisper's transcript compared with the model's text.
Clear "stock + detok" but bad "mimi" = UI decoding problem; bad everywhere = the model's audio.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from s2s.eval.lfm_audio_probe import (EOS_AUDIO, HF_REPO, INTERLEAVED, OUT_SR, generate_hybrid, load_prompt,
                                      load_tools, norm, setup_for, spoken)

QUESTIONS = ["Hello, who am I speaking with?", "What's the wifi password?", "When do we have to check out?",
             "Can you send two extra towels to our room?"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--tools-file", default="demo/thinker/tools_template.json")
    p.add_argument("--out", default=os.path.expanduser("~/lfm_check"))
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    args = p.parse_args()

    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_proc = LFM2AudioProcessor.from_pretrained(HF_REPO, device="cuda").eval()
    base = LFM2AudioModel.from_pretrained(HF_REPO, device="cuda").eval()
    if args.model_dir:
        src = Path(os.path.expanduser(args.model_dir))
        proc = LFM2AudioProcessor.from_pretrained(src, device="cuda").eval()
        model = LFM2AudioModel.from_pretrained(src, device="cuda").eval()
    else:
        proc, model = base_proc, base
    mimi = proc.mimi.eval()

    def chat(system: str, text: str | None = None, wav: np.ndarray | None = None):
        c = ChatState(proc)
        c.new_turn("system"), c.add_text(system), c.end_turn(), c.new_turn("user")
        if wav is not None:
            c.add_audio(torch.from_numpy(wav)[None], OUT_SR)
        else:
            c.add_text(text)
        c.end_turn(), c.new_turn("assistant")
        return c

    @torch.no_grad()
    def tts(text: str) -> np.ndarray:  # always the base model's TTS
        c = ChatState(base_proc)
        c.new_turn("system"), c.add_text("Perform TTS. Use the UK male voice."), c.end_turn()
        c.new_turn("user"), c.add_text(text), c.end_turn(), c.new_turn("assistant")
        fr = [t for t in base.generate_sequential(**c, max_new_tokens=1024, audio_temperature=0.8, audio_top_k=64)
              if t.numel() > 1 and not bool((t == EOS_AUDIO).any())]
        return base_proc.decode(torch.stack(fr, 1)[None]).float().cpu().numpy()[0]

    @torch.no_grad()
    def run(gen, c) -> tuple[str, list]:
        toks, frames = [], []
        for t in gen(**c, max_new_tokens=args.max_new_tokens, audio_temperature=1.0, audio_top_k=4):
            (toks if t.numel() == 1 else frames).append(t)
        text = proc.text.decode(torch.cat(toks)) if toks else ""
        return text, [f for f in frames if not bool((f == EOS_AUDIO).any())]

    @torch.no_grad()
    def decode_detok(frames) -> np.ndarray:
        return proc.decode(torch.stack(frames, 1)[None]).float().cpu().numpy()[0] if frames else np.zeros(1)

    @torch.no_grad()
    def decode_mimi(frames) -> np.ndarray:  # exactly what the live UI does
        if not frames:
            return np.zeros(1)
        with mimi.streaming(1):
            return np.concatenate([mimi.decode(f[None, :, None])[0, 0].float().cpu().numpy() for f in frames])

    variants = [("stock", lambda **k: model.generate_interleaved(**k), INTERLEAVED),
                ("hybrid", lambda **k: generate_hybrid(model, **k), INTERLEAVED)]
    if args.prompt_file:
        sys_prompt, _ = setup_for(load_prompt(Path(os.path.expanduser(args.prompt_file))),
                                  load_tools(Path(args.tools_file)), "system")
        variants.append(("prompt", lambda **k: generate_hybrid(model, **k), sys_prompt))

    rows = []
    for qi, q in enumerate(QUESTIONS):
        qwav = tts(q)
        sf.write(out / f"q{qi}.wav", qwav, OUT_SR)
        for name, gen, system in variants:
            text, frames = run(gen, chat(system, wav=qwav))
            for dec, fn in (("detok", decode_detok), ("mimi", decode_mimi)):
                w = fn(frames)
                path = out / f"q{qi}_{name}_{dec}.wav"
                sf.write(path, w, OUT_SR)
                rows.append({"q": q, "gen": name, "decoder": dec, "text": spoken(text), "frames": len(frames),
                             "seconds": round(len(w) / OUT_SR, 2), "rms": round(float(np.sqrt(np.mean(w ** 2))), 4),
                             "peak": round(float(np.abs(w).max()), 3), "wav": str(path)})
                print(f"q{qi} {name:6s} {dec:5s} | {len(frames):3d} frames {rows[-1]['seconds']:5.1f} s rms "
                      f"{rows[-1]['rms']:.3f} | {spoken(text)[:90]}", flush=True)

    del model, base
    torch.cuda.empty_cache()
    import jiwer
    import torchaudio
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=args.whisper, dtype=torch.float16, device=0)
    print("\nWhisper on each reply (WER vs the model's own text; lower = clearer)")
    for r in rows:
        w, sr = sf.read(r["wav"], dtype="float32")
        if len(w) < sr // 4 or not r["text"]:
            r["heard"], r["wer"] = "", None
        else:
            w16 = torchaudio.functional.resample(torch.from_numpy(w), sr, 16_000).numpy()
            r["heard"] = asr({"raw": w16, "sampling_rate": 16_000},
                             generate_kwargs={"language": "en", "task": "transcribe"})["text"]
            r["wer"] = round(100 * jiwer.wer(norm(r["text"]) or "-", norm(r["heard"]) or "-"))
        print(f"  {r['gen']:6s} {r['decoder']:5s} WER {r['wer'] if r['wer'] is not None else '-':>4} | heard: {r['heard'][:90]}")
    summary = {}
    for r in rows:
        if r["wer"] is not None:
            summary.setdefault(f"{r['gen']}+{r['decoder']}", []).append(r["wer"])
    print("\nmedian WER per setup:", {k: float(np.median(v)) for k, v in summary.items()})
    (out / "check.json").write_text(json.dumps(rows, indent=1))
    print(f"wavs + check.json in {out}")


if __name__ == "__main__":
    main()
