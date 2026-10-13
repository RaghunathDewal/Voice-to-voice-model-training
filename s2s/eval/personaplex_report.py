"""Score the PersonaPlex test calls (scripts/kaggle_personaplex.sh): speed, turn-taking and answers.

Per guest turn, from the agent's output audio and text:
  reply delay     from the end of the guest's turn to the agent's first speech (negative: it started early)
  no reply        nothing said before the guest's next turn
  talked over     agent speech while the guest was speaking (share of the guest's speech time)
  correct         the agent's words for that turn (its text, or Whisper on its audio) contain an expected fact
Speed: compute per 80 ms frame (live use needs < 80 ms on the serving GPU).

    python -m s2s.eval.personaplex_report --calls calls.json --run-dir out --out out/summary.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf

SPECIAL = {"EPAD", "BOS", "EOS", "PAD"}


def speech_frames(wav: np.ndarray, sr: int, hop_s: float = 0.04) -> tuple[np.ndarray, float]:
    """Voice activity per 40 ms frame (energy, threshold relative to the loud parts of this recording)."""
    hop = int(hop_s * sr)
    n = len(wav) // hop
    rms = np.sqrt(np.mean(wav[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(0)
    thr = max(0.006, 0.12 * float(np.percentile(rms, 95))) if n else 1.0
    active = rms > thr
    # at least 3 frames (120 ms) in a row counts as speech: clicks and breaths do not
    out = np.zeros_like(active)
    run = 0
    for i, a in enumerate(active):
        run = run + 1 if a else 0
        if run >= 3:
            out[i - run + 1: i + 1] = True
    return out, hop_s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    args = p.parse_args()
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", args.whisper, device=0 if torch.cuda.is_available() else -1,
                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    calls = json.load(open(args.calls))
    turns_out, speed = [], []
    for call in calls:
        jf = os.path.join(args.run_dir, f"{call['id']}.json")
        if not os.path.exists(jf):
            continue
        run = json.load(open(jf))
        speed.append(run["stats"])
        agent, sr = sf.read(os.path.join(args.run_dir, f"{call['id']}.wav"), dtype="float32")
        guest, gsr = sf.read(call["wav"], dtype="float32")
        if gsr != sr:
            import librosa

            guest = librosa.resample(guest, orig_sr=gsr, target_sr=sr)
        n = min(len(agent), len(guest))
        mix = np.stack([guest[:n], agent[:n]], axis=1)  # left: guest, right: agent (to listen to)
        sf.write(os.path.join(args.run_dir, f"{call['id']}_mix.wav"), mix, sr)
        act, hop_s = speech_frames(agent, sr)
        gact, _ = speech_frames(guest, sr)
        fs = run["frame_s"]
        turns = call.get("turns") or []
        for k, t in enumerate(turns):
            nxt = turns[k + 1]["start"] if k + 1 < len(turns) else len(agent) / sr
            a, b, e = int(t["start"] / hop_s), int(t["end"] / hop_s), int(nxt / hop_s)
            spoken = np.nonzero(gact[a:b])[0]  # the guest's last word (the clip ends with some silence)
            if len(spoken):
                a, b = a + spoken[0], a + spoken[-1] + 1
            over = float(act[a:b].mean()) if b > a else 0.0
            after = np.nonzero(act[max(b - int(0.5 / hop_s), a): e])[0]  # from 0.5 s before the end
            delay = (after[0] * hop_s - 0.5) if len(after) else None
            words = "".join(x for x in run["text"][int(t["start"] / fs): int(nxt / fs)] if x not in SPECIAL).strip()
            seg = agent[int(b * hop_s * sr): int(nxt * sr)]
            heard = asr({"raw": seg, "sampling_rate": sr}, generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip() if len(seg) > sr // 2 and delay is not None else ""
            hay = f"{words} {heard}".lower()
            ok = any(kw.lower() in hay for kw in t.get("keywords", []))
            turns_out.append({"call": call["id"], "guest": t["text"], "agent_text": words, "agent_heard": heard,
                              "delay_s": delay, "talked_over": over, "correct": ok if t.get("keywords") else None})

    L = ["# PersonaPlex-7B on hotel calls", ""]
    if speed:
        s0 = speed[0]
        p50 = float(np.median([s["frame_ms_p50"] for s in speed]))
        p95 = float(np.median([s["frame_ms_p95"] for s in speed]))
        L += [f"GPU: {s0['gpu']} | {s0['dtype']}{' | spread over GPUs/CPU (cpu-offload)' if s0['offload'] else ''} | "
              f"voice {s0['voice']} | model load {s0['load_s']:.0f}s", "",
              "## Speed", "", f"- compute per 80 ms frame: **{p50:.0f} ms** (p95 {p95:.0f} ms) -> "
              + ("keeps up live" if p95 < 80 else f"**{p50 / 80:.1f}x slower than live**: this GPU cannot run it in real time"),
              f"- reading the persona prompt before a call: {np.median([s['prompt_s'] for s in speed]):.1f}s", ""]
    scored = [t for t in turns_out if t["correct"] is not None]
    delays = [t["delay_s"] for t in turns_out if t["delay_s"] is not None]
    if turns_out:
        L += ["## Conversation (hotel calls, the facts are in its prompt)", "",
              f"- answers with the right fact: **{sum(t['correct'] for t in scored)}/{len(scored)}**",
              f"- no reply before the guest spoke again: {sum(t['delay_s'] is None for t in turns_out)}/{len(turns_out)}",
              (f"- reply delay after the guest stops: median **{np.median(delays):.2f}s** "
               f"(min {min(delays):.2f}, max {max(delays):.2f})" if delays else "- reply delay: no replies"),
              f"- talked while the guest was speaking (share of guest speech): "
              f"{np.mean([t['talked_over'] for t in turns_out]) * 100:.0f}%", "",
              "| call | guest | agent (its text) | agent (Whisper) | delay | right fact |", "|---|---|---|---|---|---|"]
        for t in turns_out:
            d = f"{t['delay_s']:.2f}s" if t["delay_s"] is not None else "none"
            c = "" if t["correct"] is None else ("yes" if t["correct"] else "**no**")
            L.append(f"| {t['call']} | {t['guest']} | {t['agent_text'][:160]} | {t['agent_heard'][:160]} | {d} | {c} |")
    L += ["", "Listen to `<call>_mix.wav` (left ear: guest, right ear: agent) for naturalness, interruptions and voice "
          "quality: the numbers cannot judge those.", "",
          "## Production checklist (from the model card / code, not measured here)", "",
          "- tool / function calling: none (it only speaks); reservations, bookings and lookups would need a "
          "separate system listening to its text",
          "- facts: only what is in the persona prompt at the start of the call; it cannot look anything up mid-call",
          "- language: English only",
          "- hardware: tested by NVIDIA on A100 80 GB / H100; one conversation per model instance in their server "
          "(serving many callers needs batching work or one GPU per few calls)",
          "- licence: NVIDIA Open Model License (commercial use allowed) + CC-BY-4.0 (Moshi base)"]
    open(args.out, "w").write("\n".join(L) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"turns": turns_out, "speed": speed}, f, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
