"""Fine-tune LFM2.5-Audio-1.5B on our hotel concierge task (system-prompt following + our six tools).

Stages (run in order; each is resumable and writes under --work, default ~/lfm_ft):

    python s2s/train/lfm_finetune.py data    # text conversations (hotel_ghb + hotel_v3), train/val split
    python s2s/train/lfm_finetune.py voice   # guest + assistant audio with LFM's own TTS (parallel workers)
    python s2s/train/lfm_finetune.py build   # -> liquid-audio training format (Mimi codes, mel features)
    python s2s/train/lfm_finetune.py train   # full fine-tune, fp32 master weights + bf16 autocast
    # then evaluate the result with the same test as the base model:
    python s2s/eval/lfm_audio_probe.py --model-dir ~/lfm_ft/model --placement system --out ~/lfm_ft/eval

or all at once: bash scripts/lfm_finetune.sh

Training format (what the model learns, matching how the probe / a server runs it):
  system     "Respond with interleaved text and audio." + the property prompt + the tool list
             (<|tool_list_start|>[...]<|tool_list_end|>)
  user       the guest's speech (85%, LFM TTS voices with speed / gain / noise augmentation) or typed text
  assistant  either a tool call, text only:  <|tool_call_start|>[name(arg="v")]<|tool_call_end|>
             or the spoken reply, interleaved text + audio (6 text tokens : 12 audio frames, as LFM generates)
  tool       <|tool_response_start|>{result json}<|tool_response_end|>
Only assistant turns are supervised. A tool-call turn is text only, so at inference a turn that starts with
<|tool_call_start|> must stay in text mode (lfm_audio_probe.generate_hybrid does that).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
GUEST_VOICES = ["US female", "UK male", "US male", "UK female"]
SR = 24_000


def device() -> str:
    """cuda (ROCm or NVIDIA); CPU only for smoke tests (liquid-audio's detokenizer calls .cuda())."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    torch.nn.Module.cuda = lambda module, *a, **k: module
    return "cpu"


def work_dir(args) -> Path:
    w = Path(os.path.expanduser(args.work))
    w.mkdir(parents=True, exist_ok=True)
    return w


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def clip_jobs(convs: list[dict], assistant_voice: str) -> list[dict]:
    """Every sentence that needs audio: guest turns (one voice per conversation) and spoken replies."""
    jobs = []
    for c in convs:
        voice = GUEST_VOICES[zlib.crc32(c["id"].encode()) % len(GUEST_VOICES)]
        for ti, t in enumerate(c["turns"]):
            if not t.get("typed"):
                jobs.append({"id": f"{c['id']}_t{ti}_u", "text": t["user"], "voice": voice})
            for si, s in enumerate(t["assistant"]):
                if "say" in s:
                    jobs.append({"id": f"{c['id']}_t{ti}_s{si}", "text": s["say"], "voice": assistant_voice})
    return jobs


# ----------------------------------------------------------------------------- 1. data
def stage_data(args) -> None:
    from s2s.data.hotel_ghb import generate_mixed

    w = work_dir(args)
    rows = generate_mixed(args.n_ghb, args.n_v3, seed=args.seed)
    rng = random.Random(args.seed)
    for r in rows:
        for t in r["turns"]:
            t["typed"] = rng.random() < args.typed_frac
    n_val = min(max(20, int(len(rows) * args.val_frac)), max(1, len(rows) // 3))
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with (w / f"convs_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    jobs = clip_jobs(rows, args.assistant_voice)
    print(f"{len(rows)} conversations ({len(rows) - n_val} train / {n_val} val), "
          f"{sum(len(r['turns']) for r in rows)} guest turns, {len(jobs)} clips to voice -> {w}")


# ----------------------------------------------------------------------------- 2. voice
def _voice_worker(rank: int, world: int, jobs: list[dict], out_dir: str, max_wer: float, retries: int) -> None:
    import jiwer
    import numpy as np
    import soundfile as sf
    import torch

    from s2s.eval.lfm_audio_probe import EOS_AUDIO, norm, spoken
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    torch.manual_seed(1000 + rank)
    dev = device()
    proc = LFM2AudioProcessor.from_pretrained(HF_REPO, device=dev).eval()
    model = LFM2AudioModel.from_pretrained(HF_REPO, device=dev,
                                           dtype=torch.bfloat16 if dev == "cuda" else torch.float32).eval()

    def chat(system: str, text: str | None = None, wav=None):
        c = ChatState(proc)
        c.new_turn("system"), c.add_text(system), c.end_turn(), c.new_turn("user")
        if wav is not None:
            c.add_audio(torch.from_numpy(wav)[None], SR)
        else:
            c.add_text(text)
        c.end_turn(), c.new_turn("assistant")
        return c

    @torch.no_grad()
    def tts(text: str, voice: str):
        frames = [t for t in model.generate_sequential(**chat(f"Perform TTS. Use the {voice} voice.", text),
                                                       max_new_tokens=1500, audio_temperature=0.8, audio_top_k=64)
                  if t.numel() > 1 and not bool((t == EOS_AUDIO).any())]
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return proc.decode(torch.stack(frames, 1)[None]).float().cpu().numpy()[0]

    @torch.no_grad()
    def asr(wav) -> str:
        toks = [t for t in model.generate_sequential(**chat("Perform ASR.", wav=wav), max_new_tokens=256)
                if t.numel() == 1]
        return spoken(proc.text.decode(torch.cat(toks))) if toks else ""

    mine = [j for i, j in enumerate(jobs) if i % world == rank]
    log = open(Path(out_dir) / f"voice_log_{rank}.jsonl", "a", encoding="utf-8")
    t0, done = time.time(), 0
    for j in mine:
        path = Path(out_dir) / f"{j['id']}.wav"
        if path.exists():
            continue
        best = None
        for _ in range(retries + 1):
            wav = tts(j["text"], j["voice"])
            if len(wav) < SR // 4:
                continue
            wer = jiwer.wer(norm(j["text"]) or "-", norm(asr(wav)) or "-")
            if best is None or wer < best[1]:
                best = (wav, wer)
            if wer <= max_wer:
                break
        if best is None:
            log.write(json.dumps({"id": j["id"], "failed": True}) + "\n")
            continue
        sf.write(path, best[0], SR, subtype="PCM_16")
        log.write(json.dumps({"id": j["id"], "wer": round(best[1], 3), "sec": round(len(best[0]) / SR, 2)}) + "\n")
        log.flush()
        done += 1
        if rank == 0 and done % 50 == 0:
            rate = done / (time.time() - t0)
            print(f"[voice] worker 0: {done}/{len(mine)} clips, {rate:.2f}/s per worker, "
                  f"~{(len(mine) - done) / max(rate, 1e-6) / 60:.0f} min left", flush=True)


def stage_voice(args) -> None:
    import torch.multiprocessing as mp

    w = work_dir(args)
    convs = read_jsonl(w / "convs_train.jsonl") + read_jsonl(w / "convs_val.jsonl")
    jobs = clip_jobs(convs, args.assistant_voice)
    out = w / "wavs"
    out.mkdir(exist_ok=True)
    todo = [j for j in jobs if not (out / f"{j['id']}.wav").exists()]
    print(f"[voice] {len(jobs)} clips, {len(todo)} to do, {args.workers} workers on one GPU")
    if todo:
        mp.start_processes(_voice_worker, args=(args.workers, todo, str(out), args.max_wer, args.retries),
                           nprocs=args.workers, join=True, start_method="spawn")
    logs = [json.loads(line) for p in out.glob("voice_log_*.jsonl") for line in p.open() if line.strip()]
    ok = [x for x in logs if not x.get("failed")]
    if ok:
        bad = sum(x["wer"] > args.max_wer for x in ok)
        print(f"[voice] {len(ok)} clips, mean ASR check WER {sum(x['wer'] for x in ok) / len(ok):.3f}, "
              f"{bad} above {args.max_wer} (kept best of {args.retries + 1} tries), "
              f"{sum(x['sec'] for x in ok) / 3600:.1f} h audio, {len(logs) - len(ok)} failed")


# ----------------------------------------------------------------------------- 3. build
def _augment(wav, rng: random.Random):
    """Guest audio only: speed 0.9-1.1, gain, and background noise on half of the clips."""
    import numpy as np
    import torch
    import torchaudio

    speed = rng.uniform(0.9, 1.1)
    if abs(speed - 1) > 0.01:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), int(SR * speed), SR).numpy()
    wav = wav * rng.uniform(0.4, 1.0)
    if rng.random() < 0.5:
        snr = rng.uniform(12, 35)
        p = float(np.mean(wav**2)) + 1e-12
        wav = wav + np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(wav)).astype(np.float32) \
            * math.sqrt(p / 10 ** (snr / 10))
    return np.clip(wav, -1, 1).astype(np.float32)


def to_messages(conv: dict, wavs: Path, rng: random.Random, augment: bool):
    """One conversation -> liquid-audio ChatMessages (None if a clip is missing)."""
    import soundfile as sf

    from liquid_audio.data.types import AudioSegment, ChatMessage, InterleavedSegment, TextSegment
    from s2s.data.hotel_ghb import call_text
    from s2s.eval.lfm_audio_probe import setup_for

    def wav_bytes(cid: str, aug: bool) -> bytes | None:
        p = wavs / f"{cid}.wav"
        if not p.exists():
            return None
        if not aug:
            return p.read_bytes()
        w, _ = sf.read(p, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, _augment(w, rng), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    tools = [t.get("function", t) for t in conv.get("tools") or []]
    system, _ = setup_for(conv["system"], tools, "system")
    msgs = [ChatMessage(role="system", content=[TextSegment(text=system)])]
    for ti, t in enumerate(conv["turns"]):
        if t.get("typed"):
            msgs.append(ChatMessage(role="user", content=[TextSegment(text=t["user"])]))
        else:
            b = wav_bytes(f"{conv['id']}_t{ti}_u", augment)
            if b is None:
                return None
            msgs.append(ChatMessage(role="user", content=[AudioSegment(audio=b)]))
        for si, s in enumerate(t["assistant"]):
            if "call" in s:
                msgs.append(ChatMessage(role="assistant", content=[TextSegment(text=call_text([s["call"]]))]))
                msgs.append(ChatMessage(role="tool", content=[TextSegment(
                    text=f"<|tool_response_start|>{json.dumps(s['result'])}<|tool_response_end|>")]))
            else:
                b = wav_bytes(f"{conv['id']}_t{ti}_s{si}", False)
                if b is None:
                    return None
                msgs.append(ChatMessage(role="assistant", content=[InterleavedSegment(text=s["say"], audio=b)]))
    return msgs


class ConvIterator:
    """Re-iterable, picklable source of ChatMessage lists (datasets.Dataset.from_generator needs both)."""

    def __init__(self, convs: list[dict], wavs: str, seed: int, augment: bool):
        self.convs, self.wavs, self.seed, self.augment = convs, wavs, seed, augment

    def complete(self, conv: dict) -> bool:
        need = [f"{conv['id']}_t{ti}_u" for ti, t in enumerate(conv["turns"]) if not t.get("typed")]
        need += [f"{conv['id']}_t{ti}_s{si}" for ti, t in enumerate(conv["turns"])
                 for si, s in enumerate(t["assistant"]) if "say" in s]
        return all((Path(self.wavs) / f"{n}.wav").exists() for n in need)

    def __iter__(self):
        rng = random.Random(self.seed)
        for c in self.convs:
            m = to_messages(c, Path(self.wavs), rng, self.augment) if self.complete(c) else None
            if m is not None:
                yield m


def stage_build(args) -> None:
    from liquid_audio import LFM2AudioProcessor
    from liquid_audio.data.mapper import LFM2AudioChatMapper
    from liquid_audio.data.preprocess import preprocess_dataset

    w = work_dir(args)
    proc = LFM2AudioProcessor.from_pretrained(HF_REPO, device=device()).eval()
    mapper = LFM2AudioChatMapper(proc)
    for split in ("train", "val"):
        out = w / "data" / split
        if out.exists():
            shutil.rmtree(out)
        convs = read_jsonl(w / f"convs_{split}.jsonl")
        data = ConvIterator(convs, str(w / "wavs"), args.seed + (split == "val"), augment=(split == "train"))
        missing = sum(1 for c in convs if not data.complete(c))
        t0 = time.time()
        preprocess_dataset(data=data, output_path=out, mapper=mapper, max_context_length=args.context)
        print(f"[build] {split}: {len(convs) - missing} conversations -> {out} "
              f"({missing} skipped for missing audio, {time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- 4. train
def export(model, out: Path) -> None:
    """A folder LFM2AudioModel / LFM2AudioProcessor.from_pretrained(Path(...)) can load: the base model's
    files (config, tokenizer, Mimi, detokenizer) + our weights in bf16."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    base = Path(snapshot_download(HF_REPO))
    out.mkdir(parents=True, exist_ok=True)
    for p in base.iterdir():
        if p.name == "model.safetensors" or (out / p.name).exists():
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p.resolve(), out / p.name)
    seen, sd = set(), {}
    for k, v in model.state_dict().items():
        if v.data_ptr() in seen:  # tied weights are stored once
            continue
        seen.add(v.data_ptr())
        sd[k] = v.detach().to(torch.bfloat16).contiguous().cpu()
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})


def stage_train(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from liquid_audio import LFM2AudioModel
    from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator

    w = work_dir(args)
    torch.manual_seed(args.seed)
    dev = device()
    model = LFM2AudioModel.from_pretrained(HF_REPO, device=dev, dtype=torch.float32)
    model.conf.text_loss_multiplier = args.text_loss_weight
    # fp32 weights + bf16 autocast: liquid-audio's _prefill writes the audio adapter's output (bf16 under
    # autocast) into a buffer typed like the text embeddings (fp32) -> keep the adapter output in fp32
    _adapter_forward = model.audio_adapter.forward
    model.audio_adapter.forward = lambda *a, **k: _adapter_forward(*a, **k).float()
    if args.freeze_encoder:  # keep the (already good) hearing; train the adapter, LM, audio head
        for p in model.conformer.parameters():
            p.requires_grad_(False)
    if args.train_only:  # e.g. "lfm." for the language model only (or a tiny module for a smoke test)
        for n, p in model.named_parameters():
            p.requires_grad_(any(n.startswith(x) for x in args.train_only.split(",")))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] trainable {sum(p.numel() for p in params) / 1e6:.0f}M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    train = LFM2DataLoader(str(w / "data" / "train"), context_length=args.context)
    val = LFM2DataLoader(str(w / "data" / "val"), context_length=args.context)
    dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=lfm2_collator,
                    num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    vdl = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=lfm2_collator,
                     num_workers=args.num_workers)
    steps = args.max_steps or math.ceil(args.epochs * len(dl) / args.grad_accum)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"[train] {len(train)} train / {len(val)} val conversations, batch {args.batch_size} x accum "
          f"{args.grad_accum}, {steps} steps (~{steps * args.batch_size * args.grad_accum / max(1, len(train)):.1f} "
          f"epochs), lr {args.lr}, text loss weight {args.text_loss_weight}")

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        tot = {"loss": 0.0, "text": 0.0, "audio": 0.0}
        n = 0
        for b in vdl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(b.to(dev))
            tot["loss"] += o.loss.item()
            tot["text"] += o.text_loss.item()
            tot["audio"] += o.audio_loss.item()
            n += 1
        model.train()
        return {k: v / max(n, 1) for k, v in tot.items()}

    v = validate()
    print(f"[train] step 0 | val loss {v['loss']:.3f} text {v['text']:.3f} audio {v['audio']:.3f}", flush=True)
    log = open(w / "train_log.jsonl", "a", encoding="utf-8")
    log.write(json.dumps({"step": 0, **{f"val_{k}": x for k, x in v.items()}}) + "\n")
    best = v["text"]
    step, t0, it = 0, time.time(), iter(dl)
    acc = {"loss": 0.0, "text": 0.0, "audio": 0.0}
    while step < steps:
        for _ in range(args.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(b.to(dev))
            (o.loss / args.grad_accum).backward()
            acc["loss"] += o.loss.item() / args.grad_accum
            acc["text"] += o.text_loss.item() / args.grad_accum
            acc["audio"] += o.audio_loss.item() / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            el = time.time() - t0
            print(f"[train] step {step}/{steps} | loss {acc['loss'] / args.log_every:.3f} text "
                  f"{acc['text'] / args.log_every:.3f} audio {acc['audio'] / args.log_every:.3f} | grad {gn:.2f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {el / 60:.1f} min, ~{el / step * (steps - step) / 60:.0f} min left",
                  flush=True)
            log.write(json.dumps({"step": step, **{k: x / args.log_every for k, x in acc.items()}}) + "\n")
            acc = {"loss": 0.0, "text": 0.0, "audio": 0.0}
        if step % args.eval_every == 0 or step == steps:
            v = validate()
            print(f"[train] step {step} | val loss {v['loss']:.3f} text {v['text']:.3f} audio {v['audio']:.3f}",
                  flush=True)
            log.write(json.dumps({"step": step, **{f"val_{k}": x for k, x in v.items()}}) + "\n")
            log.flush()
            if v["text"] < best:
                best = v["text"]
                export(model, w / "model")
                print(f"[train] saved best (val text loss {best:.3f}) -> {w / 'model'}", flush=True)
    export(model, w / "model_last")
    if not (w / "model" / "model.safetensors").exists():
        export(model, w / "model")
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min. best -> {w / 'model'}, last -> {w / 'model_last'}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["data", "voice", "build", "train"])
    p.add_argument("--work", default="~/lfm_ft")
    p.add_argument("--seed", type=int, default=0)
    # data
    p.add_argument("--n-ghb", type=int, default=2600, help="conversations with our six tools")
    p.add_argument("--n-v3", type=int, default=1000, help="conversations with other tool names (generalisation)")
    p.add_argument("--typed-frac", type=float, default=0.15, help="guest turns given as text instead of audio")
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--assistant-voice", default="US female", choices=GUEST_VOICES)
    # voice
    p.add_argument("--workers", type=int, default=8, help="TTS processes on the GPU (each holds the model, ~4 GB)")
    p.add_argument("--max-wer", type=float, default=0.25, help="re-synthesise a clip if LFM's ASR check is worse")
    p.add_argument("--retries", type=int, default=2)
    # build / train
    p.add_argument("--context", type=int, default=2048, help="max positions per conversation (longer are skipped)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=0, help="overrides --epochs")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--text-loss-weight", type=float, default=3.0,
                   help="weight of text tokens (replies, tool calls) vs audio codes in the loss")
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--train-only", default="", help="comma-separated parameter-name prefixes to train (default: all)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    args = p.parse_args()
    {"data": stage_data, "voice": stage_voice, "build": stage_build, "train": stage_train}[args.stage](args)


if __name__ == "__main__":
    main()
