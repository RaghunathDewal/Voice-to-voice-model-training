"""LFM2.5-Audio-1.5B (Liquid AI) on our hotel concierge use case, out of the box (no fine-tuning).

Standalone: needs only `liquid-audio` (Python >= 3.12), torch, soundfile, jiwer. Set up with
scripts/lfm_setup.sh, then:

    python s2s/eval/lfm_audio_probe.py --out ~/lfm_results
    # your own prompt / tools / canned tool results / questions (keep private files out of the repo):
    python s2s/eval/lfm_audio_probe.py --prompt-file ~/my_prompt.txt --tools-file ~/my_tools.json \
        --mock-results ~/my_mock.json --questions-file ~/my_questions.json --out ~/lfm_results

What it measures (everything goes to <out>/summary.md, summary.json, turns.jsonl and wavs/):

  env         GPU, memory, load time
  placement   where our prompt works best (system message first/last, or in the guest's turn), typed
              questions; the best one is used for the rest (or force it with --placement)
  tts         guest questions spoken with LFM's own TTS (4 voices) -- or your recordings via --audio-dir
  asr         LFM's own transcript of each question (word error rate = how well it hears)
  s2s         MAIN TEST. Spoken question -> our system prompt + tools -> spoken answer (interleaved mode).
              Accuracy: facts from the prompt, "I don't know" for missing facts, right tool + arguments,
              no tool when none fits. Tool calls get the canned result and a second (spoken) round.
              Latency: encoder+prompt prefill, first text token, first audio frame, first audible audio
              (frame decoded by Mimi in streaming mode), whole reply, real-time factor, reply length.
  text        same questions TYPED (control): speech-vs-text accuracy gap = cost of the speech interface
  noisy       (only with --noise-snr) the s2s test again with background noise added to the questions
  multiturn   one 6-turn conversation: context kept? how latency grows with history (no prompt cache)
  concurrent  N conversations at once with the stock liquid-audio code (one stream per call, threads)
  batched     upper bound for a batched server: backbone + audio-frame steps for B streams in ONE batch,
              -> how many streams one GPU can keep in real time
  whisper     intelligibility of the spoken replies (Whisper transcript vs the model's own text)
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import platform
import random
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = REPO / "demo/thinker/example_prompt.txt"
DEFAULT_TOOLS = REPO / "demo/thinker/tools_template.json"
DEFAULT_MOCK = REPO / "demo/thinker/mock_results_template.json"
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
INTERLEAVED = "Respond with interleaved text and audio."
VOICES = ["US female", "UK male", "US male", "UK female"]
EOS_AUDIO = 2048
OUT_SR = 24_000

# kind: fact (answer from the prompt), unknown (not in the prompt -> say so / reception), tool (call one of
# `tools`, optional `args` subset), no_tool (no tool fits -> no call), other. `expect`: groups of
# alternatives; every group must match the spoken text (case-insensitive substring).
QUESTIONS = [
    {"q": "What's the wifi password?", "kind": "fact", "expect": [["otter2026", "otter 2026", "otter twenty"]]},
    {"q": "Hey, till when can we swim?", "kind": "fact", "expect": [["7", "seven"]]},
    {"q": "Can I bring my dog, and is it extra?", "kind": "fact", "expect": [["15", "fifteen"]]},
    {"q": "When do we have to leave on the last day?", "kind": "fact", "expect": [["10", "ten"]]},
    {"q": "Which lodge are we in again?", "kind": "fact", "expect": [["heron"], ["14", "fourteen"]]},
    {"q": "What time is breakfast?", "kind": "fact", "expect": [["7:30", "7.30", "seven thirty"]]},
    {"q": "Is there a gym?", "kind": "unknown", "expect": [["reception", "don't have", "do not have", "not sure"]]},
    {"q": "How far is the nearest supermarket?", "kind": "unknown",
     "expect": [["reception", "don't have", "do not have", "not sure"]]},
    {"q": "Could you send two more towels to the lodge?", "kind": "tool",
     "tools": ["place_product_order", "browse_products"], "args": {"place_product_order": {"quantity": 2}}},
    {"q": "What can I order to the lodge?", "kind": "tool", "tools": ["browse_products"]},
    {"q": "The barbecue won't light, can someone check it?", "kind": "tool", "tools": ["report_unit_issue"]},
    {"q": "There's no hot water in the shower.", "kind": "tool", "tools": ["report_unit_issue"]},
    {"q": "What's the status of my orders?", "kind": "tool", "tools": ["get_orders_by_reservation"]},
    {"q": "Can you check order number 8002 for me?", "kind": "tool", "tools": ["get_order_by_id"],
     "args": {"get_order_by_id": {"order_id": 8002}}},
    {"q": "Is there any update on the problem I reported?", "kind": "tool",
     "tools": ["get_reported_issues_by_reservation"]},
    {"q": "Can you book us a table for dinner tonight?", "kind": "no_tool"},
    {"q": "Please wake me up at 6 tomorrow.", "kind": "no_tool"},
    {"q": "Can we stay one more night?", "kind": "no_tool"},
    {"q": "Who are you?", "kind": "other", "expect": [["aria", "concierge"]]},
]
MULTITURN = [
    "Hi, what's the wifi password?",
    "Great. Could you also send us two extra towels?",
    "Actually, make that three towels.",
    "And until when is the pool open?",
    "What did I just order?",
    "Perfect, thank you, bye!",
]


# ----------------------------------------------------------------------------- small helpers
def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now() -> float:
    return time.perf_counter()


def pct(xs: list, q: float) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def ms(x: float | None) -> str:
    return "-" if x is None else f"{1000 * x:.0f} ms"


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(statistics.mean(xs)) if xs else None


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("{{", "{").replace("}}", "}").strip()


def load_tools(path: Path) -> list[dict]:
    tools = json.loads(path.read_text(encoding="utf-8"))
    # OpenAI style {"type": "function", "function": {...}} -> the plain {"name", "description", "parameters"}
    return [t.get("function", t) for t in tools]


PLACEMENTS = ["system", "system_last", "user"]


def setup_for(prompt: str, tools: list[dict], placement: str) -> tuple[str, str | None]:
    """(system message, text put before the guest's first turn). LFM2 chat template: tools go at the end of the
    system message between <|tool_list_start|> markers. LFM2.5-Audio was trained with the fixed system line
    "Respond with interleaved text and audio.", so where our prompt goes may matter:
      system       instruction line, then our prompt (system message)
      system_last  our prompt, then the instruction line (system message)
      user         system = instruction line + tools; our prompt is text at the start of the guest's first turn"""
    tool_list = ("\nList of tools: <|tool_list_start|>[" + ", ".join(json.dumps(t) for t in tools)
                 + "]<|tool_list_end|>") if tools else ""
    if placement == "system":
        return f"{INTERLEAVED}\n{prompt}{tool_list}", None
    if placement == "system_last":
        return f"{prompt}\n{INTERLEAVED}{tool_list}", None
    return f"{INTERLEAVED}{tool_list}", f"{prompt}\n\nThe guest says:\n"


def parse_tool_calls(text: str) -> list[dict]:
    """LFM2 writes calls as <|tool_call_start|>[name(arg=value, ...)]<|tool_call_end|> (Python syntax);
    JSON ({"name":..,"arguments":..}) is accepted too. Returns [{"name", "arguments"}] or [] ."""
    calls = []
    for body in re.findall(r"<\|tool_call_start\|>(.*?)(?:<\|tool_call_end\|>|$)", text, flags=re.S):
        body = re.sub(r"<\|[^|]*\|>", "", body).strip()
        if not body:
            continue
        try:
            node = ast.parse(body if body.startswith("[") else f"[{body}]", mode="eval").body
            for c in getattr(node, "elts", []):
                if isinstance(c, ast.Call):
                    name = c.func.id if isinstance(c.func, ast.Name) else ast.unparse(c.func)
                    kwargs = {k.arg: ast.literal_eval(k.value) for k in c.keywords if k.arg}
                    calls.append({"name": name, "arguments": kwargs})
            continue
        except (SyntaxError, ValueError):
            pass
        try:
            obj = json.loads(body)
            for o in obj if isinstance(obj, list) else [obj]:
                calls.append({"name": o.get("name"), "arguments": o.get("arguments", o.get("parameters", {}))})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": None, "arguments": {}, "unparsed": body})
    return calls


def spoken(text: str) -> str:
    """What the guest hears: the text without tool-call sections and special tokens."""
    text = re.sub(r"<\|tool_call_start\|>.*?(<\|tool_call_end\|>|$)", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<\|[^|]*\|>", " ", text)).strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9: ]", " ", s.lower().replace("’", "'").replace("'", ""))).strip()


def add_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = float(np.mean(wav**2)) + 1e-12
    noise = rng.standard_normal(len(wav)).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))
    return np.clip(wav + noise, -1, 1).astype(np.float32)


# ----------------------------------------------------------------------------- scoring
def score(item: dict, rounds: list[dict]) -> dict:
    calls = [c for r in rounds for c in r["calls"]]
    first = calls[0]["name"] if calls else None
    said = norm(" ".join(spoken(r["text"]) for r in rounds))
    kind = item["kind"]
    res = {"tool_called": first, "n_calls": len(calls)}
    expect_ok = all(any(norm(alt) in said for alt in group) for group in item.get("expect", []))
    if kind == "tool":
        res["tool_ok"] = first in item["tools"]
        want = item.get("args", {}).get(first or "", {})
        got = calls[0]["arguments"] if calls else {}
        res["args_ok"] = bool(res["tool_ok"]) and all(str(got.get(k)).strip() == str(v) for k, v in want.items())
        res["pass"] = res["tool_ok"] and res["args_ok"]
    else:
        res["false_tool"] = first is not None
        res["expect_ok"] = expect_ok
        res["pass"] = (not res["false_tool"]) and expect_ok
    return res


# ----------------------------------------------------------------------------- the model
class LFM:
    def __init__(self, args):
        from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality

        self.ChatState, self.Mod = ChatState, LFMModality
        self.dev = args.device
        dtype = torch.bfloat16 if self.dev != "cpu" else torch.float32
        if self.dev == "cpu":  # liquid-audio's detokenizer calls .cuda(); keep it on CPU for smoke tests
            torch.nn.Module.cuda = lambda module, *a, **k: module
        t0 = now()
        self.proc = LFM2AudioProcessor.from_pretrained(HF_REPO, device=self.dev).eval()
        self.model = LFM2AudioModel.from_pretrained(HF_REPO, device=self.dev, dtype=dtype).eval()
        self.dtype = dtype
        try:
            self.mimi = self.proc.mimi.eval()
        except Exception as e:  # noqa: BLE001 - streaming decode is optional
            print(f"[lfm] Mimi streaming decoder unavailable ({e}); first-audible time not measured")
            self.mimi = None
        sync()
        self.load_s = now() - t0
        self.args = args

    def chat(self, system: str | tuple[str, str | None]):
        """A fresh conversation. `system` is a system message, or (system message, prefix) from setup_for."""
        system, prefix = system if isinstance(system, tuple) else (system, None)
        c = self.ChatState(self.proc, dtype=self.dtype)
        c.new_turn("system")
        c.add_text(system)
        c.end_turn()
        c.pending_prefix = prefix
        return c

    def user_turn(self, chat, wav: np.ndarray | None = None, sr: int = OUT_SR, text: str | None = None,
                  role: str = "user") -> None:
        chat.new_turn(role)
        if getattr(chat, "pending_prefix", None):
            chat.add_text(chat.pending_prefix)
            chat.pending_prefix = None
        if wav is not None:
            chat.add_audio(torch.from_numpy(wav.astype(np.float32))[None], sr)
        else:
            chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")

    @torch.no_grad()
    def prefill_breakdown(self, chat) -> dict:
        """Time the two prompt stages separately: audio encoder + embedding assembly, then the LM prefill."""
        sync()
        t0 = now()
        emb = self.model._prefill(text=chat.text, audio_in=chat.audio_in, audio_in_lens=chat.audio_in_lens,
                                  audio_out=chat.audio_out, modality_flag=chat.modality_flag)
        sync()
        t1 = now()
        self.model.lfm(inputs_embeds=emb, use_cache=True)
        sync()
        return {"positions": int(emb.shape[1]), "encode_s": t1 - t0, "lm_prefill_s": now() - t1}

    @torch.no_grad()
    def generate(self, chat, mode: str = "interleaved", stream_decode: bool = True, max_new: int | None = None,
                 audio_temperature: float | None = None, audio_top_k: int | None = None) -> dict:
        a = self.args
        fn = self.model.generate_interleaved if mode == "interleaved" else self.model.generate_sequential
        kw = dict(max_new_tokens=max_new or a.max_new_tokens,
                  audio_temperature=a.audio_temperature if audio_temperature is None else audio_temperature,
                  audio_top_k=a.audio_top_k if audio_top_k is None else audio_top_k)
        text_toks, frames, mod = [], [], []
        first_tok = first_text = first_audio = first_audible = None
        decode_s = []
        use_mimi = stream_decode and self.mimi is not None
        ctx = self.mimi.streaming(1) if use_mimi else contextlib.nullcontext()
        sync()
        t0 = now()
        with ctx:
            for t in fn(**chat, **kw):
                if first_tok is None:
                    sync()
                    first_tok = now() - t0
                if t.numel() == 1:
                    text_toks.append(t)
                    mod.append(self.Mod.TEXT)
                    if first_text is None:
                        first_text = now() - t0
                    continue
                frames.append(t)
                mod.append(self.Mod.AUDIO_OUT)
                if first_audio is None:
                    sync()
                    first_audio = now() - t0
                if use_mimi and not bool((t == EOS_AUDIO).any()):
                    s = now()
                    self.mimi.decode(t[None, :, None])
                    sync()
                    decode_s.append(now() - s)
                    if first_audible is None:
                        first_audible = now() - t0
        sync()
        total = now() - t0
        text = self.proc.text.decode(torch.cat(text_toks)) if text_toks else ""
        codes = [f for f in frames if not bool((f == EOS_AUDIO).any())]
        audio_s = len(codes) / 12.5
        gen_only = total - sum(decode_s)
        return {"text": text, "calls": parse_tool_calls(text), "n_text": len(text_toks), "n_frames": len(codes),
                "audio_s": audio_s, "first_token_s": first_tok, "first_text_s": first_text,
                "first_audio_frame_s": first_audio, "first_audible_s": first_audible, "total_s": total,
                "gen_only_s": gen_only, "rtf": gen_only / audio_s if audio_s else None,
                "mimi_frame_decode_s": mean(decode_s), "_text_toks": text_toks, "_frames": frames, "_mod": mod,
                "_codes": codes}

    def append(self, chat, r: dict) -> None:
        dev = chat.text.device
        text = torch.stack(r["_text_toks"], 1) if r["_text_toks"] else torch.empty((1, 0), dtype=torch.long, device=dev)
        audio = torch.stack(r["_frames"], 1) if r["_frames"] else torch.empty((8, 0), dtype=torch.long, device=dev)
        chat.append(text=text, audio_out=audio, modality_flag=torch.tensor([int(m) for m in r["_mod"]], dtype=torch.long))
        chat.end_turn()

    @torch.no_grad()
    def to_wav(self, codes: list) -> np.ndarray:
        if not codes:
            return np.zeros(0, dtype=np.float32)
        return self.proc.decode(torch.stack(codes, 1)[None]).float().cpu().numpy()[0]

    @torch.no_grad()
    def tts(self, text: str, voice: str) -> np.ndarray:
        c = self.chat(f"Perform TTS. Use the {voice} voice.")
        self.user_turn(c, text=text)
        frames = [t for t in self.model.generate_sequential(**c, max_new_tokens=1024, audio_temperature=0.8,
                                                            audio_top_k=64) if t.numel() > 1]
        return self.to_wav([f for f in frames if not bool((f == EOS_AUDIO).any())])

    @torch.no_grad()
    def asr(self, wav: np.ndarray, sr: int) -> str:
        c = self.chat("Perform ASR.")
        self.user_turn(c, wav=wav, sr=sr)
        toks = [t for t in self.model.generate_sequential(**c, max_new_tokens=256) if t.numel() == 1]
        return spoken(self.proc.text.decode(torch.cat(toks))) if toks else ""

    def respond(self, chat, item_text: str | None, wav: np.ndarray | None, mock: dict, mode: str,
                stream_decode: bool = True) -> list[dict]:
        """One guest turn incl. tool rounds: tool call -> canned result -> next assistant round."""
        self.user_turn(chat, wav=wav, sr=OUT_SR, text=item_text)
        rounds = []
        for i in range(self.args.max_tool_rounds):
            pre = self.prefill_breakdown(chat) if self.args.breakdown else {}
            r = self.generate(chat, mode, stream_decode)
            r.update({f"prefill_{k}": v for k, v in pre.items()})
            self.append(chat, r)
            rounds.append(r)
            if not r["calls"]:
                break
            results = [mock.get(c["name"], {"error": f"unknown tool {c['name']}"}) for c in r["calls"]]
            payload = json.dumps(results[0] if len(results) == 1 else results)
            chat.new_turn("tool")
            chat.add_text(f"<|tool_response_start|>{payload}<|tool_response_end|>")
            chat.end_turn()
            chat.new_turn("assistant")
        return rounds


# ----------------------------------------------------------------------------- tests
def public(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def run_set(lfm: LFM, name: str, items: list[dict], wavs: dict, system, mock: dict, out: Path,
            log, typed: bool = False, quiet: bool = False) -> list[dict]:
    rows = []
    for i, item in enumerate(items):
        chat = lfm.chat(system)
        wav = None if typed else wavs[i]
        rounds = lfm.respond(chat, item["q"] if typed else None, wav, mock, lfm.args.mode)
        sc = score(item, rounds)
        if not typed:
            sf.write(out / "wavs" / f"{name}_{i:02d}_reply.wav",
                     np.concatenate([lfm.to_wav(r["_codes"]) for r in rounds] or [np.zeros(1)]), OUT_SR)
        row = {"test": name, "i": i, "q": item["q"], "kind": item["kind"], **sc,
               "rounds": [public(r) for r in rounds]}
        rows.append(row)
        log(row)
        if quiet:
            continue
        r0 = rounds[0]
        calls = [c for r in rounds for c in r["calls"]]
        tool = f" -> {calls[0]['name']}{json.dumps(calls[0]['arguments'])}" if calls else ""
        print(f"  [{name} {i:02d}] {'PASS' if sc['pass'] else 'FAIL'} {item['kind']:8s} | {item['q']}\n"
              f"      said: {' / '.join(spoken(r['text']) for r in rounds)[:220]}{tool}\n"
              f"      first audio frame {ms(r0['first_audio_frame_s'])}, audible {ms(r0['first_audible_s'])}, "
              f"reply {r0['audio_s']:.1f} s audio in {ms(r0['total_s'])}")
    return rows


def acc_summary(rows: list[dict]) -> dict:
    def rate(xs):
        return round(100 * sum(bool(x) for x in xs) / len(xs), 1) if xs else None

    tool = [r for r in rows if r["kind"] == "tool"]
    notool = [r for r in rows if r["kind"] != "tool"]
    return {
        "n": len(rows), "pass_pct": rate([r["pass"] for r in rows]),
        "tool_selection_pct": rate([r["tool_ok"] for r in tool]),
        "tool_args_pct": rate([r["args_ok"] for r in tool]),
        "false_tool_call_pct": rate([r["false_tool"] for r in notool]),
        "facts_pct": rate([r["pass"] for r in rows if r["kind"] == "fact"]),
        "unknown_handled_pct": rate([r["pass"] for r in rows if r["kind"] == "unknown"]),
        "no_tool_actions_pct": rate([r["pass"] for r in rows if r["kind"] == "no_tool"]),
    }


def lat_summary(rows: list[dict]) -> dict:
    first = [r["rounds"][0] for r in rows]
    after_tool = [r["rounds"][1] for r in rows if len(r["rounds"]) > 1]

    def block(rs, key):
        xs = [x[key] for x in rs if x.get(key) is not None]
        return {"p50": pct(xs, 50), "p95": pct(xs, 95), "max": max(xs) if xs else None}

    s = {k: block(first, k) for k in ("first_token_s", "first_text_s", "first_audio_frame_s", "first_audible_s",
                                       "total_s", "prefill_encode_s", "prefill_lm_prefill_s")}
    s["after_tool_first_audible_s"] = block(after_tool, "first_audible_s")
    s["after_tool_first_audio_frame_s"] = block(after_tool, "first_audio_frame_s")
    s["prompt_positions_mean"] = mean([x.get("prefill_positions") for x in first])
    s["reply_audio_s_mean"] = mean([x["audio_s"] for x in first])
    s["rtf_mean"] = mean([x["rtf"] for x in first])
    s["mimi_frame_decode_s_mean"] = mean([x["mimi_frame_decode_s"] for x in first])
    return s


def test_concurrent(lfm: LFM, system, wavs: list, levels: list[int], log) -> list[dict]:
    """N conversations at once with the stock (batch-1) liquid-audio generate, one thread per call."""
    res = []
    for n in levels:
        barrier = threading.Barrier(n)
        out: list[dict | None] = [None] * n

        def worker(k: int) -> None:
            chat = lfm.chat(system)
            lfm.user_turn(chat, wav=wavs[k % len(wavs)], sr=OUT_SR)
            barrier.wait()
            out[k] = lfm.generate(chat, lfm.args.mode, stream_decode=False, max_new=lfm.args.concurrent_max_new)

        ts = [threading.Thread(target=worker, args=(k,)) for k in range(n)]
        t0 = now()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = now() - t0
        rs = [r for r in out if r]
        audio = sum(r["audio_s"] for r in rs)
        row = {"test": "concurrent", "n": n, "wall_s": wall,
               "first_audio_p50_s": pct([r["first_audio_frame_s"] for r in rs], 50),
               "first_audio_p95_s": pct([r["first_audio_frame_s"] for r in rs], 95),
               "rtf_mean": mean([r["rtf"] for r in rs]), "rtf_max": max(r["rtf"] or 0 for r in rs),
               "audio_s_per_wall_s": audio / wall if wall else None,
               "all_realtime": all((r["rtf"] or 9) < 1 for r in rs)}
        res.append(row)
        log(row)
        print(f"  {n:>3} calls at once | first audio p50 {ms(row['first_audio_p50_s'])} p95 {ms(row['first_audio_p95_s'])}"
              f" | real-time factor mean {row['rtf_mean'] or 0:.2f} max {row['rtf_max']:.2f}"
              f" | {'all real time' if row['all_realtime'] else 'NOT real time'}")
    return res


@torch.no_grad()
def test_batched(lfm: LFM, system, wav: np.ndarray, levels: list[int], steps: int, log) -> list[dict]:
    """Upper bound for a batched server. One batch of B streams: LM prefill of B prompts, then per decode step
    one backbone step for all B, one audio-frame (depthformer, 8 codebooks) pass for all B, and the
    detokenizer for 1 s of audio for all B. A stream needs per second of speech ~12.5 audio frames + the
    interleaved text tokens (6 text per 12 audio -> ~19 backbone steps) -> capacity = largest B whose
    per-second compute stays under 1 s."""
    m = lfm.model
    chat = lfm.chat(system)
    lfm.user_turn(chat, wav=wav, sr=OUT_SR)
    emb1 = m._prefill(text=chat.text, audio_in=chat.audio_in, audio_in_lens=chat.audio_in_lens,
                      audio_out=chat.audio_out, modality_flag=chat.modality_flag)
    C, Dd = m.codebooks, m.depthformer_dim
    frame = torch.randint(0, 2048, (C,), device=emb1.device)
    step_emb = m.audio_embedding(frame + m.codebook_offsets).sum(0)

    def depth_pass(h: torch.Tensor) -> torch.Tensor:  # h [B, D] -> codes [B, C] (greedy)
        x = m.depth_linear(h).view(h.shape[0], C, Dd)
        tok = torch.zeros_like(x[:, 0])
        cache, codes = None, []
        for i in range(C):
            o, cache = m.depthformer.forward_cached((x[:, i] + tok)[:, None, :], cache)
            nxt = m.depth_embeddings[i].get_logits(o[:, 0]).argmax(-1)
            codes.append(nxt)
            tok = m.depth_embeddings[i](nxt)
        return torch.stack(codes, 1)

    rows = []
    steps_per_s = 12.5 * (1 + lfm.args.text_per_audio)
    for b in levels:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            emb = emb1.expand(b, -1, -1).contiguous()
            sync()
            t0 = now()
            o = m.lfm(inputs_embeds=emb, use_cache=True)
            sync()
            prefill = now() - t0
            cache = o.past_key_values
            x = step_emb.expand(b, 1, -1).contiguous().to(emb.dtype)
            o = m.lfm(inputs_embeds=x, past_key_values=cache, use_cache=True)  # warm-up step
            cache = o.past_key_values
            sync()
            t0 = now()
            for _ in range(steps):
                o = m.lfm(inputs_embeds=x, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
            sync()
            t_step = (now() - t0) / steps
            h = o.last_hidden_state[:, -1]
            depth_pass(h)
            sync()
            t0 = now()
            for _ in range(5):
                codes = depth_pass(h)
            sync()
            t_depth = (now() - t0) / 5
            t_detok = None
            with contextlib.suppress(Exception):
                c = codes.clamp(0, 2047)[:, :, None].expand(b, C, 13).contiguous()
                lfm.proc.decode(c[:1])
                sync()
                t0 = now()
                lfm.proc.decode(c)
                sync()
                t_detok = now() - t0
            per_s = steps_per_s * t_step + 12.5 * t_depth + (t_detok or 0)
            mem = (torch.cuda.max_memory_allocated() - base) / 2**30 if torch.cuda.is_available() else None
            row = {"test": "batched", "b": b, "prompt_positions": int(emb.shape[1]), "prefill_s": prefill,
                   "backbone_step_s": t_step, "audio_frame_pass_s": t_depth, "detok_1s_audio_s": t_detok,
                   "compute_per_audio_s": per_s, "realtime": per_s < 1.0, "mem_gb": mem}
        except torch.OutOfMemoryError:
            row = {"test": "batched", "b": b, "oom": True}
        rows.append(row)
        log(row)
        if row.get("oom"):
            print(f"  B={b:>4} | out of memory")
            break
        print(f"  B={b:>4} | prefill {ms(row['prefill_s'])} | backbone step {1000 * row['backbone_step_s']:.1f} ms"
              f" | frame pass {1000 * row['audio_frame_pass_s']:.1f} ms | detok 1 s {ms(row['detok_1s_audio_s'])}"
              f" | compute per audio-second {row['compute_per_audio_s']:.2f} s"
              f" -> {'REAL TIME' if row['realtime'] else 'too slow'} | peak mem {row['mem_gb'] or 0:.1f} GB")
    return rows


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.expanduser("~/lfm_results"))
    p.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--tools-file", type=Path, default=DEFAULT_TOOLS)
    p.add_argument("--mock-results", type=Path, default=DEFAULT_MOCK)
    p.add_argument("--questions-file", type=Path, default=None,
                   help='JSON list like [{"q": "...", "kind": "fact|unknown|tool|no_tool|other", "expect": '
                        '[["alt1","alt2"]], "tools": ["name"], "args": {"name": {"arg": value}}}]')
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="real recordings instead of TTS: <audio-dir>/q00.wav, q01.wav ... in question order")
    p.add_argument("--tests", nargs="+",
                   default=["placement", "tts", "asr", "s2s", "text", "noisy", "multiturn", "concurrent", "batched",
                            "whisper"])
    p.add_argument("--mode", default="interleaved", choices=["interleaved", "sequential"])
    p.add_argument("--placement", default="auto", choices=["auto", *PLACEMENTS],
                   help="where our prompt goes (see setup_for). auto = the best one in the 'placement' test")
    p.add_argument("--noise-snr", type=float, default=10.0, help="SNR (dB) for the noisy test")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-tool-rounds", type=int, default=3)
    p.add_argument("--audio-temperature", type=float, default=1.0)
    p.add_argument("--audio-top-k", type=int, default=4)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--concurrent-max-new", type=int, default=300)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64, 128, 200])
    p.add_argument("--batch-steps", type=int, default=24)
    p.add_argument("--text-per-audio", type=float, default=0.5,
                   help="backbone text steps per audio frame (interleaved 6 text : 12 audio = 0.5)")
    p.add_argument("--no-breakdown", dest="breakdown", action="store_false",
                   help="skip the separate encoder / LM-prefill timing per turn")
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (quick check)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    log_f = open(out / "turns.jsonl", "w", encoding="utf-8")

    def log(row: dict) -> None:
        log_f.write(json.dumps(row, default=str) + "\n")
        log_f.flush()

    prompt = load_prompt(args.prompt_file)
    tools = load_tools(args.tools_file)
    mock = json.loads(args.mock_results.read_text(encoding="utf-8")) if args.mock_results else {}
    items = json.loads(args.questions_file.read_text(encoding="utf-8")) if args.questions_file else QUESTIONS
    if args.limit:
        items = items[: args.limit]
    S: dict = {"args": {k: str(v) for k, v in vars(args).items()}}
    system = setup_for(prompt, tools, "system" if args.placement == "auto" else args.placement)

    # ---- env
    print("== environment")
    lfm = LFM(args)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    S["env"] = {"gpu": gpu, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                "cuda": torch.version.cuda, "python": platform.python_version(), "load_s": lfm.load_s,
                "gpu_mem_after_load_gb": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None,
                "system_prompt_tokens": len(lfm.proc.text.encode(setup_for(prompt, tools, "system")[0],
                                                                 add_special_tokens=False))}
    print(f"  {gpu} | torch {torch.__version__} | model loaded in {lfm.load_s:.1f} s | "
          f"{S['env']['gpu_mem_after_load_gb'] or 0:.1f} GB | system prompt {S['env']['system_prompt_tokens']} tokens")
    with contextlib.suppress(Exception):  # warm-up (kernels, Mimi)
        c = lfm.chat(system)
        lfm.user_turn(c, text="Hello")
        lfm.generate(c, args.mode, max_new=40)

    # ---- where to put our prompt (typed questions: cheap, isolates prompt-following from hearing)
    if "placement" in args.tests:
        print("== prompt placement (typed questions)")
        S["placement"] = {}
        for pl in PLACEMENTS:
            rows = run_set(lfm, f"placement_{pl}", items, {}, setup_for(prompt, tools, pl), mock, out, log,
                           typed=True, quiet=True)
            S["placement"][pl] = acc_summary(rows)
            print(f"  {pl:12s} pass {S['placement'][pl]['pass_pct']}% | tools {S['placement'][pl]['tool_selection_pct']}%"
                  f" | facts {S['placement'][pl]['facts_pct']}% | false tool calls {S['placement'][pl]['false_tool_call_pct']}%")
        if args.placement == "auto":
            best = max(PLACEMENTS, key=lambda k: (S["placement"][k]["pass_pct"] or 0))
            system = setup_for(prompt, tools, best)
            S["placement_used"] = best
            print(f"  -> using '{best}' for the remaining tests")
    S.setdefault("placement_used", args.placement if args.placement != "auto" else "system")

    # ---- guest audio
    wavs: list[np.ndarray] = []
    mt_wavs: list[np.ndarray] = []
    lines = [it["q"] for it in items] + (MULTITURN if "multiturn" in args.tests else [])
    if args.audio_dir:
        for i in range(len(items)):
            w, sr = sf.read(args.audio_dir / f"q{i:02d}.wav", dtype="float32")
            w = w.mean(1) if w.ndim > 1 else w
            import torchaudio

            wavs.append(torchaudio.functional.resample(torch.from_numpy(w), sr, OUT_SR).numpy())
        tts_lines = MULTITURN if "multiturn" in args.tests else []
    else:
        tts_lines = lines
    if tts_lines and any(t in args.tests for t in ("tts", "asr", "s2s", "noisy", "multiturn", "concurrent", "batched")):
        print("== guest questions spoken with LFM's TTS voices" + (" (multi-turn only)" if args.audio_dir else ""))
        t0 = now()
        for k, line in enumerate(tts_lines):
            w = lfm.tts(line, VOICES[k % len(VOICES)])
            (wavs if (not args.audio_dir and k < len(items)) else mt_wavs).append(w)
        for k, w in enumerate(wavs):
            sf.write(out / "wavs" / f"q{k:02d}.wav", w, OUT_SR)
        print(f"  {len(tts_lines)} clips in {now() - t0:.0f} s")

    # ---- ASR (how well LFM hears the guest)
    if "asr" in args.tests and wavs:
        import jiwer

        print("== LFM's own transcript of each question")
        hyps = [lfm.asr(w, OUT_SR) for w in wavs]
        refs = [it["q"] for it in items]
        S["asr_wer_pct"] = round(100 * jiwer.wer([norm(r) for r in refs], [norm(h) or "-" for h in hyps]), 1)
        for r, h in zip(refs, hyps):
            log({"test": "asr", "ref": r, "hyp": h})
        print(f"  word error rate {S['asr_wer_pct']}%")

    # ---- main tests
    if "s2s" in args.tests:
        print(f"== speech in -> our prompt + tools -> speech out ({args.mode})")
        rows = run_set(lfm, "s2s", items, dict(enumerate(wavs)), system, mock, out, log)
        S["s2s"] = {"accuracy": acc_summary(rows), "latency": lat_summary(rows)}
    if "text" in args.tests:
        print("== control: the same questions typed")
        rows = run_set(lfm, "text", items, {}, system, mock, out, log, typed=True)
        S["text"] = {"accuracy": acc_summary(rows), "latency": lat_summary(rows)}
    if "noisy" in args.tests and wavs:
        print(f"== noisy questions (SNR {args.noise_snr:g} dB)")
        noisy = {i: add_noise(w, args.noise_snr, i) for i, w in enumerate(wavs)}
        rows = run_set(lfm, "noisy", items, noisy, system, mock, out, log)
        S["noisy"] = {"accuracy": acc_summary(rows)}

    if "multiturn" in args.tests and mt_wavs:
        print("== one 6-turn conversation (history grows; liquid-audio re-reads it every turn)")
        chat = lfm.chat(system)
        S["multiturn"] = []
        for k, (line, w) in enumerate(zip(MULTITURN, mt_wavs)):
            rounds = lfm.respond(chat, None, w, mock, args.mode)
            r0 = rounds[0]
            row = {"turn": k + 1, "guest": line, "said": " / ".join(spoken(r["text"]) for r in rounds),
                   "calls": [c for r in rounds for c in r["calls"]], "positions": r0.get("prefill_positions"),
                   "lm_prefill_s": r0.get("prefill_lm_prefill_s"), "first_audible_s": r0["first_audible_s"],
                   "first_audio_frame_s": r0["first_audio_frame_s"]}
            S["multiturn"].append(row)
            log({"test": "multiturn", **row})
            print(f"  turn {k + 1} | {row['positions']} positions, prefill {ms(row['lm_prefill_s'])}, first audible "
                  f"{ms(row['first_audible_s'])} | {line}\n      said: {row['said'][:200]}"
                  f"{' -> ' + json.dumps(row['calls']) if row['calls'] else ''}")

    if "concurrent" in args.tests and wavs and torch.cuda.is_available():
        print("== simultaneous calls with the stock liquid-audio code (no batching)")
        S["concurrent"] = test_concurrent(lfm, system, wavs, args.concurrency, log)
    if "batched" in args.tests and wavs and torch.cuda.is_available():
        print("== batched server upper bound (all streams speaking at once)")
        rows = test_batched(lfm, system, wavs[0], args.batch, args.batch_steps, log)
        ok = [r["b"] for r in rows if r.get("realtime")]
        S["batched"] = {"rows": rows, "max_realtime_streams_tested": max(ok) if ok else 0}

    # ---- intelligibility of the replies
    if "whisper" in args.tests and "s2s" in S:
        try:
            import jiwer
            import torchaudio
            from transformers import pipeline

            print(f"== intelligibility of the spoken replies ({args.whisper})")
            del lfm.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            asr = pipeline("automatic-speech-recognition", model=args.whisper,
                           torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
                           device=0 if args.device != "cpu" else -1)
            refs, hyps, qrefs, qhyps = [], [], [], []
            for line in open(out / "turns.jsonl", encoding="utf-8"):
                row = json.loads(line)
                if row.get("test") != "s2s":
                    continue
                said = " ".join(spoken(r["text"]) for r in row["rounds"])
                w, sr = sf.read(out / "wavs" / f"s2s_{row['i']:02d}_reply.wav", dtype="float32")
                if said and len(w) > sr // 4:
                    w16 = torchaudio.functional.resample(torch.from_numpy(w), sr, 16_000).numpy()
                    refs.append(norm(said))
                    hyps.append(norm(asr({"raw": w16, "sampling_rate": 16_000})["text"]) or "-")
                qw, qsr = sf.read(out / "wavs" / f"q{row['i']:02d}.wav", dtype="float32") \
                    if (out / "wavs" / f"q{row['i']:02d}.wav").exists() else (None, None)
                if qw is not None:
                    q16 = torchaudio.functional.resample(torch.from_numpy(qw), qsr, 16_000).numpy()
                    qrefs.append(norm(row["q"]))
                    qhyps.append(norm(asr({"raw": q16, "sampling_rate": 16_000})["text"]) or "-")
            S["reply_intelligibility_wer_pct"] = round(100 * jiwer.wer(refs, hyps), 1) if refs else None
            S["question_audio_wer_pct"] = round(100 * jiwer.wer(qrefs, qhyps), 1) if qrefs else None
            print(f"  replies: Whisper word error rate {S['reply_intelligibility_wer_pct']}% "
                  f"(lower = clearer) | guest questions: {S['question_audio_wer_pct']}%")
        except Exception as e:  # noqa: BLE001
            print(f"  skipped ({e})")

    if torch.cuda.is_available():
        S["env"]["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str), encoding="utf-8")
    md = report(S)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"files: {out}/summary.md, summary.json, turns.jsonl, wavs/")


def report(S: dict) -> str:
    e = S["env"]
    L = [f"# LFM2.5-Audio-1.5B on the hotel concierge task (no fine-tuning)\n",
         f"GPU {e['gpu']} | torch {e['torch']} | load {e['load_s']:.1f} s | weights {e['gpu_mem_after_load_gb'] or 0:.1f} GB"
         f" | peak {e.get('gpu_peak_mem_gb') or 0:.1f} GB | system prompt + tools {e['system_prompt_tokens']} tokens\n"]
    acc_rows = [(f"typed, prompt in {k}", a) for k, a in S.get("placement", {}).items()]
    acc_rows += [(k, S[k]["accuracy"]) for k in ("s2s", "text", "noisy") if k in S]
    if acc_rows:
        L.append(f"## Accuracy (%) -- prompt placement used for s2s/text/noisy: {S.get('placement_used')}\n")
        L.append("| input | all pass | tool chosen right | tool args right | false tool calls (lower=better) | prompt facts | unknown facts | no-tool actions |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, a in acc_rows:
            L.append(f"| {k} | {a['pass_pct']} | {a['tool_selection_pct']} | {a['tool_args_pct']} | "
                     f"{a['false_tool_call_pct']} | {a['facts_pct']} | {a['unknown_handled_pct']} | {a['no_tool_actions_pct']} |")
        if "s2s" in S and "text" in S and S["text"]["accuracy"]["pass_pct"] is not None:
            L.append(f"\nSpeech-vs-text gap: {S['text']['accuracy']['pass_pct'] - S['s2s']['accuracy']['pass_pct']:+.1f} points"
                     " (text minus speech; the cost of hearing instead of reading)\n")
    for k in ("asr_wer_pct", "reply_intelligibility_wer_pct", "question_audio_wer_pct"):
        if S.get(k) is not None:
            L.append(f"- {k}: {S[k]}%")
    if "s2s" in S:
        lt = S["s2s"]["latency"]
        L.append("\n## Latency, one call at a time (from the end of the guest's audio)\n")
        L.append("| step | p50 | p95 | max |")
        L.append("|---|---|---|---|")
        for key, name in [("prefill_encode_s", "audio encoder + prompt assembly"), ("prefill_lm_prefill_s", "LM prompt prefill"),
                          ("first_text_s", "first text token"), ("first_audio_frame_s", "first audio frame"),
                          ("first_audible_s", "first audible audio (frame decoded)"), ("total_s", "whole reply"),
                          ("after_tool_first_audible_s", "after a tool result: first audible audio")]:
            b = lt[key]
            L.append(f"| {name} | {ms(b['p50'])} | {ms(b['p95'])} | {ms(b['max'])} |")
        L.append(f"\nprompt positions (mean) {lt['prompt_positions_mean'] or 0:.0f} | reply audio (mean) "
                 f"{lt['reply_audio_s_mean'] or 0:.1f} s | real-time factor (mean, compute per audio second) "
                 f"{lt['rtf_mean'] or 0:.3f} | Mimi frame decode {ms(lt['mimi_frame_decode_s_mean'])}")
    if S.get("multiturn"):
        L.append("\n## Multi-turn (history grows)\n")
        L.append("| turn | positions | LM prefill | first audible | guest | said |")
        L.append("|---|---|---|---|---|---|")
        for r in S["multiturn"]:
            L.append(f"| {r['turn']} | {r['positions']} | {ms(r['lm_prefill_s'])} | {ms(r['first_audible_s'])} | "
                     f"{r['guest']} | {r['said'][:90]} |")
    if S.get("concurrent"):
        L.append("\n## Simultaneous calls, stock liquid-audio code (no batching)\n")
        L.append("| calls | first audio p50 | first audio p95 | real-time factor mean / max | all real time |")
        L.append("|---|---|---|---|---|")
        for r in S["concurrent"]:
            L.append(f"| {r['n']} | {ms(r['first_audio_p50_s'])} | {ms(r['first_audio_p95_s'])} | "
                     f"{r['rtf_mean'] or 0:.2f} / {r['rtf_max']:.2f} | {r['all_realtime']} |")
    if S.get("batched"):
        L.append("\n## Batched server upper bound (all streams speaking)\n")
        L.append("| streams | prefill of all prompts | backbone step | audio-frame pass | detok 1 s | compute per audio-second | real time | peak mem |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in S["batched"]["rows"]:
            if r.get("oom"):
                L.append(f"| {r['b']} | out of memory | | | | | | |")
                continue
            L.append(f"| {r['b']} | {ms(r['prefill_s'])} | {1000 * r['backbone_step_s']:.1f} ms | "
                     f"{1000 * r['audio_frame_pass_s']:.1f} ms | {ms(r['detok_1s_audio_s'])} | "
                     f"{r['compute_per_audio_s']:.2f} s | {r['realtime']} | {r['mem_gb'] or 0:.1f} GB |")
        L.append(f"\nLargest tested batch that stays real time: {S['batched']['max_realtime_streams_tested']} streams "
                 "speaking at once (with ~40% of a call spent speaking, roughly 2.5x that many connected calls). "
                 "Upper bound: no scheduler, sampling or network overhead.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
