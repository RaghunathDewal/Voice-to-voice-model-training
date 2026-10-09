"""Make generated hotel conversations sound like real people, with an open LLM (Qwen, Apache-2.0) on vLLM.

The template generator (hotel_ghb / hotel_v3) decides WHAT happens: facts in the prompt, which tool is called
with which arguments, tool results, the expected answer. This script only changes HOW it is worded:

  questions  the guest's words, in varied styles (casual, polite, indirect, Indian / British English, fillers)
  replies    the assistant's spoken replies, warm and natural, one or two sentences
  prompts    the system prompt re-laid-out (email, FAQ, table-like lines, JSON, terse notes ...)

Tool calls and tool results are never touched. Every rewrite is checked and the original is kept when a check
fails: all numbers, times, passwords, order / issue numbers and names must survive, no new numbers may appear,
tool-turn questions must still name the item / problem, replies stay short and plain. Conversations that were
changed get "-q" appended to their id, so their audio is generated fresh by the voice stage.

Standalone (stdlib + vllm), runs inside a vLLM container; see scripts/lfm_qwen_rewrite.sh:

    python s2s/data/llm_rewrite.py --in ~/lfm_ft2/convs_train.jsonl --out ~/lfm_ft2/convs_train.jsonl
    python s2s/data/llm_rewrite.py --in convs.jsonl --out /tmp/x.jsonl --dry-run   # plumbing test, no GPU
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # MoE, 3B active: fast; Apache-2.0
TOOL_KINDS = {"order", "order_missing", "issue", "order_id", "orders", "issues", "browse", "v3_tool"}
NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
             "thirty": "30", "noon": "12", "midnight": "12"}
STOP = set("""a an the to of and or for in on at by with from is are was be been it its this that these those i you we
they he she my your our their me us can could would will shall should may might please hi hello hey thanks thank
excuse sorry just also some any get got have has had do does did send bring want need like would there here what
when where which how who whom why not no yes so um uh okay ok well really very more much many another extra""".split())

Q_STYLES = ["casual and relaxed", "very polite", "short and direct", "indirect, hinting at the need",
            "Indian English", "British English", "American English with a filler word like 'um' or 'so'",
            "slightly rambling, as spoken on the phone", "tired and a bit annoyed", "friendly, starts with a greeting"]
P_LAYOUTS = ["an email from the property manager to the assistant", "a FAQ with one question and answer per line",
             "short paragraphs per topic", "a JSON object", "terse notes with abbreviations",
             "lines like 'Topic | details'", "a welcome booklet page", "headed sections in plain text"]

SYS_Q = ("You rewrite what a hotel guest says to a voice assistant. Keep the exact meaning and request, and keep "
         "every number, name, item and problem. Vary the wording naturally in the requested style. Never add new "
         "details, items, numbers or requests. Reply with only the rewritten words of the guest.")
SYS_R = ("You rewrite what a hotel voice concierge says out loud. Keep every fact exactly: numbers, times, prices, "
         "names, passwords, order and issue numbers. One or two short spoken sentences, warm and natural; no lists, "
         "no markdown, no emojis. Do not add any new facts, offers, numbers or questions that were not there. "
         "Reply with only the rewritten words.")
SYS_P = ("You re-format the system prompt of a hotel voice assistant. Keep every rule, fact, value, name and "
         "number exactly, including the rules about tools and who to hand over to, but present them in a different, "
         "realistic layout. Do not add facts. Output only the new prompt.")


# ----------------------------------------------------------------------------- checks
def _norm_nums(text: str) -> str:
    return re.sub(r"\b(" + "|".join(NUM_WORDS) + r")\b", lambda m: NUM_WORDS[m.group(1).lower()], text,
                  flags=re.I)


def digit_tokens(text: str) -> set[str]:
    """Tokens that carry a value: '7:30', '9750', 'heron877', '15' (number words count as digits)."""
    toks = re.findall(r"[A-Za-z]*\d[\w:.\-/]*", _norm_nums(text))
    return {t.strip(".:-/").lower() for t in toks if t.strip(".:-/")}


def names(text: str) -> set[str]:
    """Capitalised words that are not sentence starts (product names, places, persona, network names)."""
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        words = re.findall(r"[A-Za-z][\w'-]*", sent)
        for w in words[1:]:
            if w[0].isupper() and w not in ("I", "AM", "PM", "OK") and not w.startswith("I'") and len(w) > 1:
                out.add(w.lower())
    return out


def content_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOP}


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^(layout:[^\n]*\n)?(prompt:\s*\n)?", "", text, flags=re.I)
    text = re.sub(r"^(guest|concierge|assistant|rewritten[^:]*):\s*", "", text, flags=re.I).strip()
    return text.strip().strip('"').strip("“”").strip()


def ok_question(orig: str, new: str, turn: dict) -> str | None:
    if not new or len(new) > 3 * len(orig) + 60 or "\n" in new.strip():
        return "shape"
    if digit_tokens(orig) - digit_tokens(new):
        return "lost value"
    if digit_tokens(new) - digit_tokens(orig):
        return "new value"
    if turn.get("kind") in TOOL_KINDS | {"cannot"}:
        keys = set()
        for s in turn["assistant"]:
            for v in (s.get("call", {}).get("arguments") or {}).values():
                if isinstance(v, str):
                    keys |= content_words(v)
        keys = (keys & content_words(orig)) or content_words(orig)
        if keys and not (keys & content_words(new)):
            return "lost item"
    return None


def ok_reply(orig: str, new: str) -> str | None:
    if not new or "\n" in new.strip() or any(c in new for c in "*#•[]{}") or len(new.split()) > 45:
        return "shape"
    if digit_tokens(orig) - digit_tokens(new):
        return "lost value"
    if digit_tokens(new) - digit_tokens(orig):
        return "new value"
    lost = {n for n in names(orig) if n not in new.lower()}
    if lost:
        return "lost name"
    return None


def ok_prompt(orig: str, new: str) -> str | None:
    if not new or not 0.5 * len(orig) < len(new) < 2.0 * len(orig):
        return "length"
    if digit_tokens(orig) - digit_tokens(new):
        return "lost value"
    if digit_tokens(new) - digit_tokens(orig):
        return "new value"
    low = new.lower()
    if {n for n in names(orig) if n not in low}:
        return "lost name"
    if "List of tools" in new or "<|" in new:
        return "tool block"
    return None


# ----------------------------------------------------------------------------- generation
class FakeLLM:
    """--dry-run: no model; returns the input slightly changed so the plumbing and checks can be tested."""

    def chat(self, batch, params=None, **kw):
        class O:
            def __init__(self, t):
                self.outputs = [type("C", (), {"text": t})()]

        return [O(m[-1]["content"].rsplit("said: ", 1)[-1].replace("please", "kindly").replace("Sure", "Of course"))
                for m in batch]


def run_llm(llm, batch: list[list[dict]], max_tokens: int, temperature: float):
    if not batch:
        return []
    try:
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    except ImportError:
        params = None
    try:
        outs = llm.chat(batch, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    except TypeError:  # older vLLM or the fake
        outs = llm.chat(batch, params)
    return [clean(o.outputs[0].text) for o in outs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", nargs="+", required=True, help="one or more conversation .jsonl files")
    p.add_argument("--out", default=None, help="output file (one input only); default: rewrite in place, keeping "
                                                "a .orig.jsonl backup")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--frac-question", type=float, default=0.8)
    p.add_argument("--frac-reply", type=float, default=0.7)
    p.add_argument("--frac-prompt", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    srcs = [Path(x) for x in args.inp]
    if args.out and len(srcs) > 1:
        raise SystemExit("--out works with one --in file; omit it to rewrite several files in place")
    outs = [Path(args.out)] if args.out else srcs
    convs, owner = [], []
    for fi, src in enumerate(srcs):
        rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
        convs += rows
        owner += [fi] * len(rows)
        if outs[fi].resolve() == src.resolve():
            shutil.copy2(src, src.with_suffix(".orig.jsonl"))

    # jobs: (kind, conv index, turn index, step index, original text)
    jobs = []
    for ci, c in enumerate(convs):
        if rng.random() < args.frac_prompt:
            jobs.append(("prompt", ci, None, None, c["system"]))
        for ti, t in enumerate(c["turns"]):
            if rng.random() < args.frac_question:
                jobs.append(("question", ci, ti, None, t["user"]))
            for si, s in enumerate(t["assistant"]):
                if "say" in s and rng.random() < args.frac_reply:
                    jobs.append(("reply", ci, ti, si, s["say"]))

    if args.dry_run:
        llm = FakeLLM()
    else:
        from vllm import LLM

        llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len, seed=args.seed)

    t0 = time.time()
    results = {}
    for kind, sys_msg, max_tok in (("question", SYS_Q, 120), ("reply", SYS_R, 160), ("prompt", SYS_P, 2500)):
        mine = [j for j in jobs if j[0] == kind]
        batch = []
        for _, ci, ti, si, text in mine:
            if kind == "question":
                user = f"Style: {rng.choice(Q_STYLES)}\nGuest said: {text}"
            elif kind == "reply":
                user = f"Concierge said: {text}"
            else:
                user = f"Layout: {rng.choice(P_LAYOUTS)}\nPrompt:\n{text}"
            batch.append([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}])
        print(f"[rewrite] {kind}: {len(batch)} items", flush=True)
        for job, new in zip(mine, run_llm(llm, batch, max_tok, args.temperature)):
            results[job] = new

    stats: dict[str, dict[str, int]] = {}
    samples = []
    changed = set()
    for job, new in results.items():
        kind, ci, ti, si, orig = job
        c = convs[ci]
        if kind == "question":
            why = ok_question(orig, new, c["turns"][ti])
        elif kind == "reply":
            why = ok_reply(orig, new)
        else:
            why = ok_prompt(orig, new)
        st = stats.setdefault(kind, {"accepted": 0})
        if why is None and new != orig:
            st["accepted"] += 1
            changed.add(ci)
            if kind == "question":
                c["turns"][ti]["user"] = new
            elif kind == "reply":
                c["turns"][ti]["assistant"][si]["say"] = new
            else:
                c["system"] = new
            if len(samples) < 60 and (kind != "prompt" or sum(s[0] == "prompt" for s in samples) < 3):
                samples.append((kind, orig, new))
        else:
            st[why or "unchanged"] = st.get(why or "unchanged", 0) + 1
    for ci in changed:
        convs[ci]["id"] += "-q"
        convs[ci]["rewritten"] = True

    for fi, out in enumerate(outs):
        with open(out, "w", encoding="utf-8") as f:
            for c, o in zip(convs, owner):
                if o == fi:
                    f.write(json.dumps(c) + "\n")
    sample_file = outs[0].with_suffix(".samples.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        for kind, a, b in samples:
            f.write(f"=== {kind}\n--- before\n{a}\n--- after\n{b}\n\n")
    print(f"[rewrite] {len(changed)}/{len(convs)} conversations changed in {time.time() - t0:.0f} s -> "
          + ", ".join(str(o) for o in outs))
    for kind, st in stats.items():
        tot = sum(st.values())
        print(f"   {kind:8s} accepted {st['accepted']}/{tot} | rejected: "
              + ", ".join(f"{k} {v}" for k, v in st.items() if k != "accepted"))
    print(f"   examples: {sample_file}")


if __name__ == "__main__":
    main()
