"""Build the text-only fine-tuning set for the thinker LoRA (no audio).

Mixes three commercially usable sources into one JSONL of chat conversations:

  * hotel_v2 (ours)          tool calls, the spoken reply after the tool result, reservation
                             answers, small talk, out-of-scope requests and follow-up turns
  * hotel_v3 (ours, --version 3)  the same, but each conversation has its own property facts in the
                             system prompt and a random subset of a larger tool pool: facts are
                             answered from the prompt and only listed tools are called
  * Salesforce xLAM 60k      general function calling (CC BY 4.0, gated: accept on its HF page)
  * NousResearch Hermes v1   general function calling (Apache 2.0); only the tool-call turn is
                             kept (its final answers are long markdown, wrong for a voice agent)

    python -m s2s.prep.thinker_sft_data --config configs/small.yaml --hotel 16000 --xlam 4000 --hermes 1500
    python -m s2s.prep.thinker_sft_data --config configs/small.yaml --version 3 --hotel 20000 --hotel-v2 2000 \
        --xlam 4000 --hermes 1500

Writes <data_dir>/manifests/thinker_sft_train.jsonl and thinker_sft_valid.jsonl. Each row is
{"id", "source", "messages": [...], "tools": [...] | null}; messages use the chat-template
format (assistant tool calls in "tool_calls", tool results as role "tool").
Hotel rows use seeds 0 (train), 1 (valid) and 3 (--extra-wakeup); the hotel accuracy eval uses seed 2.
"""

from __future__ import annotations

import json
import os
import random
import re

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HotelBackend, row_system, tools_by_name
from s2s.data.hotel_v2 import generate_examples_v2
from s2s.data.hotel_v3 import GenericBackend, generate_examples_v3
from s2s.utils import write_jsonl

GENERIC_SYSTEM = ("You are a helpful assistant. When a request needs one of the available functions, call it "
                  "with the right arguments; otherwise answer briefly.")


def _call_msg(calls: list[dict]) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                           for c in calls]}


def hotel_conversation(row: dict, system_prompt: str) -> dict:
    """hotel_v2 / v3 row -> messages. Tool rows also get the tool result and the spoken confirmation."""
    tools = tools_by_name(row.get("tools"))
    msgs = [{"role": "system", "content": row_system(row, system_prompt)}]
    msgs += [dict(m) for m in row.get("history") or []]
    msgs.append({"role": "user", "content": row["text"]})
    if row.get("tool_calls"):
        backend = GenericBackend(tools) if row.get("system") else \
            HotelBackend(json.loads(row["context"].split("\n", 1)[1]))
        msgs.append(_call_msg(row["tool_calls"]))
        for call in row["tool_calls"]:
            msgs.append({"role": "tool", "content": json.dumps(backend.execute(call))})
        msgs.append({"role": "assistant", "content": row["reply_after_tool"]})
    else:
        msgs.append({"role": "assistant", "content": row["reply"]})
    source = "hotel_v3" if row.get("system") else "hotel_v2"
    return {"id": row["id"], "source": source, "messages": msgs, "tools": tools}


_XLAM_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
               "number": "number", "bool": "boolean", "boolean": "boolean", "list": "array", "dict": "object",
               "tuple": "array", "set": "array"}


def xlam_tool(spec: dict) -> dict:
    """xLAM's compact parameter format -> an OpenAI-style function schema."""
    props, required = {}, []
    for name, p in (spec.get("parameters") or {}).items():
        raw = str(p.get("type", "str"))
        base = re.split(r"[\[,\s]", raw.strip())[0].lower()
        props[name] = {"type": _XLAM_TYPES.get(base, "string"), "description": p.get("description", "")}
        if "optional" not in raw.lower() and "default" not in p:
            required.append(name)
    return {"type": "function", "function": {"name": spec["name"], "description": spec.get("description", ""),
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


def xlam_rows(path: str, n: int, rng: random.Random) -> list[dict]:
    data = json.load(open(path))
    rng.shuffle(data)
    rows = []
    for d in data:
        if len(rows) >= n:
            break
        try:
            tools = [xlam_tool(t) for t in json.loads(d["tools"])]
            calls = json.loads(d["answers"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not calls or len(tools) > 6:  # long tool lists only cost sequence length
            continue
        rows.append({"id": f"xlam_{d['id']}", "source": "xlam", "tools": tools,
                     "messages": [{"role": "system", "content": GENERIC_SYSTEM},
                                  {"role": "user", "content": d["query"]}, _call_msg(calls)]})
    return rows


_HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def hermes_rows(path: str, n: int, rng: random.Random) -> list[dict]:
    data = json.load(open(path))
    rng.shuffle(data)
    rows = []
    for d in data:
        if len(rows) >= n:
            break
        conv = d.get("conversations") or []
        human = next((m["value"] for m in conv if m["from"] == "human"), None)
        gpt = next((m["value"] for m in conv if m["from"] == "gpt"), "")
        try:
            tools = json.loads(d["tools"]) if isinstance(d.get("tools"), str) else d.get("tools")
            calls = [json.loads(c) for c in _HERMES_CALL.findall(gpt)]
        except json.JSONDecodeError:
            continue
        if not human or not tools or not calls or any("name" not in c for c in calls):
            continue
        rows.append({"id": f"hermes_{d['id']}", "source": "hermes", "tools": tools,
                     "messages": [{"role": "system", "content": GENERIC_SYSTEM}, {"role": "user", "content": human},
                                  _call_msg(calls)]})
    return rows


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--hotel", type=int, default=16000, help="hotel training conversations (of --version)")
    p.add_argument("--version", type=int, default=2, choices=[2, 3], help="hotel generator, see the module docstring")
    p.add_argument("--hotel-v2", type=int, default=0,
                   help="with --version 3: also mix in this many v2 conversations (fixed hotel prompt and tools)")
    p.add_argument("--hotel-valid", type=int, default=400)
    p.add_argument("--xlam", type=int, default=4000)
    p.add_argument("--hermes", type=int, default=1500)
    p.add_argument("--extra-wakeup", type=int, default=0,
                   help="extra wake-up-call conversations (spoken times: 'quarter to six', 'half past five', ...)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = config_from_args(args)
    from huggingface_hub import hf_hub_download

    rng = random.Random(args.seed)
    system = cfg.thinker.system_prompt
    gen = generate_examples_v3 if args.version == 3 else generate_examples_v2
    train = [hotel_conversation(r, system) for r in gen(args.hotel, seed=0)]
    valid = [hotel_conversation(r, system) for r in gen(args.hotel_valid, seed=1)]
    if args.version == 3 and args.hotel_v2:  # seed 4: not v2's train / valid / eval seeds
        train += [hotel_conversation(r, system) for r in generate_examples_v2(args.hotel_v2, seed=4)]
    if args.extra_wakeup:  # seed 3: not the eval rows (seed 2)
        wake = [r for r in generate_examples_v2(args.extra_wakeup * 8, seed=3)
                if r.get("tool_calls") and r["tool_calls"][0]["name"] == "schedule_wakeup_call"]
        train += [hotel_conversation(r, system) for r in wake[: args.extra_wakeup]]
    if args.xlam:
        path = hf_hub_download("Salesforce/xlam-function-calling-60k", "xlam_function_calling_60k.json",
                               repo_type="dataset")
        train += xlam_rows(path, args.xlam, rng)
    if args.hermes:
        path = hf_hub_download("NousResearch/hermes-function-calling-v1", "func-calling-singleturn.json",
                               repo_type="dataset")
        train += hermes_rows(path, args.hermes, rng)
    rng.shuffle(train)
    out = os.path.join(cfg.paths.data_dir, "manifests")
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "thinker_sft_train.jsonl"), train)
    write_jsonl(os.path.join(out, "thinker_sft_valid.jsonl"), valid)
    counts = {s: sum(r["source"] == s for r in train) for s in ("hotel_v2", "hotel_v3", "xlam", "hermes")}
    print(f"train {len(train)} rows {counts}, valid {len(valid)} hotel rows -> {out}")


if __name__ == "__main__":
    main()
