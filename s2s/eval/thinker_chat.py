"""Talk to the thinker by typing, with your own system prompt and tool list (no audio).

    python -m s2s.eval.thinker_chat --config configs/small.yaml --model checkpoints/thinker_merged_v3 \
        --system-prompt-file my_property.txt --tools order_product,create_issue
    # or a fixed list of guest lines, one per line (each starts a fresh conversation unless --multi-turn)
    python -m s2s.eval.thinker_chat ... --questions my_questions.txt
    # your own tool schemas and a canned (fake) response per tool, so multi-step flows can be tried
    python -m s2s.eval.thinker_chat ... --tools-file my_tools.json --mock-results my_mock_results.json

Prints the reply, or the tool call, the mock tool result and the spoken confirmation, exactly as the
voice agent would produce them (greedy decoding, same prompt layout). Tool results come from a mock
backend that always succeeds, so this checks *what the model decides*, not your real APIs.
"""

from __future__ import annotations

import json

import torch

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel_v3 import GenericBackend, select_tools
from s2s.eval.text_tools import extract_calls
from s2s.models.thinker import Thinker
from s2s.utils import resolve_device, resolve_dtype


class MockBackend(GenericBackend):
    """Canned responses per tool name from a JSON file ({"browse_products": {...}, ...}); other listed tools
    fall back to GenericBackend's echo. Calls to tools that are not in the tool list still get an error."""

    def __init__(self, tools: list | None, canned: dict):
        super().__init__(tools)
        self.canned = canned

    def execute(self, call: dict) -> dict:
        if call.get("name") in self.names and call["name"] in self.canned:
            return self.canned[call["name"]]
        return super().execute(call)


def load_prompt(path: str) -> str:
    """A system prompt file. `{{` / `}}` (template escaping) become single braces."""
    with open(path, encoding="utf-8") as f:
        return f.read().replace("{{", "{").replace("}}", "}").strip()


@torch.no_grad()
def generate(thinker: Thinker, messages: list[dict], tools: list | None, max_new: int = 150) -> str:
    text = thinker.prompts._render(messages, tools, add_generation_prompt=True)
    ids = torch.tensor([thinker.prompts.ids(text)], device=thinker.device)
    out = thinker.model.generate(ids, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                 eos_token_id=sorted(thinker.eos_ids), pad_token_id=thinker.pad_id)
    return thinker.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()


def respond(thinker: Thinker, messages: list[dict], tools: list | None, backend: GenericBackend,
            max_rounds: int = 3) -> list[str]:
    """One guest turn: appends the assistant / tool messages to `messages`, returns printable lines."""
    lines = []
    for _ in range(max_rounds):
        out = generate(thinker, messages, tools)
        calls = [c for c in extract_calls(out) if c.get("name")]
        if not calls:
            messages.append({"role": "assistant", "content": out})
            lines.append(f"  AGENT: {out}")
            return lines
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}} for c in calls]})
        for c in calls:
            result = backend.execute(c)
            lines.append(f"  TOOL CALL: {c['name']}({json.dumps(c.get('arguments') or {})}) -> {json.dumps(result)}")
            messages.append({"role": "tool", "content": json.dumps(result)})
    lines.append("  (stopped after the maximum number of tool rounds)")
    return lines


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default=None, help="thinker (default: thinker.model from the config)")
    p.add_argument("--system-prompt-file", required=True)
    p.add_argument("--tools", default="none", help="none | comma-separated built-in names (see ws_live --tools)")
    p.add_argument("--tools-file", default=None, help="JSON list of your own tool schemas")
    p.add_argument("--mock-results", default=None, help="JSON: tool name -> the (fake) response it returns")
    p.add_argument("--questions", default=None, help="file with one guest line per line; default: type them")
    p.add_argument("--multi-turn", action="store_true", help="with --questions: keep one conversation")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    thinker = Thinker(args.model or cfg.thinker.model, device, resolve_dtype(cfg.thinker.dtype, device),
                      attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    system = load_prompt(args.system_prompt_file)
    tools = select_tools(args.tools, args.tools_file)
    print(f"tools: {[t['function']['name'] for t in tools or []]}\n")
    canned = {}
    if args.mock_results:
        with open(args.mock_results, encoding="utf-8") as f:
            canned = json.load(f)

    def new_backend() -> GenericBackend:
        return MockBackend(tools, canned)

    def fresh() -> list[dict]:
        return [{"role": "system", "content": system}]

    messages = fresh()
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [q.strip() for q in f if q.strip() and not q.startswith("#")]
        for q in questions:
            if not args.multi_turn:
                messages = fresh()
            messages.append({"role": "user", "content": q})
            print(f"GUEST: {q}")
            print("\n".join(respond(thinker, messages, tools, new_backend())) + "\n")
        return
    print("Type as the guest. Empty line = new conversation, Ctrl-D = quit.")
    backend = new_backend()
    while True:
        try:
            q = input("GUEST: ").strip()
        except EOFError:
            break
        if not q:
            messages, backend = fresh(), new_backend()
            print("  (new conversation)")
            continue
        messages.append({"role": "user", "content": q})
        print("\n".join(respond(thinker, messages, tools, backend)))


if __name__ == "__main__":
    main()
