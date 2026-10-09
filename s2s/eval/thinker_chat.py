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
import traceback

import torch

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel_v3 import GenericBackend, select_tools
from s2s.eval.text_tools import extract_calls
from s2s.models.thinker import Thinker
from s2s.runtime.guards import VOICE_RULES, claims_action, retry_note, speakable, unsupported
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


MAX_NEW = 150


@torch.no_grad()
def generate(thinker: Thinker, messages: list[dict], tools: list | None, max_new: int | None = None) -> str:
    text = thinker.prompts._render(messages, tools, add_generation_prompt=True)
    ids = torch.tensor([thinker.prompts.ids(text)], device=thinker.device)
    try:
        out = thinker.model.generate(ids, max_new_tokens=max_new or MAX_NEW, do_sample=False, use_cache=True,
                                     eos_token_id=sorted(thinker.eos_ids), pad_token_id=thinker.pad_id)
        return thinker.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _succeeded(result) -> bool:
    return isinstance(result, dict) and result.get("success", True) is not False and "error" not in result


FALLBACK_UNKNOWN = ("I apologize, but I don't have that information at the moment. "
                    "Is there anything else I can assist you with?")
FALLBACK_CLAIM = "I can take care of that for you. Shall I go ahead?"


def respond(thinker: Thinker, messages: list[dict], tools: list | None, backend: GenericBackend,
            max_rounds: int = 3, guard: bool = False) -> list[str]:
    """One guest turn: appends the assistant / tool messages to `messages`, returns printable lines.

    guard: check the final reply with s2s.runtime.guards (an action claimed without a successful tool call,
    details found in none of system prompt / guest words / tool results). Each check gets one retry with a
    note that is shown to the model for that generation only; if it fails again a safe line is spoken.
    The reply is always cleaned for speech (no markdown / emojis)."""
    lines, succeeded, note, retried = [], [], None, set()
    for _ in range(max_rounds + 2):
        try:
            out = generate(thinker, messages + ([{"role": "user", "content": note}] if note else []), tools)
        except Exception as e:  # noqa: BLE001 - keep the run going, show what broke
            lines.append(f"  ERROR (generate): {type(e).__name__}: {e}")
            return lines
        calls = [c for c in extract_calls(out) if c.get("name")]
        if not calls:
            spoken = speakable(out)
            if guard:
                problem = None
                if claims_action(spoken, succeeded):
                    problem = ("claim", None)
                else:
                    sources = [m["content"] for m in messages if m["role"] in ("system", "user", "tool")]
                    sources += [json.dumps(tc["function"]["arguments"]) for m in messages
                                for tc in m.get("tool_calls") or []]
                    bad = unsupported(spoken, sources)
                    if bad:
                        problem = ("unsupported", bad)
                if problem:
                    kind, detail = problem
                    lines.append(f"  GUARD ({kind}{': ' + ', '.join(detail) if detail else ''}): {spoken}")
                    if kind not in retried:
                        retried.add(kind)
                        note = retry_note(kind, detail)
                        continue
                    spoken = FALLBACK_CLAIM if kind == "claim" else FALLBACK_UNKNOWN
                    lines.append("  GUARD: still failing after the retry, speaking the safe line")
            messages.append({"role": "assistant", "content": spoken})
            lines.append(f"  AGENT: {spoken}")
            return lines
        note = None  # the model acted; the note no longer applies
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}} for c in calls]})
        for c in calls:
            try:
                result = backend.execute(c)
            except Exception as e:  # noqa: BLE001
                result = {"success": False, "error": f"{type(e).__name__}: {e}"}
            if _succeeded(result):
                succeeded.append(c["name"])
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
    p.add_argument("--voice-rules", action="store_true", help="append guards.VOICE_RULES to the system prompt")
    p.add_argument("--guards", action="store_true", help="check replies (claimed actions, invented details)")
    p.add_argument("--device-map", default=None, help="auto = split the model over all GPUs (Kaggle 2x T4)")
    p.add_argument("--max-new", type=int, default=150, help="max new tokens per generation")
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    thinker = Thinker(args.model or cfg.thinker.model, device, resolve_dtype(cfg.thinker.dtype, device),
                      attn_implementation=cfg.thinker.attn_implementation, device_map=args.device_map)
    thinker.model.eval()
    system = load_prompt(args.system_prompt_file)
    if args.voice_rules:
        system = f"{system}\n\n{VOICE_RULES}"
    global MAX_NEW
    MAX_NEW = args.max_new
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
            try:
                print("\n".join(respond(thinker, messages, tools, new_backend(), guard=args.guards)) + "\n", flush=True)
            except Exception:  # noqa: BLE001 - one broken turn must not end the run
                traceback.print_exc()
                print(flush=True)
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
        print("\n".join(respond(thinker, messages, tools, backend, guard=args.guards)))


if __name__ == "__main__":
    main()
