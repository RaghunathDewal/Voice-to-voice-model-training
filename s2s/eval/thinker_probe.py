"""Out-of-template check for the thinker (text): requests phrased unlike any training template.

    python -m s2s.eval.thinker_probe --config configs/small.yaml --model checkpoints/thinker_merged_v2

Each probe states what a good answer is: a tool name (with optional argument check), "front desk"
(hand over, no tool call) or "answer" (any reply without a tool call). None of these sentences
appear in hotel_v2, so a pass means the behaviour generalises beyond the templates.
"""

from __future__ import annotations

import random

import torch

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, make_reservation, reservation_context, validate_call
from s2s.eval.text_tools import extract_calls
from s2s.models.speech_llm import greedy_generate
from s2s.models.thinker import Thinker
from s2s.utils import resolve_device, resolve_dtype

# (guest says, expected): expected = tool name, "tool:arg=value", "front desk" or "answer"
PROBES = [
    ("Can you send two towels and also the TV isn't working?", "order_product+create_issue"),
    ("Bro can I get like three more pillows real quick", "order_product:quantity=3"),
    ("Ugh, the people next door are so loud I can't sleep.", "create_issue:category=noise"),
    ("The sink is leaking all over the bathroom floor.", "create_issue:category=maintenance"),
    ("Nobody has made up my room today.", "create_issue:category=housekeeping"),
    ("Mind if I ask when breakfast starts?", "get_property_information:topic=breakfast"),
    ("Is there somewhere I can go for a swim?", "get_property_information:topic=pool"),
    ("I gotta be up at six thirty tomorrow, can you ring me?", "schedule_wakeup_call:time=06:30"),
    ("Could someone buzz my room at seven fifteen?", "schedule_wakeup_call:time=07:15"),
    ("Please don't let me oversleep, I need a call at eight.", "schedule_wakeup_call:time=08:00"),
    ("Who are you?", "answer"),
    ("What's the capital of France?", "answer"),
    ("Where can I buy some medicine around here?", "front desk"),
    ("Could I push my checkout to Sunday instead?", "front desk"),
    ("My room faces the road, can I switch to a quieter one?", "front desk"),
    ("I need a receipt for my company, can you do that?", "front desk"),
    ("Can you get me a cab to the convention centre?", "front desk"),
    ("My wife has chest pain, please help.", "front desk"),
    ("I'd like to stay two more nights.", "front desk"),
    ("Can I pay with my card now over the phone?", "front desk"),
]


def judge(expected: str, calls: list[dict], text: str) -> bool:
    if expected == "answer":
        return not calls and len(text.strip()) > 0
    if expected == "front desk":
        return not calls and "front desk" in text.lower()
    wanted = expected.split("+")
    if sorted(c.get("name") for c in calls) != sorted(w.split(":")[0] for w in wanted):
        return False
    if any(validate_call(c) for c in calls):
        return False
    for w in wanted:
        name, _, check = w.partition(":")
        if check:
            key, _, value = check.partition("=")
            call = next(c for c in calls if c["name"] == name)
            if str(call["arguments"].get(key)).lower() != value.lower():
                return False
    return True


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    thinker = Thinker(args.model, device, resolve_dtype(cfg.thinker.dtype, device),
                      attn_implementation=cfg.thinker.attn_implementation)
    thinker.model.eval()
    system = Thinker.system_content(cfg.thinker.system_prompt, reservation_context(make_reservation(random.Random(5))))
    passed = 0
    for text, expected in PROBES:
        ids = thinker.prompts.text_prompt_ids(system, text, HOTEL_TOOLS)
        with torch.no_grad():
            out = thinker.tokenizer.decode(greedy_generate(thinker, thinker.embed(torch.tensor([ids], device=device)), 100),
                                           skip_special_tokens=False)
        calls = extract_calls(out)
        ok = judge(expected, calls, out)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  [{expected}]  {text}\n      -> {out.strip()!r}")
    print(f"\nout-of-template probes: {passed}/{len(PROBES)} passed")


if __name__ == "__main__":
    main()
