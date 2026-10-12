"""Two-speaker hotel phone calls for the full-duplex experiment (scripts only, no audio).

A call is an ordered list of events between a guest and the agent, built from the hotel_v3
generator (random property, prompt facts, tool subset, guest requests and agent replies):

    {"who": "agent", "kind": "say", "text": ...}                        agent speech
    {"who": "agent", "kind": "tool", "call": {...}, "result": {...}}      tool call (text only, no audio)
    {"who": "guest", "kind": "turn", "text": ..., "preface": ...?}       a complete guest turn; with a
                                                                          preface the guest pauses after it
                                                                          and may get an agent backchannel
    {"who": "guest", "kind": "backchannel", "text": "Mhm."}             during the previous agent speech;
                                                                          the agent keeps talking
    agent "say" events may carry "interrupted": True: the next guest turn starts while the agent is
    still speaking, and the agent stops shortly after.

Timing is decided later, from the synthesised durations (s2s.prep.duplex_data).
"""

from __future__ import annotations

import random

from s2s.data.hotel_v3 import _Ctx, _turn, generic_tool_result, kind_of

AGENT_BACKCHANNELS = ["Mhm.", "Okay.", "Right.", "Sure.", "I see."]
GUEST_BACKCHANNELS = ["Mhm.", "Okay.", "Uh-huh.", "Right.", "Yeah."]
PREFACES = ["Hi, yes, this is {unit}.", "Hello, I'm calling from {unit}.", "So, um, I just got back to my room.",
            "Hi there, quick question.", "Yes, hello.", "Okay, so", "Hi, sorry to bother you.",
            "Hello, it's {name} here."]
INTERRUPT_PREFIXES = ["Sorry, ", "Actually, ", "Wait, ", "Oh, sorry, ", "Sorry to cut in, "]
CLOSINGS = [("Thank you, that's all.", "You're welcome! Enjoy your stay."),
            ("Great, thanks. Bye.", "Goodbye, have a great day!"),
            ("Okay, thanks a lot.", "You're welcome. Have a lovely evening!"),
            ("Perfect, thank you.", "My pleasure. Enjoy your stay!")]


def _greeting(ctx: _Ctx, rng: random.Random) -> str:
    prop = ctx.prop
    me = f"this is {ctx.persona}" if ctx.persona else "this is the guest assistant"
    return rng.choice([f"Hello, {me} at {prop}. How can I help you?", f"Good evening, {prop}, {me}. What can I do for you?",
                       f"Hi, you've reached {prop}. How can I help?"])


def _lower_first(text: str) -> str:
    return text if text.startswith("I ") or text.startswith("I'") else text[0].lower() + text[1:]


def generate_call(rng: random.Random, idx: int, seed: int, max_turns: int = 3, p_preface: float = 0.35,
                  p_agent_backchannel: float = 0.6, p_guest_backchannel: float = 0.25,
                  p_interrupt: float = 0.2) -> dict:
    ctx = _Ctx(rng)
    tools = list(ctx.tools.values()) or None
    events: list[dict] = [{"who": "agent", "kind": "say", "text": _greeting(ctx, rng)}]
    n_calls = 0
    n_turns = rng.randint(1, max_turns)
    pending = _turn(ctx)
    turn_i = 0
    while pending is not None:
        guest = {"who": "guest", "kind": "turn", "text": pending["text"]}
        if rng.random() < p_preface and not guest.get("interrupt"):
            guest["preface"] = rng.choice(PREFACES).format(unit=ctx.stay["unit"], name=ctx.stay["guest_name"].split()[0])
            if rng.random() < p_agent_backchannel:
                guest["agent_backchannel"] = rng.choice(AGENT_BACKCHANNELS)
        if pending.get("interrupt"):
            guest["interrupt"] = True
        events.append(guest)
        if pending["call"] is not None:
            kind, args = pending["call"]
            name = ctx.tools[kind]["function"]["name"]
            n_calls += 1
            events.append({"who": "agent", "kind": "tool", "call": {"name": name, "arguments": args},
                           "result": generic_tool_result(kind_of(name) or kind, args, n_calls)})
        say = {"who": "agent", "kind": "say", "text": pending["reply"]}
        events.append(say)
        turn_i += 1
        if turn_i >= n_turns:
            pending = None
            continue
        nxt = _turn(ctx, wrap=rng.random() < 0.5)
        long_reply = len(say["text"].split()) >= 7
        if long_reply and rng.random() < p_interrupt:  # the guest cuts in with the next request
            say["interrupted"] = True
            nxt["text"] = rng.choice(INTERRUPT_PREFIXES) + _lower_first(nxt["text"])
            nxt["interrupt"] = True
        elif long_reply and rng.random() < p_guest_backchannel:
            events.append({"who": "guest", "kind": "backchannel", "text": rng.choice(GUEST_BACKCHANNELS)})
        pending = nxt
    bye, reply = rng.choice(CLOSINGS)
    events += [{"who": "guest", "kind": "turn", "text": bye}, {"who": "agent", "kind": "say", "text": reply}]
    return {"id": f"call_{seed}_{idx:06d}", "system": ctx.system, "tools": tools, "events": events}


def generate_calls(n: int, seed: int = 0, **kw) -> list[dict]:
    rng = random.Random(seed)
    return [generate_call(rng, i, seed, **kw) for i in range(n)]
