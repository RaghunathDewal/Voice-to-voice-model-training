"""Hotel conversations, version 3: the property and the tool set come from the prompt.

v2 trained one fixed setup: four hotel tools, and hotel facts fetched with
get_property_information. A deployment, however, puts its own property details
(unit names, opening hours, Wi-Fi, policies) in the system prompt and exposes only
the tools it supports (e.g. just order_product and create_issue). v3 teaches that:

  * every conversation gets a random property profile (name, unit naming, a random
    subset of facts with random values) written into the system prompt in one of
    several formats (JSON, key: value lines, bullets, prose), plus a persona and the
    team the assistant hands over to ("the front desk", "reception", "the host", ...)
  * a random subset of a larger tool pool, with varied function names and
    descriptions, so the model reads the schema instead of memorising one tool list
  * property questions are answered from the prompt (no tool); a fact that is not
    in the prompt gets "I don't have that information" plus the hand-over
  * an action request is a tool call only when a matching tool is listed; otherwise
    the assistant says it can't do that and names the hand-over team

Rows: {id, system, tools (schema list), text, history?, tool_calls, reply_after_tool | reply + answer_contains}.
`system` is the complete system message, so callers use it instead of the default prompt.
"""

from __future__ import annotations

import json
import random

from s2s.data.hotel_v2 import _issue as _v2_issue
from s2s.data.hotel_v2 import _order as _v2_order
from s2s.data.hotel_v2 import _wakeup as _v2_wakeup
from s2s.data.hotel_v2 import _wrap

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties,
                                                            "required": required}}}


# ------------------------------------------------------------------ tool pool
# kind -> list of (function name, description); the arguments are fixed per kind
_TOOL_NAMES = {
    "order": [("order_product", "Order an item or amenity delivered to the guest's room."),
              ("order_item", "Send an item to the guest's unit."),
              ("request_amenity", "Request an amenity or supply for the guest.")],
    "issue": [("create_issue", "Report a problem so maintenance or housekeeping can fix it."),
              ("report_issue", "Log a maintenance, housekeeping or noise problem."),
              ("create_ticket", "Create a service ticket for a problem the guest reports.")],
    "wakeup": [("schedule_wakeup_call", "Schedule a wake-up call."),
               ("set_wakeup_call", "Set a wake-up call for the guest.")],
    "cleaning": [("request_housekeeping", "Ask housekeeping to clean or service the guest's unit."),
                 ("schedule_cleaning", "Schedule a cleaning or turndown service.")],
    "late_checkout": [("request_late_checkout", "Request a later checkout time."),
                      ("extend_checkout", "Ask for a late checkout.")],
    "table": [("book_table", "Book a table at the on-site restaurant."),
              ("reserve_restaurant", "Reserve a table at the property's restaurant.")],
    "transport": [("book_transport", "Book a taxi or shuttle for the guest."),
                  ("request_taxi", "Arrange a taxi for the guest.")],
    "handover": [("transfer_to_staff", "Hand the conversation to a human staff member."),
                 ("escalate_to_front_desk", "Pass the request to the front desk team.")],
}
_TOOL_ARGS = {
    "order": ({"product": {"type": "string", "description": "Item, singular noun, e.g. towel, pillow"},
               "quantity": {"type": "integer", "description": "How many, default 1"}}, ["product"]),
    "issue": ({"category": {"type": "string", "enum": ["maintenance", "housekeeping", "noise", "other"]},
               "description": {"type": "string", "description": "Short description of the problem"}},
              ["category", "description"]),
    "wakeup": ({"time": {"type": "string", "description": "24-hour time HH:MM"}}, ["time"]),
    "cleaning": ({"service": {"type": "string", "enum": ["room_cleaning", "turndown", "linen_change"]}}, ["service"]),
    "late_checkout": ({"time": {"type": "string", "description": "Requested checkout time, 24-hour HH:MM"}}, ["time"]),
    "table": ({"time": {"type": "string", "description": "24-hour time HH:MM"},
               "guests": {"type": "integer", "description": "Number of people"}}, ["time", "guests"]),
    "transport": ({"destination": {"type": "string", "description": "Where to go"},
                   "time": {"type": "string", "description": "Pickup time, 24-hour HH:MM"}}, ["destination", "time"]),
    "handover": ({"reason": {"type": "string", "description": "Short summary of what the guest needs"}}, ["reason"]),
}
# free-text arguments: the scorer only checks they are present
FREE_TEXT_ARGS_V3 = {"description", "destination", "reason"}
# how often each kind is offered (order and issue are the most common deployments)
_TOOL_WEIGHT = {"order": 0.8, "issue": 0.8, "wakeup": 0.45, "cleaning": 0.35, "late_checkout": 0.3, "table": 0.3,
                "transport": 0.3, "handover": 0.25}


def tool_schema(kind: str, variant: int = 0) -> dict:
    name, desc = _TOOL_NAMES[kind][variant]
    props, req = _TOOL_ARGS[kind]
    return _fn(name, desc, props, req)


def _pick_tools(rng: random.Random) -> dict[str, dict]:
    """kind -> schema. ~8% of conversations offer no tools at all."""
    if rng.random() < 0.08:
        return {}
    kinds = [k for k, w in _TOOL_WEIGHT.items() if rng.random() < w]
    if not kinds:
        kinds = [rng.choice(["order", "issue"])]
    rng.shuffle(kinds)
    return {k: tool_schema(k, rng.randrange(len(_TOOL_NAMES[k]))) for k in kinds}


def generic_tool_result(kind: str, args: dict, n: int = 1) -> dict:
    """What a mock backend returns for a call of this kind."""
    if kind == "order":
        return {"success": True, "order_id": str(8000 + n), "eta_minutes": 15, **args}
    if kind == "issue":
        return {"success": True, "ticket_id": str(500 + n), **args}
    if kind == "table":
        return {"success": True, "booking_id": str(300 + n), **args}
    if kind == "transport":
        return {"success": True, "booking_id": str(700 + n), **args}
    return {"success": True, **args}


def kind_of(name: str) -> str | None:
    for kind, variants in _TOOL_NAMES.items():
        if any(name == v[0] for v in variants):
            return kind
    return None


class GenericBackend:
    """Mock backend for any tool in the v3 pool (and the v1 tool names)."""

    def __init__(self, tools: list | None):
        self.names = {t["function"]["name"] for t in tools or []}
        self.n = 0

    def execute(self, call: dict) -> dict:
        name, args = call.get("name"), call.get("arguments") or {}
        if name not in self.names:
            return {"error": f"unknown tool {name}"}
        kind = kind_of(name)
        if kind is None:
            return {"success": True, **args}
        self.n += 1
        return generic_tool_result(kind, args, self.n)


# ---------------------------------------------------------- property profile
_PROPERTY_NAMES = ["Aurora Grand", "Seaview Resort", "Maple Lodge", "Palm Villas", "City Suites", "Riverside Inn",
                   "Lakeshore Holiday Park", "The Harbour Hotel", "Pinewood Cabins", "Sunset Bay Resort",
                   "Old Town Apartments", "Coral Reef Retreat", "Highland Park Lodges", "Marina View Hotel"]
# (unit word, how a unit is named)
_UNITS = [("room", "number"), ("room", "number"), ("suite", "number"), ("villa", "number"), ("apartment", "number"),
          ("cabin", "name"), ("lodge", "name"), ("cottage", "name")]
_UNIT_NAMES = ["Birch", "Willow", "Cedar", "Heron", "Kestrel", "Oak", "Juniper", "Linden", "Rowan", "Aspen"]
_PERSONAS = ["ARIA", "Maya", "Leo", "Nova", "Sam", "Ella", "Kai", None, None, None]
_HANDOVER = ["the front desk", "the front desk", "reception", "our guest services team", "the host", "our team"]


def _clock(h: int, m: int = 0) -> str:
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    if h == 12 and m == 0:
        return "noon"
    return f"{h12}:{m:02d} {suffix}" if m else f"{h12} {suffix}"


def _make_profile(rng: random.Random) -> dict:
    """topic -> (fact sentence the assistant says, key words a correct answer contains)."""
    floor = lambda: rng.choice(["ground floor", "first floor", "second floor", "third floor", "rooftop", "lobby level"])  # noqa: E731
    b0, b1 = rng.choice([6, 7, 7]), rng.choice([10, 10, 11])
    restaurant = rng.choice(["the Garden Restaurant", "the Terrace", "Blue Fig", "the Lighthouse Grill", "Saffron"])
    pw = rng.choice(["sunrise", "harbour", "maple", "welcome", "seaside", "stay"]) + str(rng.randint(10, 99))
    net = rng.choice(["Guest", "Guest-WiFi", "Free-WiFi", "Lodge"]) + rng.choice(["", "-5G"])
    co_h = rng.choice([10, 11, 11, 12])
    ci_h = rng.choice([14, 15, 15, 16])
    price = rng.choice([10, 15, 20, 25, 30])
    pool_close = rng.choice([20, 21, 22])
    spa_open, spa_close = rng.choice([9, 10]), rng.choice([19, 20, 21])
    bar_close = rng.choice([23, 0, 1])
    facts = {
        "breakfast": (f"Breakfast is served from {_clock(b0, 30 if rng.random() < .5 else 0)} to {_clock(b1)} "
                      f"at {restaurant}.", [restaurant.split()[-1].lower(), _clock(b1).split()[0]]),
        "wifi": (f"The Wi-Fi network is {net} and the password is {pw}.", [pw]),
        "pool": (f"The pool is on the {floor()} and is open from 7 AM to {_clock(pool_close)}.",
                 [_clock(pool_close).split()[0]]),
        "gym": (rng.choice(["The gym is open 24 hours with your key card.",
                            f"The gym is on the {floor()} and opens at 6 AM."]), ["gym"]),
        "parking": (rng.choice([f"Parking is {price} dollars per night.", "Parking is free for guests.",
                                f"Valet parking is available for {price} dollars a night."]),
                    ["free"] if price == 0 else ["parking"]),
        "checkout": (f"Checkout time is {_clock(co_h)}.", [_clock(co_h).split()[0]]),
        "checkin": (f"Check-in starts at {_clock(ci_h)}.", [_clock(ci_h).split()[0]]),
        "restaurant": (f"{restaurant[0].upper() + restaurant[1:]} serves dinner from 6 to {_clock(rng.choice([21, 22]))}.",
                       [restaurant.split()[-1].lower()]),
        "spa": (f"The spa is open from {_clock(spa_open)} to {_clock(spa_close)}; booking ahead is recommended.",
                [_clock(spa_close).split()[0]]),
        "pets": (rng.choice(["Pets are welcome for a small fee.", "Sorry, pets are not allowed.",
                             "Dogs are welcome in ground floor units."]), ["pet", "dog"]),
        "smoking": (rng.choice(["The property is completely non-smoking.",
                                "Smoking is allowed only in the garden area."]), ["smok"]),
        "laundry": (rng.choice(["Same-day laundry is available if you drop it off before 10 AM.",
                                "There is a self-service laundry room next to reception."]), ["laundry"]),
        "bar": (f"The bar is open until {_clock(bar_close)}.", [_clock(bar_close).split()[0]]),
        "shuttle": (rng.choice(["A free shuttle to the city centre leaves every hour from 8 AM.",
                                "The airport shuttle runs every 30 minutes and costs 12 dollars."]), ["shuttle"]),
        "beach": (rng.choice(["The beach is a five-minute walk through the garden gate.",
                              "Beach towels are available at the pool desk."]), ["beach"]),
        "kids_club": (rng.choice(["The kids club is open from 9 AM to 5 PM for ages 4 to 12.",
                                  "There is a playground next to the pool."]), ["kids", "playground"]),
    }
    k = rng.randint(5, len(facts))
    topics = rng.sample(list(facts), k)
    return {t: facts[t] for t in topics}


_TOPIC_Q = {
    "breakfast": ["What time is breakfast?", "Where is breakfast served?", "Till what time is breakfast available?",
                  "When does breakfast start?", "Is there breakfast here?"],
    "wifi": ["What's the Wi-Fi password?", "How do I connect to the internet?", "What is the wifi name?",
             "Can you tell me the Wi-Fi details?"],
    "pool": ["Is there a pool?", "Where is the swimming pool?", "What are the pool timings?", "When does the pool close?"],
    "gym": ["Do you have a gym?", "When is the fitness center open?", "Where is the gym?"],
    "parking": ["How much is parking?", "Where can I park my car?", "Is parking free?"],
    "checkout": ["What is the checkout time?", "What time do I have to check out by?"],
    "checkin": ["What time is check-in?", "When can I check in?"],
    "restaurant": ["Is there a restaurant here?", "When is dinner served?", "What time does the restaurant close?"],
    "spa": ["Do you have a spa?", "What are the spa hours?", "Can I get a massage here?"],
    "pets": ["Can I bring my dog?", "Are pets allowed?", "Is it pet friendly?"],
    "smoking": ["Can I smoke here?", "Is there a smoking area?", "Is smoking allowed?"],
    "laundry": ["Do you have laundry service?", "Where can I wash my clothes?", "Is there a laundry?"],
    "bar": ["Is there a bar?", "Until when is the bar open?", "What time does the bar close?"],
    "shuttle": ["Is there a shuttle?", "How do I get to the city centre?", "Do you have an airport shuttle?"],
    "beach": ["How far is the beach?", "How do I get to the beach?", "Do you have beach towels?"],
    "kids_club": ["Is there anything for kids?", "Do you have a kids club?", "Is there a play area for children?"],
}
_TOPIC_LABEL = {"wifi": "Wi-Fi", "checkout": "checkout time", "checkin": "check-in time", "kids_club": "kids club"}


def _make_stay(rng: random.Random, unit_word: str, unit_naming: str) -> dict:
    first = rng.choice(["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Chris", "Priya", "Wei", "Maria", "Omar",
                        "Ananya", "Rahul", "Sofia", "Lucas", "Emma", "Noah"])
    last = rng.choice(["Smith", "Garcia", "Chen", "Patel", "Johnson", "Brown", "Kim", "Nguyen", "Lopez", "Khan",
                       "Sharma", "Rossi", "Muller", "Silva"])
    day = rng.randint(1, 20)
    nights = rng.randint(1, 6)
    month = rng.randint(1, 12)
    if unit_naming == "name":
        unit = f"{unit_word.capitalize()} {rng.choice(_UNIT_NAMES)}"
    else:
        unit = f"{unit_word.capitalize()} {rng.choice([1, 2, 3, 4, 5, 6]) * 100 + rng.randint(1, 30)}"
    return {"guest_name": f"{first} {last}", "unit": unit, "check_in": f"2026-{month:02d}-{day:02d}",
            "check_out": f"2026-{month:02d}-{day + nights:02d}", "nights": nights,
            "guests": rng.choice([1, 2, 2, 3, 4]), "balance_due": rng.choice([0, 0, 45, 120, 260, 380])}


def _spoken_date(iso: str) -> str:
    _, m, d = iso.split("-")
    return f"{_MONTHS[int(m) - 1]} {int(d)}"


def _format_block(rng: random.Random, title: str, items: list[tuple[str, str]]) -> str:
    """The same facts in one of four layouts deployments use."""
    style = rng.random()
    if style < 0.3:
        return f"{title}:\n" + json.dumps({k: v for k, v in items}, indent=1)
    if style < 0.55:
        return f"{title}:\n" + "\n".join(f"{k}: {v}" for k, v in items)
    if style < 0.8:
        return f"## {title}\n" + "\n".join(f"- {k}: {v}" for k, v in items)
    return f"{title}: " + " ".join(v if v.endswith(".") else v + "." for _, v in items)


def _capabilities(tools: dict, unit_word: str, handover: str) -> str:
    caps = {"order": f"send items to your {unit_word}", "issue": "report problems", "wakeup": "set wake-up calls",
            "cleaning": "arrange housekeeping", "late_checkout": "request a late checkout",
            "table": "book a table at the restaurant", "transport": "arrange a taxi",
            "handover": f"connect you with {handover}"}
    parts = [caps[k] for k in tools] + ["answer questions about the property and your stay"]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _system_prompt(rng: random.Random, prop: str, persona: str | None, handover: str, profile: dict, stay: dict,
                   unit_word: str) -> str:
    who = f"You are {persona}, the voice concierge for {prop}." if persona else \
        f"You are the voice assistant for {prop}, speaking with a guest out loud."
    rules = rng.choice([
        "Answer property questions only from the property information below; if something is not listed, say you "
        f"don't have that information and that {handover} can help.",
        f"Use the property information below for facts. If you don't know something, offer {handover}.",
        f"Only state facts that appear below. For anything else, refer the guest to {handover}.",
    ])
    tools_rule = rng.choice([
        "Use a tool only when the guest asks for something one of your tools can do. "
        f"If no tool fits, say you can't do it yourself and that {handover} can help.",
        f"Call a function only for actions it supports; never invent one. Otherwise hand over to {handover}.",
        "You may only act through the tools you are given.",
    ])
    style = rng.choice([
        "Keep replies to one or two short spoken sentences, with no markdown, lists or emojis.",
        "Speak naturally and briefly. No lists, no markdown.",
        "Be warm and concise; your replies are spoken aloud.",
    ])
    facts = [(t.replace("_", " "), s) for t, (s, _) in profile.items()]
    rng.shuffle(facts)
    guest = [("guest", stay["guest_name"]), (stay["unit"].split()[0].lower(), stay["unit"]),
             ("check-in", stay["check_in"]), ("check-out", stay["check_out"]), ("guests", str(stay["guests"])),
             ("balance due", f"{stay['balance_due']} USD")]
    parts = [who, rules, tools_rule, style, _format_block(rng, f"Property information ({prop})", facts),
             _format_block(rng, "Current guest", guest)]
    if rng.random() < 0.3:  # some prompts put the guest first
        parts[4], parts[5] = parts[5], parts[4]
    return "\n\n".join(parts)


# -------------------------------------------------------------- intents
def _hhmm(h: int, m: int = 0) -> str:
    return f"{h:02d}:{m:02d}"


def _ask_order(rng, wrap=True):
    text, call, reply = _v2_order(rng, wrap)
    return text, call["arguments"], (lambda unit, r=reply: r)


def _ask_issue(rng, wrap=True):
    text, call, reply = _v2_issue(rng, wrap)
    return text, call["arguments"], (lambda unit, r=reply: r.replace("your room", f"your {unit}"))


def _ask_wakeup(rng, wrap=True):
    text, call, reply = _v2_wakeup(rng, wrap)
    return text, call["arguments"], (lambda unit, r=reply: r)


def _ask_cleaning(rng, wrap=True):
    service, phrases, reply = rng.choice([
        ("room_cleaning", ["can you send someone to clean my {u}?", "I'd like my {u} cleaned, please.",
                           "can housekeeping come and clean now?"], "Of course, housekeeping will come to clean shortly."),
        ("turndown", ["can I get turndown service tonight?", "please arrange turndown service."],
         "Done, turndown service is arranged for this evening."),
        ("linen_change", ["can you change the bed sheets?", "I'd like fresh linen on the bed, please."],
         "Sure, housekeeping will change the linen for you."),
    ])
    core = rng.choice(phrases)
    return core, {"service": service}, (lambda unit, r=reply: r)


def _ask_late_checkout(rng, wrap=True):
    h = rng.choice([12, 13, 14])
    spoken = {12: rng.choice(["noon", "12"]), 13: rng.choice(["1 PM", "one"]), 14: rng.choice(["2 PM", "two"])}[h]
    core = rng.choice(["can I check out at {t} instead?", "can I get a late checkout until {t}?",
                       "is it possible to stay until {t} on my last day?", "I'd like a late checkout at {t}."]
                      ).format(t=spoken)
    return core, {"time": _hhmm(h)}, (lambda unit, h=h: f"I've requested a late checkout at {_clock(h)} for you; "
                                                         "you'll get a confirmation shortly.")


def _ask_table(rng, wrap=True):
    h, m = rng.choice([(18, 0), (19, 0), (19, 30), (20, 0), (20, 30), (21, 0)])
    n = rng.choice([2, 2, 3, 4, 5, 6])
    t = _clock(h, m)
    core = rng.choice(["can you book a table for {n} at {t}?", "I'd like a dinner table for {n} people at {t}.",
                       "reserve a table for {n} tonight at {t}, please.", "table for {n} at {t}, please."]
                      ).format(n=rng.choice([_NUM[n], str(n)]), t=t)
    return core, {"time": _hhmm(h, m), "guests": n}, (lambda unit, n=n, t=t: f"Your table for {_NUM[n]} at {t} is booked.")


def _ask_transport(rng, wrap=True):
    dest = rng.choice(["the airport", "the train station", "the city centre", "the old town", "the convention centre"])
    h, m = rng.choice([(5, 30), (6, 0), (7, 0), (8, 30), (10, 0), (14, 0), (17, 30)])
    core = rng.choice(["can you book me a taxi to {d} at {t}?", "I need a cab to {d} at {t}.",
                       "please arrange a ride to {d} for {t}.", "can I get a taxi at {t} to go to {d}?"]
                      ).format(d=dest, t=_clock(h, m))
    return core, {"destination": dest.replace("the ", ""), "time": _hhmm(h, m)}, \
        (lambda unit, d=dest, t=_clock(h, m): f"Your taxi to {d} is booked for {t}.")


# kind -> request generator; a request is a tool call only when its kind is offered
_ACTIONS = {"order": _ask_order, "issue": _ask_issue, "wakeup": _ask_wakeup, "cleaning": _ask_cleaning,
            "late_checkout": _ask_late_checkout, "table": _ask_table, "transport": _ask_transport}
# staff-only requests: never a tool unless a hand-over tool exists
_STAFF_ONLY = ["I'd like to extend my stay by another night.", "can I change to a bigger {u}?",
               "can you cancel my booking?", "can I pay my bill now?", "can you email me the invoice?",
               "I lost my key card.", "can I leave my luggage after checkout?", "can I get a refund for last night?",
               "can you upgrade me?", "can I add one more guest to my booking?"]
_URGENT = ["I need a doctor, I'm not feeling well.", "my kid has a high fever.", "someone fainted in my {u}.",
           "I think I need an ambulance.", "I cut my hand badly."]
_ELSEWHERE = ["What's the weather like tomorrow?", "Can you play some music?", "Can you book a flight for me?",
              "What's the score of the cricket match?", "Set a reminder for my meeting.",
              "Can you order me a pizza from outside?", "Tell me a joke."]


class _Ctx:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.prop = rng.choice(_PROPERTY_NAMES)
        self.unit_word, naming = rng.choice(_UNITS)
        self.persona = rng.choice(_PERSONAS)
        self.handover = rng.choice(_HANDOVER)
        self.profile = _make_profile(rng)
        self.stay = _make_stay(rng, self.unit_word, naming)
        self.tools = _pick_tools(rng)
        self.system = _system_prompt(rng, self.prop, self.persona, self.handover, self.profile, self.stay,
                                     self.unit_word)

    def can_not(self) -> str:
        h = self.handover
        return self.rng.choice([f"I can't do that myself, but {h} will be happy to help you with that.",
                                f"I'm not able to arrange that, but {h} can help you right away.",
                                f"That's something {h} can help you with."])


def _u(ctx: _Ctx, s: str) -> str:
    return s.replace("{u}", ctx.unit_word)


def _turn(ctx: _Ctx, wrap: bool = True) -> dict:
    """One guest turn -> {text, call(kind, args) | None, reply, facts}."""
    rng = ctx.rng
    r = rng.random()
    if r < 0.5:  # an action request: offered tools are asked for more often than missing ones
        offered = [k for k in _ACTIONS if k in ctx.tools]
        missing = [k for k in _ACTIONS if k not in ctx.tools]
        kind = rng.choice(offered) if offered and (not missing or rng.random() < 0.7) else rng.choice(missing)
        text, args, reply_fn = _ACTIONS[kind](rng, wrap)
        text = _u(ctx, text)
        text = _wrap(rng, text) if wrap and kind not in ("order", "issue", "wakeup") else text
        if kind in ctx.tools:
            return {"text": text, "call": (kind, args), "reply": reply_fn(ctx.unit_word)}
        return {"text": text, "call": None, "reply": ctx.can_not(), "facts": [ctx.handover.split()[-1]]}
    if r < 0.7:  # property question
        if rng.random() < 0.8:
            topic = rng.choice(list(ctx.profile))
            sentence, facts = ctx.profile[topic]
            return {"text": _wrap(rng, rng.choice(_TOPIC_Q[topic])) if wrap else rng.choice(_TOPIC_Q[topic]),
                    "call": None, "reply": sentence, "facts": facts, "topic": topic}
        topic = rng.choice([t for t in _TOPIC_Q if t not in ctx.profile] or list(_TOPIC_Q))
        if topic in ctx.profile:
            sentence, facts = ctx.profile[topic]
            return {"text": rng.choice(_TOPIC_Q[topic]), "call": None, "reply": sentence, "facts": facts}
        reply = rng.choice([f"I'm sorry, I don't have that information, but {ctx.handover} can help.",
                            f"I don't have details on that. {ctx.handover[0].upper() + ctx.handover[1:]} will know."])
        return {"text": rng.choice(_TOPIC_Q[topic]), "call": None, "reply": reply,
                "facts": [ctx.handover.split()[-1]], "topic": topic}
    if r < 0.82:  # the guest's own stay
        return _stay_turn(ctx, wrap)
    if r < 0.9:
        return _small_talk(ctx)
    return _other(ctx, wrap)


def _stay_turn(ctx: _Ctx, wrap: bool) -> dict:
    rng, st = ctx.rng, ctx.stay
    first = st["guest_name"].split()[0]
    unit_word = st["unit"].split()[0].lower()
    options = [
        ([f"Which {unit_word} am I in?", f"What's my {unit_word}?", f"Can you remind me of my {unit_word} number?"
          if st["unit"].split()[1].isdigit() else f"What's the name of my {unit_word}?"],
         f"You're in {st['unit']}.", [st["unit"].split()[1].lower()]),
        (["When am I checking out?", "What's my checkout date?", "When do I leave?"],
         f"You check out on {_spoken_date(st['check_out'])}.", [_spoken_date(st["check_out"]).split()[1]]),
        (["How many nights am I staying?", "How long is my stay?"],
         f"You're staying {st['nights']} night{'s' if st['nights'] != 1 else ''}.",
         [str(st["nights"]), _NUM.get(st["nights"], "#")]),
        (["How much do I owe?", "What's my balance?", "Is anything pending on my bill?"],
         "You don't have anything to pay." if st["balance_due"] == 0 else
         f"Your remaining balance is {st['balance_due']} dollars.",
         ["nothing", "don't", "no "] if st["balance_due"] == 0 else [str(st["balance_due"])]),
        (["Whose name is the booking under?", "What name is the reservation in?"],
         f"The booking is under {st['guest_name']}.", [first.lower()]),
        (["Tell me about my booking.", "What are my reservation details?", "Can you read out my booking?"],
         f"Your booking is under {st['guest_name']}: {st['unit']}, from {_spoken_date(st['check_in'])} to "
         f"{_spoken_date(st['check_out'])}, for {st['guests']} guest{'s' if st['guests'] != 1 else ''}.",
         [st["unit"].split()[1].lower()]),
    ]
    qs, reply, facts = rng.choice(options)
    q = rng.choice(qs)
    return {"text": _wrap(rng, q) if wrap else q, "call": None, "reply": reply, "facts": facts}


def _small_talk(ctx: _Ctx) -> dict:
    rng = ctx.rng
    first = ctx.stay["guest_name"].split()[0]
    prop = ctx.prop[4:] if ctx.prop.startswith("The ") else ctx.prop  # "the Harbour Hotel", not "the The ..."
    me = f"I'm {ctx.persona}, the voice concierge for {ctx.prop}" if ctx.persona else f"I'm the {prop} voice assistant"
    options = [
        (["Hi.", "Hello.", "Good morning.", "Hey there.", "Good evening."],
         f"Hello {first}! How can I help you today?", ["help"]),
        (["Thank you.", "Thanks a lot.", "That's all, thanks.", "Great, thanks."],
         "You're welcome! Enjoy your stay.", ["welcome"]),
        (["Who are you?", "What can you do?", "What can you help me with?"],
         f"{me}. I can {_capabilities(ctx.tools, ctx.unit_word, ctx.handover)}.", ["property", "stay"]),
        (["Goodbye.", "Bye.", "Okay bye."], "Goodbye, have a great day!", ["goodbye", "great day"]),
    ]
    qs, reply, facts = rng.choice(options)
    return {"text": rng.choice(qs), "call": None, "reply": reply, "facts": facts}


def _other(ctx: _Ctx, wrap: bool) -> dict:
    rng = ctx.rng
    r = rng.random()
    if r < 0.5:
        text = _u(ctx, rng.choice(_STAFF_ONLY))
        text = _wrap(rng, text) if wrap else text
        if "handover" in ctx.tools:
            return {"text": text, "call": ("handover", {"reason": text.rstrip("?.!")}),
                    "reply": f"I've passed this to {ctx.handover}; someone will get back to you shortly."}
        return {"text": text, "call": None, "reply": ctx.can_not(), "facts": [ctx.handover.split()[-1]]}
    if r < 0.7:
        text = _u(ctx, rng.choice(_URGENT))
        return {"text": _wrap(rng, text) if wrap else text, "call": None,
                "reply": f"I'm sorry to hear that. Please contact {ctx.handover} right away, and in an emergency "
                         "call your local emergency number.", "facts": ["emergency"]}
    return {"text": rng.choice(_ELSEWHERE), "call": None,
            "reply": f"Sorry, I can't help with that, but {ctx.handover} will be happy to assist you.",
            "facts": [ctx.handover.split()[-1]]}


def _lower_first(text: str) -> str:
    return text if text.startswith("I ") or text.startswith("I'") else text[0].lower() + text[1:]


def generate_examples_v3(n: int, seed: int = 0, follow_up_prob: float = 0.3) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        ctx = _Ctx(rng)
        tools = list(ctx.tools.values()) or None
        row = {"id": f"hotel3_{seed}_{i:06d}", "system": ctx.system, "tools": tools}
        if rng.random() < follow_up_prob:
            prev = _turn(ctx)
            history = [{"role": "user", "content": prev["text"]}, {"role": "assistant", "content": prev["reply"]}]
            row["history"] = history
            if rng.random() < 0.3 and ctx.profile:  # "what about X?"
                topic = rng.choice(list(ctx.profile))
                sentence, facts = ctx.profile[topic]
                cur = {"text": rng.choice(["What about the {t}?", "And the {t}?", "How about the {t}?"]).format(
                    t=_TOPIC_LABEL.get(topic, topic.replace("_", " "))), "call": None, "reply": sentence,
                    "facts": facts}
            else:
                cur = _turn(ctx, wrap=False)
                cur["text"] = rng.choice(["Also, ", "And ", "One more thing, ", "Oh and ", ""]) + _lower_first(cur["text"]) \
                    if rng.random() < 0.5 else cur["text"]
                cur["text"] = cur["text"][0].upper() + cur["text"][1:]
        else:
            cur = _turn(ctx)
        row["text"] = cur["text"]
        if cur["call"] is not None:
            kind, args = cur["call"]
            name = ctx.tools[kind]["function"]["name"]
            row["tool_calls"] = [{"name": name, "arguments": args}]
            row["reply_after_tool"] = cur["reply"]
        else:
            row.update(tool_calls=[], reply=cur["reply"], answer_contains=cur["facts"])
        rows.append(row)
    return rows
