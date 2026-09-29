"""Hotel domain: tool schemas, a mock backend, reservation context and a
template generator for (spoken request -> tool call) training/eval examples.

Replace `HotelBackend` with calls to your real APIs; keep the tool schemas in
sync with what the model was trained on.
"""

from __future__ import annotations

import json
import random
from typing import Any

HOTEL_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "order_product",
        "description": "Order an item or amenity delivered to the guest's room.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string", "description": "Item to deliver, singular noun, e.g. towel, pillow, toothbrush"},
            "quantity": {"type": "integer", "description": "How many, default 1"}},
            "required": ["product"]}}},
    {"type": "function", "function": {
        "name": "create_issue",
        "description": "Report a problem in the room so maintenance or housekeeping can fix it.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["maintenance", "housekeeping", "noise", "other"]},
            "description": {"type": "string", "description": "Short description of the problem"}},
            "required": ["category", "description"]}}},
    {"type": "function", "function": {
        "name": "get_property_information",
        "description": "Answer questions about hotel facilities and policies.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "enum": ["breakfast", "wifi", "pool", "gym", "parking", "checkout", "restaurant", "spa"]}},
            "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "schedule_wakeup_call",
        "description": "Schedule a wake-up call to the room.",
        "parameters": {"type": "object", "properties": {
            "time": {"type": "string", "description": "24-hour time HH:MM"}},
            "required": ["time"]}}},
]

TOOL_NAMES = {t["function"]["name"] for t in HOTEL_TOOLS}

PROPERTY_INFO = {
    "breakfast": "Breakfast is served from 6:30 to 10:30 AM in the Garden Restaurant on the ground floor.",
    "wifi": "The Wi-Fi network is HotelGuest and the password is sunrise2024.",
    "pool": "The pool is on the fifth floor and is open from 7 AM to 10 PM.",
    "gym": "The gym is on the second floor and is open 24 hours with your room key.",
    "parking": "Valet parking is available at the main entrance for 25 dollars per night.",
    "checkout": "Standard checkout time is 11 AM. Late checkout until 1 PM can be requested.",
    "restaurant": "The Garden Restaurant serves dinner from 6 to 10 PM.",
    "spa": "The spa is on the sixth floor and is open from 9 AM to 8 PM; bookings are recommended.",
}


def make_reservation(rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    first = rng.choice(["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Chris", "Priya", "Wei", "Maria", "Omar"])
    last = rng.choice(["Smith", "Garcia", "Chen", "Patel", "Johnson", "Brown", "Kim", "Nguyen", "Lopez", "Khan"])
    day = rng.randint(1, 25)
    nights = rng.randint(1, 5)
    return {
        "guest_name": f"{first} {last}",
        "room": str(rng.choice([2, 3, 4, 5, 6, 7]) * 100 + rng.randint(1, 30)),
        "check_in": f"2026-10-{day:02d}",
        "check_out": f"2026-10-{day + nights:02d}",
        "checkout_time": "11:00",
        "room_type": rng.choice(["King Deluxe", "Double Queen", "Junior Suite"]),
        "balance_due": f"{rng.randint(0, 400)}.00 USD",
    }


def reservation_context(res: dict) -> str:
    return "Current guest reservation:\n" + json.dumps(res, indent=1)


class HotelBackend:
    """In-memory mock of the hotel APIs. Returns JSON-serialisable dicts."""

    def __init__(self, reservation: dict):
        self.reservation = reservation
        self.orders: list[dict] = []
        self.issues: list[dict] = []
        self.wakeups: list[str] = []

    def execute(self, call: dict) -> dict:
        name, args = call.get("name"), call.get("arguments") or {}
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def get_reservation(self) -> dict:
        return dict(self.reservation)

    def order_product(self, product: str, quantity: int = 1) -> dict:
        order = {"order_id": str(8000 + len(self.orders) + 1), "product": product, "quantity": int(quantity),
                 "eta_minutes": 15}
        self.orders.append(order)
        return {"success": True, **order}

    def create_issue(self, category: str, description: str) -> dict:
        issue = {"ticket_id": str(500 + len(self.issues) + 1), "category": category, "description": description}
        self.issues.append(issue)
        return {"success": True, **issue}

    def get_property_information(self, topic: str) -> dict:
        return {"topic": topic, "info": PROPERTY_INFO.get(topic, "I don't have information about that.")}

    def schedule_wakeup_call(self, time: str) -> dict:
        self.wakeups.append(time)
        return {"success": True, "time": time}


def validate_call(call: dict) -> str | None:
    """Minimal JSON-schema check. Returns an error string or None."""
    spec = next((t["function"] for t in HOTEL_TOOLS if t["function"]["name"] == call.get("name")), None)
    if spec is None:
        return f"unknown tool {call.get('name')}"
    params = spec["parameters"]
    args = call.get("arguments") or {}
    for req in params.get("required", []):
        if req not in args:
            return f"missing argument {req}"
    for k, v in args.items():
        prop = params["properties"].get(k)
        if prop is None:
            return f"unexpected argument {k}"
        if prop["type"] == "integer" and not isinstance(v, int):
            return f"{k} must be an integer"
        if prop["type"] == "string" and not isinstance(v, str):
            return f"{k} must be a string"
        if "enum" in prop and v not in prop["enum"]:
            return f"{k} must be one of {prop['enum']}"
    return None


# ------------------------------------------------------------ data generator
_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
_PRODUCTS = {"towel": "towels", "pillow": "pillows", "blanket": "blankets", "toothbrush": "toothbrushes",
             "bottle of water": "bottles of water", "bathrobe": "bathrobes", "hair dryer": "hair dryers",
             "phone charger": "phone chargers", "coffee pod": "coffee pods", "roll of toilet paper": "rolls of toilet paper"}
_ISSUES = [
    ("maintenance", "the air conditioning is not working", "air conditioning not working"),
    ("maintenance", "the shower has no hot water", "no hot water in shower"),
    ("maintenance", "the TV remote is broken", "TV remote broken"),
    ("maintenance", "the toilet keeps running", "toilet keeps running"),
    ("maintenance", "the light in the bathroom is out", "bathroom light out"),
    ("housekeeping", "the room hasn't been cleaned today", "room not cleaned today"),
    ("housekeeping", "there are no clean sheets on the bed", "no clean sheets on bed"),
    ("noise", "the room next door is really loud", "loud neighbours next door"),
    ("noise", "there's construction noise outside my window", "construction noise outside window"),
]
_TOPIC_Q = {
    "breakfast": ["What time is breakfast?", "Is breakfast included, and where is it served?", "When does breakfast end?"],
    "wifi": ["What's the Wi-Fi password?", "How do I connect to the internet?", "What is the wifi network called?"],
    "pool": ["Is the pool open right now?", "Where is the swimming pool?", "What are the pool hours?"],
    "gym": ["Do you have a gym?", "When is the fitness center open?"],
    "parking": ["How much is parking?", "Where can I park my car?"],
    "checkout": ["What time is checkout at this hotel?", "Can I get a late checkout?"],
    "restaurant": ["When is the restaurant open for dinner?", "Is there a restaurant in the hotel?"],
    "spa": ["Do you have a spa?", "What are the spa hours?"],
}


def _order_example(rng: random.Random) -> tuple[str, dict]:
    product = rng.choice(list(_PRODUCTS))
    qty = rng.choice([1, 1, 2, 2, 3, 4])
    noun = product if qty == 1 else _PRODUCTS[product]
    if qty == 1:
        amount = rng.choice(["an" if product[0] in "aeiou" else "a", "one"])
    else:
        amount = _NUM_WORDS[qty]
    templates = [
        "Can you send {a} {n} to my room?", "Could I get {a} {n}, please?", "I need {a} {n}.",
        "Please bring {a} {n} up to the room.", "Hi, can someone drop off {a} {n}?",
        "We're out of {n2}, can you send {a}?", "I'd like {a} extra {n}.",
    ]
    text = rng.choice(templates).format(a=amount, n=noun, n2=_PRODUCTS[product])
    return text, {"name": "order_product", "arguments": {"product": product, "quantity": qty}}


def _issue_example(rng: random.Random) -> tuple[str, dict]:
    cat, spoken, desc = rng.choice(_ISSUES)
    templates = ["Hi, {s}.", "I want to report that {s}.", "Can you send someone? {S}.", "Excuse me, {s}, can you help?"]
    t = rng.choice(templates)
    text = t.format(s=spoken, S=spoken[:1].upper() + spoken[1:])
    return text, {"name": "create_issue", "arguments": {"category": cat, "description": desc}}


def _info_example(rng: random.Random) -> tuple[str, dict]:
    topic = rng.choice(list(_TOPIC_Q))
    return rng.choice(_TOPIC_Q[topic]), {"name": "get_property_information", "arguments": {"topic": topic}}


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _spoken_date(iso: str) -> str:
    _, m, d = iso.split("-")
    return f"{_MONTHS[int(m) - 1]} {int(d)}"


def _reservation_example(rng: random.Random, res: dict) -> tuple[str, dict]:
    """Answered from the reservation in the prompt context: no tool call, a reply with the right fact."""
    nights = int(res["check_out"][-2:]) - int(res["check_in"][-2:])
    balance = res["balance_due"].split(".")[0]
    options = [
        (["What time is my checkout?", "When am I checking out?"],
         f"You check out on {_spoken_date(res['check_out'])} at 11 AM.", ["11"]),
        (["Which room am I in again?", "What's my room number?"], f"You're in room {res['room']}.", [res["room"]]),
        (["How much do I still owe?", "What's my balance?"], f"Your remaining balance is {balance} dollars.", [balance]),
        (["How many nights did I book?", "How long is my stay?"],
         f"You're staying {nights} night{'s' if nights != 1 else ''}.", [str(nights), _NUM_WORDS.get(nights, "#")]),
    ]
    questions, reply, facts = rng.choice(options)
    return rng.choice(questions), {"reply": reply, "answer_contains": facts}


def _wakeup_example(rng: random.Random) -> tuple[str, dict]:
    hour = rng.randint(5, 9)
    minute = rng.choice([0, 0, 15, 30, 45])
    spoken_min = {0: "", 15: " fifteen", 30: " thirty", 45: " forty five"}[minute]
    words = ["five", "six", "seven", "eight", "nine"][hour - 5]
    text = rng.choice(["Can I get a wake up call at {w}{m} AM?", "Please wake me up at {w}{m} tomorrow morning.",
                       "Set a wake-up call for {w}{m} in the morning."]).format(w=words, m=spoken_min)
    return text, {"name": "schedule_wakeup_call", "arguments": {"time": f"{hour:02d}:{minute:02d}"}}


_GENERATORS = [_order_example, _issue_example, _info_example, _reservation_example, _wakeup_example]


def generate_examples(n: int, seed: int = 0) -> list[dict]:
    """Rows: {id, text, context, tools: 'hotel'} plus either tool_calls=[...] (an action is needed)
    or reply + answer_contains (answered from the reservation context, no tool)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        gen = rng.choice(_GENERATORS)
        res = make_reservation(rng)
        row = {"id": f"hotel_{seed}_{i:06d}", "context": reservation_context(res), "tools": "hotel"}
        if gen is _reservation_example:
            text, extra = gen(rng, res)
            row.update(text=text, tool_calls=[], **extra)
        else:
            text, call = gen(rng)
            row.update(text=text, tool_calls=[call])
        rows.append(row)
    return rows


def tools_by_name(name: str | None) -> list[dict] | None:
    if name == "hotel":
        return HOTEL_TOOLS
    return None


FREE_TEXT_ARGS = {"description"}


def calls_match(pred: list[dict], gold: list[dict]) -> bool:
    def norm(c: dict) -> Any:
        # free-text fields (issue descriptions) only need to be present, not identical
        args = {k: ("<text>" if k in FREE_TEXT_ARGS else v.lower().strip() if isinstance(v, str) else v)
                for k, v in (c.get("arguments") or {}).items()}
        return c.get("name"), json.dumps(args, sort_keys=True)

    return sorted(map(norm, pred)) == sorted(map(norm, gold))


def score_example(row: dict, pred_calls: list[dict], output_text: str) -> bool:
    """Tool rows: exact call match. Context rows: no tool call and the reply states the fact."""
    if row.get("tool_calls"):
        return calls_match(pred_calls, row["tool_calls"])
    reply = output_text.lower()
    return not pred_calls and any(f.lower() in reply for f in row.get("answer_contains", []))
