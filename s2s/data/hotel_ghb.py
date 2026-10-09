"""Hotel conversations with the production tool set (the six tools in demo/thinker/tools_template.json):

  browse_products, place_product_order(productName, quantity), get_orders_by_reservation,
  get_order_by_id(order_id), report_unit_issue(issue_description, priority), get_reported_issues_by_reservation

Property, guest and system prompt come from hotel_v3 (random property profile, prompt formats, persona,
hand-over team); each conversation offers a random subset of the six tools. Tool flows follow the schema
descriptions: an order first looks up the catalogue (browse_products) and then orders the exact productName;
results are realistic (catalogue, order and issue lists) and the spoken reply is grounded in them.

A conversation is
    {"id", "system", "tools": [OpenAI-style schemas], "turns": [
        {"user": "...", "assistant": [{"call": {"name", "arguments"}, "result": {...}}, ..., {"say": "..."}],
         "kind": "browse|order|order_missing|orders|order_id|issue|issues|fact|stay|small_talk|cannot|other"}]}
Each assistant step is one model turn: a tool call (answered by a "tool" turn with the result) or the
spoken reply that ends the guest's turn.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from s2s.data import hotel_v3 as v3
from s2s.data.hotel_v2 import _wrap

TOOLS_FILE = Path(__file__).resolve().parents[2] / "demo/thinker/tools_template.json"
_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# catalogue entry -> (productName, description, element_type, spoken singular, spoken plural)
_CATALOGUE = [
    ("Extra Towels", "Set of 2 bath towels", "Amenity", "towel", "towels"),
    ("Extra Pillow", "Soft hypoallergenic pillow", "Amenity", "pillow", "pillows"),
    ("Extra Blanket", "Warm fleece blanket", "Amenity", "blanket", "blankets"),
    ("Coffee Pods", "Box of 10", "Supplies", "box of coffee pods", "boxes of coffee pods"),
    ("Tea Selection", "Assorted tea bags", "Supplies", "box of tea", "boxes of tea"),
    ("Bottled Water", "Pack of 6", "Supplies", "pack of water", "packs of water"),
    ("Toiletry Kit", "Toothbrush, toothpaste and soap", "Amenity", "toiletry kit", "toiletry kits"),
    ("Bathrobe", "Cotton bathrobe", "Amenity", "bathrobe", "bathrobes"),
    ("Firewood Bundle", "Bundle of dry firewood", "Add-on", "bundle of firewood", "bundles of firewood"),
    ("BBQ Gas Refill", "Gas cylinder for the barbecue", "Add-on", "gas refill", "gas refills"),
    ("Late Night Snack Box", "Chips, cookies and juice", "Add-on", "snack box", "snack boxes"),
    ("Breakfast Basket", "Bread, eggs, jam and coffee", "Add-on", "breakfast basket", "breakfast baskets"),
    ("Baby Cot", "Travel cot with linen", "Amenity", "baby cot", "baby cots"),
    ("Kitchen Starter Pack", "Dish soap, sponge and tea towels", "Supplies", "kitchen pack", "kitchen packs"),
    ("Bicycle Rental", "One bike for one day", "Service", "bike", "bikes"),
    ("Wine Bottle", "Bottle of house red or white", "Add-on", "bottle of wine", "bottles of wine"),
]
# things guests ask for that are rarely in a catalogue
_NOT_SOLD = [("hair dryer", "hair dryers"), ("phone charger", "phone chargers"), ("iron", "irons"),
             ("pizza", "pizzas"), ("umbrella", "umbrellas")]
# (spoken variants, issue_description, priority)
_ISSUES = [
    (["the shower has no hot water", "there's no hot water", "hot water is not coming"],
     "The shower has no hot water.", "High"),
    (["the heating is not working", "the heater won't turn on", "it's freezing, the heating is off"],
     "The heating is not working.", "High"),
    (["the air conditioning is not working", "the AC isn't cooling"], "The air conditioning is not cooling.", "Medium"),
    (["the barbecue won't light", "the gas barbecue isn't working"], "The gas barbecue does not light.", "Medium"),
    (["the TV is not turning on", "the television isn't working"], "The TV does not turn on.", "Low"),
    (["the toilet keeps running", "the toilet is blocked"], "The toilet is blocked or keeps running.", "High"),
    (["the bathroom light is out", "a light bulb is broken"], "A light in the bathroom is not working.", "Low"),
    (["the Wi-Fi keeps disconnecting", "the internet doesn't work in our lodge"], "The Wi-Fi keeps disconnecting.",
     "Medium"),
    (["the front door lock is stuck", "my key card doesn't open the door"], "The door lock does not open.", "High"),
    (["the kitchen sink is leaking", "there's water leaking under the sink"], "The kitchen sink is leaking.", "High"),
    (["the fridge isn't cold", "the refrigerator stopped working"], "The fridge is not cooling.", "Medium"),
    (["the dishwasher won't start", "the dishwasher is broken"], "The dishwasher does not start.", "Low"),
    (["there are no clean sheets", "the bed sheets are dirty"], "The bed sheets need to be changed.", "Medium"),
]
_STATUSES_ORDER = ["Pending", "In progress", "Delivered", "Out for delivery"]
_STATUSES_ISSUE = ["Open", "Technician assigned", "In progress", "Resolved"]
_TOOL_WEIGHT = {"browse_products": 0.9, "place_product_order": 0.85, "get_orders_by_reservation": 0.6,
                "get_order_by_id": 0.5, "report_unit_issue": 0.85, "get_reported_issues_by_reservation": 0.6}


def ghb_tools() -> dict[str, dict]:
    return {t["function"]["name"]: t for t in json.loads(TOOLS_FILE.read_text(encoding="utf-8"))}


def call_text(calls: list[dict]) -> str:
    """LFM2 tool-call syntax: <|tool_call_start|>[name(arg="value", n=2)]<|tool_call_end|>."""
    parts = []
    for c in calls:
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (c.get("arguments") or {}).items())
        parts.append(f"{c['name']}({args})")
    return "<|tool_call_start|>[" + ", ".join(parts) + "]<|tool_call_end|>"


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class _Ghb:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.ctx = v3._Ctx(rng)  # property, stay, persona, hand-over, system prompt
        pool = ghb_tools()
        names = [n for n, w in _TOOL_WEIGHT.items() if rng.random() < w] if rng.random() > 0.05 else []
        self.tools = {n: pool[n] for n in names}
        self.catalogue = rng.sample(_CATALOGUE, rng.randint(4, 9))
        self.orders = [{"order_id": 8000 + rng.randint(1, 99), "status": rng.choice(_STATUSES_ORDER),
                        "items": [rng.choice(_CATALOGUE)[0]]} for _ in range(rng.choice([0, 1, 1, 2, 3]))]
        self.issues = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            _, desc, _ = rng.choice(_ISSUES)
            self.issues.append({"issue_id": 500 + rng.randint(1, 99), "description": desc,
                                "status": rng.choice(_STATUSES_ISSUE)})
        self.next_order = 9000 + rng.randint(1, 900)
        self.next_issue = 600 + rng.randint(1, 300)

    @property
    def h(self) -> str:
        return self.ctx.handover

    def cannot(self) -> str:
        return self.ctx.can_not()

    def browse_result(self) -> dict:
        return {"products": [{"productName": p[0], "description": p[1], "element_type": p[2]} for p in self.catalogue]}

    # ---------------------------------------------------------------- intents
    def browse(self) -> dict:
        rng = self.rng
        q = rng.choice(["What can I order?", "What do you have that you can send to us?", "What can I get delivered?",
                        "Do you have anything I can order to the {u}?", "What add-ons can I buy?",
                        "What's on the menu of things you can bring?"]).replace("{u}", self.ctx.unit_word)
        if "browse_products" not in self.tools:
            return {"user": _wrap(rng, q), "assistant": [{"say": self.cannot()}], "kind": "cannot"}
        names = [p[0] for p in self.catalogue]
        shown = names[:4]
        say = rng.choice(["You can order {x}. Would you like any of these?", "We have {x}. What would you like?",
                          "I can send {x}. Just tell me what you need."]).format(x=_join(shown))
        return {"user": _wrap(rng, q), "kind": "browse",
                "assistant": [{"call": {"name": "browse_products", "arguments": {}}, "result": self.browse_result()},
                              {"say": say}]}

    def order(self) -> dict:
        rng = self.rng
        sold = rng.random() < 0.85
        if sold:
            name, _, _, one, many = rng.choice(self.catalogue)
        else:
            name, (one, many) = None, rng.choice(_NOT_SOLD)
        qty = rng.choice([1, 1, 1, 2, 2, 3])
        amount = rng.choice(["a", "one", "another"]) if qty == 1 else rng.choice([_NUM[qty], str(qty)])
        noun = one if qty == 1 else many
        if amount == "a" and noun[0] in "aeiou":
            amount = "an"
        core = rng.choice(["can you send {x} to the {u}?", "could I get {x}, please?", "I'd like to order {x}.",
                           "we need {x}.", "please bring {x}.", "can I have {x}?", "I want {x}."]
                          ).format(x=f"{amount} {noun}", u=self.ctx.unit_word)
        user = _wrap(rng, core)
        if "place_product_order" not in self.tools:
            return {"user": user, "assistant": [{"say": self.cannot()}], "kind": "cannot"}
        steps = []
        if "browse_products" in self.tools:
            steps.append({"call": {"name": "browse_products", "arguments": {}}, "result": self.browse_result()})
            if not sold:
                options = _join([p[0] for p in self.catalogue[:3]])
                steps.append({"say": f"I'm sorry, {many} aren't something I can order here. I can send {options}, "
                                     f"or {self.h} may be able to help."})
                return {"user": user, "assistant": steps, "kind": "order_missing"}
        elif not sold:
            name = noun.title()
        self.next_order += 1
        args = {"productName": name, "quantity": qty}
        steps.append({"call": {"name": "place_product_order", "arguments": args},
                      "result": {"success": True, "order_id": self.next_order, "status": "Pending"}})
        say = rng.choice(["Done, I've ordered {q} {n} for you. Your order number is {o}.",
                          "Sure, {q} {n} {v} on the way. The order number is {o}.",
                          "I've placed the order for {q} {n}; it's order {o}."]).format(
            q=_NUM[qty], n=name, o=self.next_order, v="is" if qty == 1 else "are")
        steps.append({"say": say})
        self.orders.append({"order_id": self.next_order, "status": "Pending", "items": [name]})
        return {"user": user, "assistant": steps, "kind": "order"}

    def orders_status(self) -> dict:
        rng = self.rng
        q = rng.choice(["What's the status of my orders?", "Did my order arrive yet?", "Can you check my orders?",
                        "Where is my order?", "What have I ordered so far?"])
        if "get_orders_by_reservation" not in self.tools:
            return {"user": _wrap(rng, q), "assistant": [{"say": self.cannot()}], "kind": "cannot"}
        if not self.orders:
            say = "You don't have any orders yet. Would you like to order something?"
        else:
            say = " ".join(f"Order {o['order_id']} for {o['items'][0]} is {o['status'].lower()}." for o in self.orders[:3])
        return {"user": _wrap(rng, q), "kind": "orders",
                "assistant": [{"call": {"name": "get_orders_by_reservation", "arguments": {}},
                               "result": {"orders": self.orders}}, {"say": say}]}

    def order_by_id(self) -> dict:
        rng = self.rng
        o = rng.choice(self.orders) if self.orders and rng.random() < 0.8 else \
            {"order_id": 8000 + rng.randint(1, 99), "status": rng.choice(_STATUSES_ORDER),
             "items": [rng.choice(_CATALOGUE)[0]]}
        q = rng.choice(["Can you check order {i}?", "What's the status of order number {i}?",
                        "Is order {i} on its way?", "Any news on order {i}?"]).format(i=o["order_id"])
        if "get_order_by_id" not in self.tools:
            if "get_orders_by_reservation" in self.tools:
                known = [x for x in self.orders if x["order_id"] == o["order_id"]]
                say = (f"Order {o['order_id']} for {o['items'][0]} is {o['status'].lower()}." if known else
                       f"I can't find order {o['order_id']} on your reservation. {self.h[0].upper() + self.h[1:]} can check it for you.")
                return {"user": _wrap(rng, q), "kind": "orders",
                        "assistant": [{"call": {"name": "get_orders_by_reservation", "arguments": {}},
                                       "result": {"orders": self.orders}}, {"say": say}]}
            return {"user": _wrap(rng, q), "assistant": [{"say": self.cannot()}], "kind": "cannot"}
        result = {"order_id": o["order_id"], "status": o["status"],
                  "line_items": [{"productName": o["items"][0], "quantity": 1}]}
        say = f"Order {o['order_id']} for {o['items'][0]} is {o['status'].lower()}."
        return {"user": _wrap(rng, q), "kind": "order_id",
                "assistant": [{"call": {"name": "get_order_by_id", "arguments": {"order_id": o["order_id"]}},
                               "result": result}, {"say": say}]}

    def issue(self) -> dict:
        rng = self.rng
        spoken, desc, prio = rng.choice(_ISSUES)
        s = rng.choice(spoken)
        core = rng.choice(["{s}.", "I want to report that {s}.", "{s}, can you send someone?", "{s}, can you help?",
                           "just to let you know, {s}.", "there's a problem, {s}."]).format(s=s)
        user = _wrap(rng, core)
        if "report_unit_issue" not in self.tools:
            return {"user": user, "assistant": [{"say": self.cannot()}], "kind": "cannot"}
        self.next_issue += 1
        self.issues.append({"issue_id": self.next_issue, "description": desc, "status": "Open"})
        say = rng.choice(["I'm sorry about that. I've reported it, issue number {i}, and someone will look at it soon.",
                          "Sorry for the trouble. It's reported as issue {i} and the team has been notified.",
                          "I've logged that for you as issue {i}; maintenance will be in touch shortly."]).format(
            i=self.next_issue)
        return {"user": user, "kind": "issue",
                "assistant": [{"call": {"name": "report_unit_issue",
                                        "arguments": {"issue_description": desc, "priority": prio}},
                               "result": {"success": True, "issue_id": self.next_issue, "status": "Open"}},
                              {"say": say}]}

    def issues_status(self) -> dict:
        rng = self.rng
        q = rng.choice(["Is there any update on the problem I reported?", "What's happening with my issue?",
                        "Has anyone looked at the problem yet?", "Can you check my reported issues?"])
        if "get_reported_issues_by_reservation" not in self.tools:
            return {"user": _wrap(rng, q), "assistant": [{"say": self.cannot()}], "kind": "cannot"}
        if not self.issues:
            say = "I don't see any reported issues on your reservation. Is something not working?"
        else:
            say = " ".join(f"Issue {i['issue_id']}, {i['description'][0].lower() + i['description'][1:-1]}, is "
                           f"{i['status'].lower()}." for i in self.issues[:2])
        return {"user": _wrap(rng, q), "kind": "issues",
                "assistant": [{"call": {"name": "get_reported_issues_by_reservation", "arguments": {}},
                               "result": {"issues": self.issues}}, {"say": say}]}

    def info(self) -> dict:
        """Property facts, the guest's stay, small talk, staff-only / emergency / out-of-scope (hotel_v3)."""
        rng, ctx = self.rng, self.ctx
        r = rng.random()
        if r < 0.62:
            if rng.random() < 0.78 and ctx.profile:
                topic = rng.choice(list(ctx.profile))
                sentence, _ = ctx.profile[topic]
                return {"user": _wrap(rng, rng.choice(v3._TOPIC_Q[topic])), "assistant": [{"say": sentence}],
                        "kind": "fact"}
            topic = rng.choice([t for t in v3._TOPIC_Q if t not in ctx.profile] or list(v3._TOPIC_Q))
            if topic in ctx.profile:
                return {"user": rng.choice(v3._TOPIC_Q[topic]), "assistant": [{"say": ctx.profile[topic][0]}],
                        "kind": "fact"}
            say = rng.choice([f"I'm sorry, I don't have that information, but {self.h} can help.",
                              f"I don't have details on that. {self.h[0].upper() + self.h[1:]} will know."])
            return {"user": rng.choice(v3._TOPIC_Q[topic]), "assistant": [{"say": say}], "kind": "unknown"}
        if r < 0.86:
            t = v3._stay_turn(ctx, True)
            return {"user": t["text"], "assistant": [{"say": t["reply"]}], "kind": "stay"}
        if r < 0.94:
            t = v3._small_talk(ctx)
            if "I can " in t["reply"]:  # "who are you / what can you do": describe these tools
                caps = [c for n, c in [("place_product_order", f"order things to your {ctx.unit_word}"),
                                       ("get_orders_by_reservation", "check your orders"),
                                       ("report_unit_issue", "report problems"),
                                       ("get_reported_issues_by_reservation", "follow up on reported problems")]
                        if n in self.tools] + ["answer questions about the property and your stay"]
                t["reply"] = t["reply"].split(". I can ")[0] + f". I can {_join(caps)}."
            return {"user": t["text"], "assistant": [{"say": t["reply"]}], "kind": "small_talk"}
        t = v3._other(ctx, True)
        return {"user": t["text"], "assistant": [{"say": t["reply"]}], "kind": "other"}

    def turn(self) -> dict:
        """System-prompt following is the priority: ~55% of turns are answered from the prompt (property facts,
        the guest's booking, "not in the prompt -> hand over", persona), ~45% exercise the tools."""
        r = self.rng.random()
        for p, fn in [(0.07, self.browse), (0.19, self.order), (0.25, self.orders_status), (0.30, self.order_by_id),
                      (0.41, self.issue), (0.45, self.issues_status)]:
            if r < p:
                return fn()
        return self.info()


def generate_ghb(n: int, seed: int = 0, max_turns: int = 3) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        g = _Ghb(rng)
        k = 1 if rng.random() < 0.5 else rng.randint(2, max_turns)
        out.append({"id": f"ghb_{seed}_{i:06d}", "system": g.ctx.system, "tools": list(g.tools.values()),
                    "turns": [g.turn() for _ in range(k)]})
    return out


def from_v3(row: dict) -> dict:
    """hotel_v3 row (generic tool names) -> the same conversation format."""
    tools = row.get("tools") or []
    backend = v3.GenericBackend(tools)
    turns = []
    hist = row.get("history") or []
    for u, a in zip(hist[::2], hist[1::2]):
        turns.append({"user": u["content"], "assistant": [{"say": a["content"]}], "kind": "history"})
    if row.get("tool_calls"):
        steps = [{"call": c, "result": backend.execute(c)} for c in row["tool_calls"]]
        turns.append({"user": row["text"], "assistant": steps + [{"say": row["reply_after_tool"]}], "kind": "v3_tool"})
    else:
        turns.append({"user": row["text"], "assistant": [{"say": row["reply"]}], "kind": "v3_reply"})
    return {"id": row["id"], "system": row["system"], "tools": tools, "turns": turns}


def generate_mixed(n_ghb: int, n_v3: int, seed: int = 0) -> list[dict]:
    rows = generate_ghb(n_ghb, seed) + [from_v3(r) for r in v3.generate_examples_v3(n_v3, seed=seed + 1)]
    random.Random(seed).shuffle(rows)
    return rows


if __name__ == "__main__":
    import sys

    for conv in generate_ghb(int(sys.argv[1]) if len(sys.argv) > 1 else 3, seed=7):
        print(f"--- {conv['id']} tools={[t['function']['name'] for t in conv['tools']]}")
        for t in conv["turns"]:
            print(f"  guest [{t['kind']}]: {t['user']}")
            for s in t["assistant"]:
                print(f"    -> {call_text([s['call']])}  result={json.dumps(s['result'])[:100]}" if "call" in s
                      else f"    says: {s['say']}")
