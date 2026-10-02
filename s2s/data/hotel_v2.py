"""Hotel conversations, version 2: much more varied than the v1 templates.

v1 (s2s/data/hotel.py) has a handful of phrasings per intent and single turns
only; a model trained on it echoed unfamiliar requests back to the guest. v2 adds:

  * many phrasings per intent (polite, casual, terse, Indian-English), with
    random openers and closers
  * more products, problems, wake-up time formats and reservation questions
  * a whole-reservation summary ("tell me about my booking")
  * greetings, thanks, "who are you", and polite declines for out-of-scope requests
  * follow-up turns: a previous exchange is put in `history` (corrections such as
    "No, I meant my reservation details", "Also send a toothbrush", "What about the pool?")

Rows use the same fields as v1 (text, context, tools, tool_calls | reply + answer_contains)
plus `reply_after_tool` on tool rows (the spoken confirmation once the tool has run) and an optional `history` list of {"role", "content"} chat messages, so the
existing scorer (hotel.score_example) works unchanged.
"""

from __future__ import annotations

import random

from s2s.data.hotel import PROPERTY_INFO, make_reservation, reservation_context

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# product -> (plural, extra spoken forms)
_PRODUCTS = {
    "towel": ("towels", ["bath towel", "fresh towel"]),
    "pillow": ("pillows", ["extra pillow"]),
    "blanket": ("blankets", ["extra blanket"]),
    "toothbrush": ("toothbrushes", []),
    "toothpaste": ("tubes of toothpaste", []),
    "bottle of water": ("bottles of water", ["water bottle"]),
    "bathrobe": ("bathrobes", []),
    "hair dryer": ("hair dryers", []),
    "phone charger": ("phone chargers", ["charger"]),
    "coffee pod": ("coffee pods", []),
    "tea bag": ("tea bags", []),
    "roll of toilet paper": ("rolls of toilet paper", ["toilet roll"]),
    "soap": ("bars of soap", []),
    "shampoo": ("bottles of shampoo", []),
    "hanger": ("hangers", ["clothes hanger"]),
    "iron": ("irons", []),
    "ironing board": ("ironing boards", []),
    "pair of slippers": ("pairs of slippers", ["slippers"]),
    "ice bucket": ("ice buckets", []),
    "razor": ("razors", ["shaving kit"]),
}

_ISSUES = [
    ("maintenance", ["the air conditioning is not working", "the AC isn't cooling", "the AC is not working properly"],
     "air conditioning not working"),
    ("maintenance", ["the shower has no hot water", "there's no hot water", "hot water is not coming"],
     "no hot water"),
    ("maintenance", ["the TV is not turning on", "the TV remote is broken", "the television isn't working"],
     "TV not working"),
    ("maintenance", ["the toilet keeps running", "the toilet is blocked", "the flush is not working"],
     "toilet problem"),
    ("maintenance", ["the light in the bathroom is out", "the bedside lamp doesn't work", "a light bulb is fused"],
     "light not working"),
    ("maintenance", ["the Wi-Fi keeps disconnecting", "the internet is very slow in my room"], "Wi-Fi problem"),
    ("maintenance", ["the door lock is not working", "my key card isn't opening the door"], "door lock problem"),
    ("maintenance", ["the sink is leaking", "there's water leaking in the bathroom"], "water leak"),
    ("housekeeping", ["the room hasn't been cleaned today", "nobody came to clean my room"], "room not cleaned"),
    ("housekeeping", ["there are no clean sheets on the bed", "the bed sheets are dirty"], "sheets need changing"),
    ("housekeeping", ["the bathroom is dirty", "the trash hasn't been taken out"], "room needs cleaning"),
    ("noise", ["the room next door is really loud", "my neighbours are making a lot of noise"], "loud neighbours"),
    ("noise", ["there's construction noise outside my window", "it's very noisy outside"], "outside noise"),
    ("other", ["there's a strange smell in the room", "the room smells of smoke"], "bad smell in room"),
]

_TOPIC_Q = {
    "breakfast": ["What time is breakfast?", "Is breakfast included?", "Where is breakfast served?",
                  "Till what time is breakfast available?", "When does breakfast start?"],
    "wifi": ["What's the Wi-Fi password?", "How do I connect to the internet?", "What is the wifi name?",
             "Can you tell me the Wi-Fi details?"],
    "pool": ["Is the pool open now?", "Where is the swimming pool?", "What are the pool timings?",
             "Do you have a pool?"],
    "gym": ["Do you have a gym?", "When is the fitness center open?", "Where is the gym?", "What are the gym timings?"],
    "parking": ["How much is parking?", "Where can I park my car?", "Do you have valet parking?"],
    "checkout": ["What is the hotel's checkout time?", "Can I get a late checkout?", "Is late checkout possible?"],
    "restaurant": ["When is the restaurant open for dinner?", "Is there a restaurant in the hotel?",
                   "What time does the restaurant close?"],
    "spa": ["Do you have a spa?", "What are the spa hours?", "Can I book a massage at the spa?"],
}

_OPENERS = ["", "", "", "Hi, ", "Hello, ", "Hey, ", "Excuse me, ", "Good morning, ", "Good evening, ",
            "Sorry to bother you, ", "Yeah hi, ", "Hi there, "]
_CLOSERS = ["", "", "", " Thanks.", " Thank you.", " Thanks a lot.", " Please.", " That's it."]

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _spoken_date(iso: str) -> str:
    _, m, d = iso.split("-")
    return f"{_MONTHS[int(m) - 1]} {int(d)}"


def _wrap(rng: random.Random, core: str) -> str:
    opener = rng.choice(_OPENERS)
    if opener and not (core.startswith("I ") or core.startswith("I'")):
        core = core[0].lower() + core[1:]
    text = opener + core
    text = text[0].upper() + text[1:]
    return text + rng.choice(_CLOSERS)


# ------------------------------------------------------------------ intents
def _order(rng, wrap=True):
    product = rng.choice(list(_PRODUCTS))
    plural, alts = _PRODUCTS[product]
    qty = rng.choice([1, 1, 1, 2, 2, 3, 4])
    spoken_single = rng.choice([product] + alts)
    if qty == 1:
        amount = rng.choice(["an" if spoken_single[0] in "aeiou" else "a", "one", "one more", "another"])
        noun = spoken_single
    else:
        amount = rng.choice([_NUM[qty], str(qty)])
        noun = plural
    a_n = f"{amount} {noun}"
    core = rng.choice([
        "can you send {x} to my room?", "could I get {x}, please?", "I need {x}.", "please bring {x} up to the room.",
        "can someone drop off {x}?", "I'd like {x}.", "kindly send {x} to my room.", "I want {x}.",
        "can I have {x}?", "we need {x} in the room.", "send {x} please.", "is it possible to get {x}?",
        "could you arrange {x} for me?", "my room needs {x}.", "please do send {x}.",
    ]).format(x=a_n)
    call = {"name": "order_product", "arguments": {"product": product, "quantity": qty}}
    reply = f"Sure, I've ordered {_NUM.get(qty, qty)} {product if qty == 1 else plural} for you; it should arrive in about 15 minutes."
    return (_wrap(rng, core) if wrap else core), call, reply


def _issue(rng, wrap=True):
    cat, spoken_list, desc = rng.choice(_ISSUES)
    s = rng.choice(spoken_list)
    core = rng.choice([
        "{s}.", "I want to report that {s}.", "{s}, can you send someone?", "{s}, can you help?",
        "just to let you know, {s}.", "{s}. Please get it fixed.", "there's a problem, {s}.", "{s}, please check.",
    ]).format(s=s)
    call = {"name": "create_issue", "arguments": {"category": cat, "description": desc}}
    reply = "I'm sorry about that. I've reported it and someone will come to your room shortly."
    return (_wrap(rng, core) if wrap else core), call, reply


def _info(rng, wrap=True):
    topic = rng.choice(list(_TOPIC_Q))
    call = {"name": "get_property_information", "arguments": {"topic": topic}}
    q = rng.choice(_TOPIC_Q[topic])
    return (_wrap(rng, q) if wrap else q), call, PROPERTY_INFO[topic]


def _wakeup(rng, wrap=True):
    hour = rng.randint(5, 9)
    minute = rng.choice([0, 0, 15, 30, 30, 45])
    word = ["five", "six", "seven", "eight", "nine"][hour - 5]
    nxt = ["six", "seven", "eight", "nine", "ten"][hour - 5]
    forms = {0: [f"{word} AM", f"{word} o'clock", f"{hour} AM", f"{hour}:00"],
             15: [f"{word} fifteen", f"quarter past {word}", f"{hour}:15"],
             30: [f"{word} thirty", f"half past {word}", f"{hour}:30"],
             45: [f"{word} forty five", f"quarter to {nxt}", f"{hour}:45"]}[minute]
    t = rng.choice(forms)
    core = rng.choice([
        "can I get a wake up call at {t}?", "please wake me up at {t} tomorrow.", "set a wake-up call for {t}.",
        "I need a wake up call at {t} in the morning.", "wake me up at {t} please.",
        "kindly give me a wake up call at {t}.", "could you call my room at {t} to wake me up?",
    ]).format(t=t)
    call = {"name": "schedule_wakeup_call", "arguments": {"time": f"{hour:02d}:{minute:02d}"}}
    reply = f"Done, you'll get a wake-up call at {hour}:{minute:02d} AM."
    return (_wrap(rng, core) if wrap else core), call, reply


def _reservation_fact(rng, res, wrap=True):
    nights = int(res["check_out"][-2:]) - int(res["check_in"][-2:])
    bal = res["balance_due"].split(".")[0]
    first = res["guest_name"].split()[0]
    options = [
        (["What time is my checkout?", "When am I checking out?", "When do I have to leave?", "What's my checkout date?"],
         f"You check out on {_spoken_date(res['check_out'])} at 11 AM.", ["11"]),
        (["Which room am I in?", "What's my room number?", "Can you remind me of my room number?"],
         f"You're in room {res['room']}.", [res["room"]]),
        (["How much do I still owe?", "What's my balance?", "How much is pending on my bill?", "What is my bill amount?"],
         f"Your remaining balance is {bal} dollars.", [bal]),
        (["How many nights did I book?", "How long is my stay?", "How many nights am I staying?"],
         f"You're staying {nights} night{'s' if nights != 1 else ''}.", [str(nights), _NUM.get(nights, "#")]),
        (["When did I check in?", "What was my check-in date?"],
         f"You checked in on {_spoken_date(res['check_in'])}.", [_spoken_date(res["check_in"]).split()[1]]),
        (["What type of room do I have?", "Which room type did I book?"],
         f"You have a {res['room_type']} room.", [res["room_type"].split()[0]]),
        (["Whose name is the booking under?", "What name is the reservation in?"],
         f"The reservation is under {res['guest_name']}.", [first]),
    ]
    questions, reply, facts = rng.choice(options)
    q = rng.choice(questions)
    return (_wrap(rng, q) if wrap else q), None, (reply, facts)


def _reservation_summary(rng, res):
    bal = res["balance_due"].split(".")[0]
    core = rng.choice([
        "tell me about my reservation.", "what are my reservation details?", "can you give me my booking details?",
        "tell me my reservation details.", "what does my booking look like?", "can you check my reservation?",
        "I want to know my reservation details.", "please read out my booking information.",
    ])
    reply = (f"Your reservation is under {res['guest_name']}: a {res['room_type']} room, number {res['room']}, "
             f"from {_spoken_date(res['check_in'])} to {_spoken_date(res['check_out'])}, checking out at 11 AM. "
             f"Your remaining balance is {bal} dollars.")
    return _wrap(rng, core), None, (reply, [res["room"]])


def _smalltalk(rng, res):
    first = res["guest_name"].split()[0]
    options = [
        (["Hi.", "Hello.", "Hello there.", "Good morning.", "Hey.", "Good evening."],
         f"Hello {first}! How can I help you today?", ["help"]),
        (["Thank you.", "Thanks a lot.", "That's all, thanks.", "Okay thank you so much.", "Great, thanks."],
         "You're welcome! Enjoy your stay.", ["welcome"]),
        (["Who are you?", "What can you do?", "What can you help me with?"],
         "I'm the hotel's voice assistant. I can send items to your room, report problems, answer questions "
         "about the hotel and your reservation, and set wake-up calls.", ["room"]),
        (["Goodbye.", "Bye.", "Okay bye."], "Goodbye, have a great day!", ["goodbye", "great day"]),
    ]
    qs, reply, facts = rng.choice(options)
    return rng.choice(qs), None, (reply, facts)


def _out_of_scope(rng, res):
    qs = ["Can you book me a taxi to the airport?", "What's the weather like tomorrow?", "Can you play some music?",
          "Book me a table at a restaurant downtown.", "Can you order me a pizza from outside?",
          "What's the score of the cricket match?", "Can you book a flight for me?", "Set a reminder for my meeting."]
    reply = "Sorry, I can't help with that, but the front desk will be happy to assist you."
    return _wrap(rng, rng.choice(qs)), None, (reply, ["front desk"])


_ACTION = [_order, _issue, _info, _wakeup]
_CONTEXT = [_reservation_fact, _reservation_summary, _smalltalk, _out_of_scope]


def _single(rng, res):
    """One user turn -> (text, call or None, reply, answer facts)."""
    r = rng.random()
    if r < 0.62:
        text, call, reply = rng.choice(_ACTION)(rng)
        return text, call, reply, None
    fn = rng.choices(_CONTEXT, weights=[0.40, 0.25, 0.20, 0.15])[0]
    text, _, (reply, facts) = fn(rng, res)
    return text, None, reply, facts


def _follow_up(rng, res):
    """A previous exchange in history, then a follow-up / correction turn."""
    p_text, _, p_reply, _ = _single(rng, res)
    history = [{"role": "user", "content": p_text}, {"role": "assistant", "content": p_reply}]
    kind = rng.random()
    if kind < 0.3:     # correction: the assistant misunderstood
        if rng.random() < 0.6:
            _, _, (reply, facts) = _reservation_summary(rng, res)
            core = rng.choice(["No, I meant my reservation details.", "No no, I asked about my booking.",
                               "That's not what I asked. What are my reservation details?",
                               "Sorry, I meant my reservation."])
        else:
            q, _, (reply, facts) = _reservation_fact(rng, res, wrap=False)
            core = "No, I meant " + q[0].lower() + q[1:]
        return history, core, None, reply, facts
    if kind < 0.65:    # "also ..." another action
        text, call, reply = rng.choice(_ACTION)(rng, wrap=False)
        if not (text.startswith("I ") or text.startswith("I'")):
            text = text[0].lower() + text[1:]
        core = rng.choice(["Also, ", "And ", "One more thing, ", "Oh and ", "Also "]) + text
        return history, core, call, reply, None
    if kind < 0.85:    # "what about X?"
        topic = rng.choice(list(_TOPIC_Q))
        core = rng.choice(["What about the {t}?", "And the {t}?", "How about the {t}?"]).format(
            t={"wifi": "Wi-Fi", "checkout": "checkout time"}.get(topic, topic))
        call = {"name": "get_property_information", "arguments": {"topic": topic}}
        return history, core, call, PROPERTY_INFO[topic], None
    reply = "You're welcome! Enjoy your stay."
    core = rng.choice(["Okay thanks, that's all.", "Great, thank you.", "Perfect, thanks."])
    return history, core, None, reply, ["welcome"]


def generate_examples_v2(n: int, seed: int = 0, follow_up_prob: float = 0.3) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        res = make_reservation(rng)
        row = {"id": f"hotel2_{seed}_{i:06d}", "context": reservation_context(res), "tools": "hotel"}
        if rng.random() < follow_up_prob:
            history, text, call, reply, facts = _follow_up(rng, res)
            row["history"] = history
        else:
            text, call, reply, facts = _single(rng, res)
        row["text"] = text
        if call is not None:
            row["tool_calls"] = [call]
            row["reply_after_tool"] = reply  # what the assistant says once the tool has run
        else:
            row.update(tool_calls=[], reply=reply, answer_contains=facts)
        rows.append(row)
    return rows
