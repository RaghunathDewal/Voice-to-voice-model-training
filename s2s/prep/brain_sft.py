"""Extra text fine-tuning data that keeps the thinker's own voice while it learns the hotel rules.

1. hotel self-answers: the thinker's own answers to the hotel requests (<set>_q4.jsonl from s2s.prep.distill
   --row-prompts) that are CORRECT against the gold row (<set>_nm.jsonl): same tool call, or the right fact and
   no call. These teach "keep answering like this" in the thinker's own wording, so the template replies of
   the hotel generator (used for what it got wrong) do not replace its style.
2. replay: the thinker's own answers to general speech (the public sets' _q4 rows) - the behaviour on
   everything that is not a hotel request stays the base model's.
Rows are appended to the thinker_sft_train.jsonl of s2s.prep.thinker_sft_data. Held-out rows for the checks
(s2s.eval.brain_check): --general-out (public requests + the base answer), --talker-out (reply texts).

    python -m s2s.prep.brain_sft --config configs/qwen4b_v3.yaml --hotel data/manifests/hotel5_train \\
        --replay data/manifests/{cv_train,spokenwoz_train,slurp_train}_q4.jsonl --n-replay 6000
"""

from __future__ import annotations

import json
import os
import random

from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import row_system, score_example
from s2s.text import sentence_case
from s2s.utils import read_jsonl


def as_tool_call(c: dict) -> dict:
    return c if "function" in c else {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments", {})}}


def plain_call(c: dict) -> dict:
    f = c.get("function", c)
    return {"name": f["name"], "arguments": f.get("arguments", {})}


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--hotel", required=True, help="prefix: <prefix>_nm.jsonl (gold) and <prefix>_q4.jsonl (thinker)")
    p.add_argument("--replay", nargs="*", default=[], help="public _q4.jsonl manifests (thinker answers)")
    p.add_argument("--n-replay", type=int, default=6000)
    p.add_argument("--n-general", type=int, default=100, help="held-out public requests for the general check")
    p.add_argument("--n-talker", type=int, default=40)
    p.add_argument("--train", default=None, help="default: <data_dir>/manifests/thinker_sft_train.jsonl")
    p.add_argument("--general-out", default=None, help="default: <data_dir>/manifests/brain_general_check.jsonl")
    p.add_argument("--talker-out", default=None, help="default: <data_dir>/manifests/brain_talker_check.jsonl")
    args = p.parse_args()
    cfg = config_from_args(args)
    man = os.path.join(cfg.paths.data_dir, "manifests")
    train_path = args.train or os.path.join(man, "thinker_sft_train.jsonl")
    general_out = args.general_out or os.path.join(man, "brain_general_check.jsonl")
    talker_out = args.talker_out or os.path.join(man, "brain_talker_check.jsonl")
    rng = random.Random(cfg.seed)
    default_system = cfg.thinker.system_prompt

    gold = {r["id"]: r for r in read_jsonl(args.hotel + "_nm.jsonl")}
    own = [r for r in read_jsonl(args.hotel + "_q4.jsonl") if r.get("distilled") and r["id"] in gold]
    hotel_rows, right, talker_texts = [], 0, []
    for r in own:
        g = gold[r["id"]]
        calls = [plain_call(c) for c in (r.get("tool_calls") or [])]
        reply = (r.get("reply") or "").strip()
        if not score_example(g, calls, reply):
            continue
        right += 1
        msg = {"role": "assistant", "content": reply}
        if calls:
            msg["tool_calls"] = [as_tool_call(c) for c in calls]
        elif not reply:
            continue
        hotel_rows.append({"id": f"self_{r['id']}", "source": "hotel_self", "tools": g.get("tools"),
                           "messages": [{"role": "system", "content": row_system(g, default_system)},
                                        *(g.get("history") or []),
                                        {"role": "user", "content": g["text"]}, msg]})
        if reply and not calls:
            talker_texts.append(reply)
    print(f"hotel: the thinker's own answer is right in {right} of {len(own)} rows -> {len(hotel_rows)} self rows")

    pub = [r for path in args.replay for r in read_jsonl(path) if r.get("response") and r.get("text")]
    rng.shuffle(pub)
    general = pub[: args.n_general]
    replay = [{"id": f"replay_{r['id']}", "source": "replay",
               "messages": [{"role": "system", "content": row_system(r, default_system)},
                            {"role": "user", "content": sentence_case(r["text"])},
                            {"role": "assistant", "content": r["response"].strip()}]}
              for r in pub[args.n_general: args.n_general + args.n_replay]]
    print(f"replay: {len(replay)} general requests with the thinker's own answers; {len(general)} held out")

    with open(train_path, "a") as f:
        for row in hotel_rows + replay:
            f.write(json.dumps(row) + "\n")
    with open(general_out, "w") as f:
        for r in general:
            f.write(json.dumps({"id": r["id"], "text": sentence_case(r["text"]), "response": r["response"]}) + "\n")
    rng.shuffle(talker_texts)
    with open(talker_out, "w") as f:
        for t in talker_texts[: args.n_talker]:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"appended {len(hotel_rows) + len(replay)} rows -> {train_path}; checks -> {general_out}, {talker_out}")


if __name__ == "__main__":
    main()
