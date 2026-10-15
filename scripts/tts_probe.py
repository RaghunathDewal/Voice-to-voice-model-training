"""Time TTS servers on phrases of new lengths (as calls produce them).

    python scripts/tts_probe.py http://127.0.0.1:7791/tts
        -> each of 6 phrases in ms, one at a time on one server, then their median (last number)
    python scripts/tts_probe.py http://127.0.0.1:7791/tts,http://127.0.0.1:7792/tts --parallel 16 --requests 48
        -> under load: `parallel` phrases in flight at once, spread over the servers:
           "median_ms p95_ms phrases_per_s"
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TEXTS = ["Of course.", "Sure, one moment please.", "Checkout is at eleven in the morning.",
         "I have asked housekeeping to bring two extra towels to your room.",
         "Breakfast is served from seven to half past ten in the lobby restaurant, every day.",
         "The pool is open from seven in the morning until ten at night, and towels are provided at the entrance."]


def one(url: str, text: str) -> float:
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=300).read()
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("urls")
    p.add_argument("--parallel", type=int, default=0)
    p.add_argument("--requests", type=int, default=48)
    a = p.parse_args()
    urls = a.urls.split(",")
    if not a.parallel:
        ms = [one(urls[0], t) for t in TEXTS]
        print(" ".join(f"{m:.0f}" for m in ms), int(statistics.median(ms)))
        return
    jobs = [(urls[i % len(urls)], f"{TEXTS[i % len(TEXTS)]} Request {i}.") for i in range(a.requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.parallel) as ex:
        ms = sorted(ex.map(lambda j: one(*j), jobs))
    wall = time.perf_counter() - t0
    print(f"{statistics.median(ms):.0f} {ms[round(0.95 * (len(ms) - 1))]:.0f} {len(ms) / wall:.1f}")


if __name__ == "__main__":
    main()
