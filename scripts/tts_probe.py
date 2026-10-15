"""Time a TTS server on phrases of new lengths (as calls produce them). Prints each time in ms, the median last.

    python scripts/tts_probe.py http://127.0.0.1:7791/tts
"""

import json
import statistics
import sys
import time
import urllib.request

TEXTS = ["Of course.", "Sure, one moment please.", "Checkout is at eleven in the morning.",
         "I have asked housekeeping to bring two extra towels to your room.",
         "Breakfast is served from seven to half past ten in the lobby restaurant, every day.",
         "The pool is open from seven in the morning until ten at night, and towels are provided at the entrance."]

ms = []
for text in TEXTS:
    t0 = time.perf_counter()
    req = urllib.request.Request(sys.argv[1], data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=120).read()
    ms.append((time.perf_counter() - t0) * 1000)
print(" ".join(f"{m:.0f}" for m in ms), int(statistics.median(ms)))
