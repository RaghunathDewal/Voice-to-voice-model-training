"""Per-stage turn times from a batch_live server log ("[turn] key=ms ..." lines): median / p95 per stage.

    python scripts/turn_timings.py server.log
"""

import re
import statistics
import sys

rows = [dict(re.findall(r"(\w+)=(-?\d+)", line)) for line in open(sys.argv[1]) if line.startswith("[turn]")]
print(f"server side, {len(rows)} turns, ms from the end-of-turn decision (endpoint_wait: silence before it):")
for key in ("listener_lag", "endpoint_wait", "listener_wait", "first_token", "first_audio", "tts_first"):
    v = sorted(int(r[key]) for r in rows if key in r)
    if v:
        print(f"  {key:14s} median {statistics.median(v):7.0f}   p95 {v[round(0.95 * (len(v) - 1))]:7.0f}")
reasons = {}
for line in open(sys.argv[1]):
    if line.startswith("[turn]"):
        m = re.search(r"reason=(\S+)", line)
        if m:
            reasons[m.group(1)] = reasons.get(m.group(1), 0) + 1
if reasons:
    print("  end-of-turn reasons:", ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
