"""(vision/smoke_check.py) Read the smoke-test log of run_kaggle.sh: did training steps finish, how fast, how much memory."""

import re
import sys

t = open(sys.argv[1], errors="replace").read()
speed = [float(x) for x in re.findall(r"'train_speed\(s/it\)': '([0-9.]+)'", t)]
mem = re.findall(r"'memory\(GiB\)': '([0-9.]+)'", t)
if not speed:
    print("SMOKE TEST FAILED: no training update finished. The error is above, full log: " + sys.argv[1])
    sys.exit(1)
print(f"SMOKE TEST OK: {len(speed)} updates, {speed[-1]:.0f} s per update (average, the first one is slower), "
      f"GPU memory {mem[-1] if mem else '?'} GiB")
print("  ~20-40 s per update = good; above 80 s = rerun with IMAGE_MAX_TOKEN_NUM=512")
