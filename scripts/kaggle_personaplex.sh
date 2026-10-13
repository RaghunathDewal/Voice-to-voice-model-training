#!/usr/bin/env bash
# Test NVIDIA PersonaPlex-7B (full-duplex speech-to-speech, Moshi-based) on hotel calls, Kaggle 2x T4 (~45-75 min).
#   1. four spoken hotel calls (a guest asks 3 things each, pauses for the answers), voiced with Kokoro, plus
#      NVIDIA's own two sample calls
#   2. PersonaPlex hears each call as live (80 ms frames) with a hotel persona prompt; every frame is timed
#   3. report: speed vs the 80 ms live budget, reply delay, talking over the guest, right facts (its text and
#      Whisper on its audio), and stereo files to listen to (left guest, right agent)
# A T4 is too small and too slow for this 7B to run live (NVIDIA tests on A100 / H100): it is spread over both
# GPUs (+ CPU) and runs slower than real time here; quality and turn-taking are still judged correctly, the
# speed numbers tell what GPU it would need.
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN whose account has accepted the licence at
# https://huggingface.co/nvidia/personaplex-7b-v1 (click "Agree", approved at once).
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_personaplex.sh
# Options (before `bash`): VOICE=NATF2.pt (NATF0-3 / NATM0-3 natural, VARF0-4 / VARM0-4), DTYPE=bf16|fp16.
# Results: /kaggle/working/personaplex/summary.md and */*_mix.wav
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/personaplex || echo /tmp/personaplex_out)}
VOICE=${VOICE:-NATF2.pt}
DTYPE=${DTYPE:-bf16}
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/calls" "$OUT/run"
cd "$REPO_DIR"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> licence check"
python - <<'EOF'
import sys
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("nvidia/personaplex-7b-v1", "config.json")
    print("access to nvidia/personaplex-7b-v1: ok")
except Exception as e:
    sys.exit(f"no access ({type(e).__name__}): with the HF account of this token, open "
             "https://huggingface.co/nvidia/personaplex-7b-v1 and click Agree, then run again")
EOF

say ">>> setup (two own environments: PersonaPlex pins torch<2.5; Kokoro voices the guests)"
apt-get -qq install -y libopus-dev espeak-ng > /dev/null 2>&1 || true
pip install -q uv
[ -d /tmp/personaplex ] || git clone -q --depth 1 https://github.com/NVIDIA/personaplex /tmp/personaplex
if ! /tmp/ppenv/bin/python -c "import moshi, accelerate" 2>/dev/null; then
    uv venv -q -p 3.11 --seed /tmp/ppenv
    uv pip install -p /tmp/ppenv/bin/python -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    uv pip install -p /tmp/ppenv/bin/python -q /tmp/personaplex/moshi accelerate
fi
if ! /tmp/ktts/bin/python -c "import kokoro" 2>/dev/null; then
    [ -x /tmp/ktts/bin/python ] || uv venv -q -p 3.12 --seed /tmp/ktts
    uv pip install -p /tmp/ktts/bin/python -q torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install -p /tmp/ktts/bin/python -q "kokoro>=0.9.4" "misaki[en]" soundfile numpy
fi
pip install -q soundfile librosa

say ">>> the hotel calls (Kokoro guests)"
/tmp/ktts/bin/python s2s/eval/personaplex_calls.py --out-dir "$OUT/calls"
python - "$OUT/calls/calls.json" <<'EOF'
import json, sys
f = sys.argv[1]
calls = [c for c in json.load(open(f)) if not c["id"].startswith("nvidia_")]
calls += [  # NVIDIA's own samples, as a sanity check of the setup (no fact checks)
    {"id": "nvidia_assistant", "wav": "/tmp/personaplex/assets/test/input_assistant.wav",
     "prompt": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.", "turns": []},
    {"id": "nvidia_service", "wav": "/tmp/personaplex/assets/test/input_service.wav",
     "prompt": open("/tmp/personaplex/assets/test/prompt_service.txt").read().strip(), "turns": []}]
json.dump(calls, open(f, "w"), indent=1)
EOF

MEM=$( { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0; } | head -1)
OFFLOAD=()
if [ "${MEM:-0}" -lt 30000 ]; then  # the 7B (16.7 GB) does not fit one 16 GB GPU: spread it (no CUDA graphs then)
    OFFLOAD=(--cpu-offload); export NO_CUDA_GRAPH=1
    say "    GPU has ${MEM} MiB: spreading the model over the GPUs (+ CPU)"
fi
say ">>> PersonaPlex on the calls (voice $VOICE, $DTYPE; first run downloads 17 GB)"
/tmp/ppenv/bin/python s2s/eval/personaplex_run.py --calls "$OUT/calls/calls.json" --out-dir "$OUT/run" \
    --voice "$VOICE" --dtype "$DTYPE" "${OFFLOAD[@]}" 2>&1 | grep -v "text token" | tee "$OUT/run.log"

say ">>> report"
python -m s2s.eval.personaplex_report --calls "$OUT/calls/calls.json" --run-dir "$OUT/run" --out "$OUT/summary.md"
say "DONE: $OUT/summary.md, listen to $OUT/run/*_mix.wav (left: guest, right: agent)"
