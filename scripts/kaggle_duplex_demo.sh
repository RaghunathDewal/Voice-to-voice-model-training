#!/usr/bin/env bash
# Talk to the full-duplex model (checkpoints/duplex) in the browser. Works on Kaggle (2x T4) and on a GPU
# droplet: the model runs on GPU 0, the streaming listener on GPU 1 if there is one. Prints a public
# https:// link (Cloudflare quick tunnel). Use HEADPHONES: the microphone is never muted (full duplex).
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints.
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_duplex_demo.sh
# Each call is saved as a stereo wav (left = you, right = agent) in $SAVE_DIR.
# The page shows "compute / real time": it must stay below 1.0, otherwise the agent falls behind
# (then use a faster GPU, e.g. RTX 4090 / L40S / MI300X; the same script runs there).
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_duplex_demo}
SAVE_DIR=${SAVE_DIR:-$( [ -d /kaggle/working ] && echo /kaggle/working/duplex_calls || echo "$WORK/calls")}
: "${HF_TOKEN:?HF_TOKEN is not set}"
export HF_TOKEN PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/checkpoints"
cd "$REPO_DIR"
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q peft accelerate soundfile librosa jiwer huggingface_hub fastapi uvicorn websockets
python -c "from transformers import AutoModelForRNNT" 2>/dev/null || pip install -q -U "transformers>=5.13"
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

say ">>> download the duplex model"
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=["checkpoints/duplex/*"])
assert os.path.exists("checkpoints/duplex/duplex.pt"), "checkpoints/duplex not found on Hugging Face"
EOF

say ">>> starting (first start downloads Qwen3-0.6B, Nemotron and Mimi; wait for the trycloudflare.com link)"
say "    open the link in Chrome, put on headphones, press Start call and talk; the agent greets you first"
python -m s2s.cli.duplex_live --ckpt checkpoints/duplex --tunnel --save-dir "$SAVE_DIR"
