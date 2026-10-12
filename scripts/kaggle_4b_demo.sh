#!/usr/bin/env bash
# Live voice demo of the Qwen3-4B system on Kaggle (2x T4), in the browser UI (s2s/cli/static/live.html):
#   Parakeet listener -> speech adapter (speech_llm_4b_v2) -> Qwen3-4B-Instruct -> talker (talker_4b) -> Mimi
# The 4B, adapter, talker and Mimi run on GPU 0; the listener on GPU 1. Prints a public https:// link
# (Cloudflare quick tunnel; browsers need https for the microphone). The cell keeps running while the demo is up.
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints.
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_4b_demo.sh
# Options (put before `bash`):
#   TALKER=best            the talker checkpoint with the lowest eval loss (step 1000) instead of the final one
#   PROMPT=/path/file.txt  your own system prompt (property facts, booking); default: the built-in hotel demo
#   TOOLS=order_product,create_issue   only these tools (default: all built-in hotel tools); TOOLS=none: no tools
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
ADAPTER=${ADAPTER:-speech_llm_4b_v2}
TALKER=${TALKER:-last}
TOOLS=${TOOLS:-all}
PROMPT=${PROMPT:-}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_demo}
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN ADAPTER TALKER PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/checkpoints"
cd "$REPO_DIR"
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q peft accelerate soundfile librosa huggingface_hub fastapi uvicorn websockets silero-vad
python -c "from transformers import ParakeetForCTC" 2>/dev/null || pip install -q -U "transformers>=4.57"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

say ">>> download the adapter and the talker"
python - <<'EOF'
import os, shutil
from huggingface_hub import snapshot_download
repo, adapter, talker = os.environ["CKPT_REPO"], os.environ["ADAPTER"], os.environ["TALKER"]
snapshot_download(repo, local_dir=".", allow_patterns=[f"checkpoints/{adapter}/*.pt", f"checkpoints/{adapter}/*.yaml",
                                                       f"checkpoints/{adapter}/*.json", "checkpoints/talker_4b/*.pt",
                                                       "checkpoints/talker_4b/*.json", "checkpoints/talker_4b/*.yaml",
                                                       "checkpoints/talker_4b/last/*"])
assert os.path.exists(f"checkpoints/{adapter}/adapter.pt"), f"{adapter} not found"
if talker == "last" and os.path.exists("checkpoints/talker_4b/last/talker.pt"):  # the final step, not the lowest loss
    shutil.copy("checkpoints/talker_4b/last/talker.pt", "checkpoints/talker_4b/talker.pt")
print(f"adapter {adapter}, talker_4b ({talker})")
EOF

EXTRA=()
[ "$NGPU" -gt 1 ] && EXTRA+=(--set runtime.encoder_device=cuda:1)
[ -n "$PROMPT" ] && EXTRA+=(--system-prompt-file "$PROMPT")
say ">>> starting the live demo (first start downloads Qwen3-4B, ~8 GB; wait for the trycloudflare.com link)"
say "    open the https://....trycloudflare.com link, allow the microphone, press Start and just talk"
python -m s2s.cli.ws_live --config configs/qwen4b.yaml --speech-llm-dir "checkpoints/$ADAPTER" \
    --talker-dir checkpoints/talker_4b --tools "$TOOLS" --tunnel "${EXTRA[@]}"
