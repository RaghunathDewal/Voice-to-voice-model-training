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
#   MODEL=0.6b             the earlier compact system instead: Parakeet -> speech_llm_pk4 -> fine-tuned Qwen3-0.6B
#                          (thinker_merged_v3) -> talker_v3 -> Mimi (faster, less expressive)
#   TALKER=best            the talker checkpoint with the lowest eval loss (step 1000) instead of the final one
#   PROMPT=/path/file.txt  your own system prompt (property facts, booking); default: the built-in hotel demo
#   TOOLS=order_product,create_issue   only these tools (default: all built-in hotel tools); TOOLS=none: no tools
#   EXTRA_SETS="runtime.emit_every_frames=1"   more config overrides (space-separated key=value)
# Applications (e.g. a Gemini-Live-style bridge server) connect to wss://<link>/live: s2s/cli/live_api.py.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
MODEL=${MODEL:-4b}
if [ "$MODEL" = 0.6b ]; then
    ADAPTER=${ADAPTER:-speech_llm_pk4}; TALKER_DIR=talker_v3; THINKER=thinker_merged_v3; CFG=configs/small.yaml
else
    ADAPTER=${ADAPTER:-speech_llm_4b_v2}; TALKER_DIR=talker_4b; THINKER=""; CFG=configs/qwen4b.yaml
fi
TALKER=${TALKER:-last}
TOOLS=${TOOLS:-all}
PROMPT=${PROMPT:-}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_demo}
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN ADAPTER TALKER TALKER_DIR THINKER PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
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
repo, adapter, talker, tdir, thinker = (os.environ[k] for k in ("CKPT_REPO", "ADAPTER", "TALKER", "TALKER_DIR", "THINKER"))
pats = [f"checkpoints/{adapter}/*.pt", f"checkpoints/{adapter}/*.yaml", f"checkpoints/{adapter}/*.json",
        f"checkpoints/{tdir}/*.pt", f"checkpoints/{tdir}/*.json", f"checkpoints/{tdir}/*.yaml", f"checkpoints/{tdir}/last/*"]
if thinker:
    pats.append(f"checkpoints/{thinker}/*")
snapshot_download(repo, local_dir=".", allow_patterns=pats)
assert os.path.exists(f"checkpoints/{adapter}/adapter.pt"), f"{adapter} not found"
if talker == "last" and os.path.exists(f"checkpoints/{tdir}/last/talker.pt"):  # the final step, not the lowest loss
    shutil.copy(f"checkpoints/{tdir}/last/talker.pt", f"checkpoints/{tdir}/talker.pt")
print(f"adapter {adapter}, {tdir} ({talker})" + (f", thinker {thinker}" if thinker else ""))
EOF

SETS=()   # one --set with every override (a second --set would replace the first)
[ "$NGPU" -gt 1 ] && SETS+=(runtime.encoder_device=cuda:1)
[ -n "$THINKER" ] && SETS+=("thinker.model=checkpoints/$THINKER")
for kv in ${EXTRA_SETS:-}; do SETS+=("$kv"); done
EXTRA=()
[ ${#SETS[@]} -gt 0 ] && EXTRA+=(--set "${SETS[@]}")
[ -n "$PROMPT" ] && EXTRA+=(--system-prompt-file "$PROMPT")
say ">>> starting the live demo (first start downloads the thinker; wait for the trycloudflare.com link)"
say "    open the https://....trycloudflare.com link, allow the microphone, press Start and just talk"
python -m s2s.cli.ws_live --config "$CFG" --speech-llm-dir "checkpoints/$ADAPTER" \
    --talker-dir "checkpoints/$TALKER_DIR" --tools "$TOOLS" --tunnel "${EXTRA[@]}"
