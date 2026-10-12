#!/usr/bin/env bash
# Live browser demo of the full-duplex model (checkpoints/duplex) on one AMD MI300X droplet (DigitalOcean).
# Runs in the background: closing SSH / the droplet console does not stop it. Prints a public https:// link.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...        # READ access to ThunderBlade7773/s2s-checkpoints is enough
#   export DO_TOKEN=dop_v1_...    # optional: the droplet deletes itself after MAX_HOURS (default 2)
#   bash scripts/mi300x_duplex_demo.sh start    # setup (first time ~5 min), then shows the log with the link
#   bash scripts/mi300x_duplex_demo.sh url      # print the public link again
#   bash scripts/mi300x_duplex_demo.sh log      # view the log
#   bash scripts/mi300x_duplex_demo.sh stop     # stop the demo (no self-delete)
#
# Open the link in Chrome, put on HEADPHONES (the microphone is never muted), press Start call and talk.
# Calls are saved as stereo wavs (left = you, right = agent) in ~/duplex_demo/calls.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
MAX_HOURS=${MAX_HOURS:-2}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$HOME/duplex_demo}
VENV=$HOME/s2s-venv
LOG=$WORK/demo.log
mkdir -p "$WORK"/{checkpoints,calls}

case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/demo.pid" ] && kill -0 "$(cat "$WORK/demo.pid")" 2>/dev/null; then
            echo "already running: bash $0 url"; exit 0
        fi
        : > "$LOG"
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$WORK/demo.pid"
        echo "started in the background. Ctrl-C below only stops viewing the log."
        sleep 2; exec tail -n 40 -f "$LOG" ;;
    url)
        grep -o "https://[a-z0-9-]*\.trycloudflare\.com" "$LOG" | tail -1 || echo "no link yet: bash $0 log"; exit 0 ;;
    log)
        exec tail -n 80 -f "$LOG" ;;
    stop)
        if [ -f "$WORK/demo.pid" ]; then
            touch "$WORK/stopped"
            pid=$(cat "$WORK/demo.pid"); pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            kill -- -"${pgid:-$pid}" 2>/dev/null || kill "$pid" 2>/dev/null || true
            echo "stopped"
        fi
        exit 0 ;;
    run) ;;
    *) sed -n 2,14p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ run
echo $$ > "$WORK/demo.pid"
rm -f "$WORK/stopped"
export HF_TOKEN PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date '+%H:%M:%S')] $*"; }

DROPLET_ID=$(curl -s --max-time 3 http://169.254.169.254/metadata/v1/id || true)
if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ]; then
    say "kill switch on: the droplet is deleted after $MAX_HOURS h (bash $0 stop cancels it)"
    ( sleep $(( MAX_HOURS * 3600 )); [ -f "$WORK/stopped" ] || curl -s -X DELETE -H "Authorization: Bearer $DO_TOKEN" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" ) &
else
    say "no DO_TOKEN: remember to destroy the droplet yourself when you are done"
fi

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import fastapi, peft" 2>/dev/null; then
    say ">>> setup (first time only)"
    if command -v apt-get >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl libsndfile1 >/dev/null 2>&1 || true
    fi
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 --managed-python --seed "$VENV"
    uv pip install -p "$VENV/bin/python" torch --index-url "$TORCH_INDEX"
    uv pip install -p "$VENV/bin/python" -q "transformers>=5.13" "peft>=0.11" accelerate numpy scipy soundfile \
        librosa pyyaml tqdm huggingface_hub fastapi uvicorn websockets
fi
uv pip install -p "$VENV/bin/python" -q --no-deps -e .
export PATH="$VENV/bin:$PATH"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"

say ">>> download the duplex model"
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=["checkpoints/duplex/*"])
assert os.path.exists("checkpoints/duplex/duplex.pt"), "checkpoints/duplex not found on Hugging Face"
EOF

say ">>> starting the demo (first start downloads Qwen3-0.6B, Nemotron and Mimi)"
say "    the public link appears below as: OPEN THIS: https://....trycloudflare.com   (or: bash $0 url)"
exec python -m s2s.cli.duplex_live --ckpt checkpoints/duplex --tunnel --save-dir "$WORK/calls"
