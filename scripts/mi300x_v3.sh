#!/usr/bin/env bash
# Adapter v3 stage B (PART=polish of scripts/kaggle_v3.sh) on one AMD MI300X droplet (DigitalOcean), after
# PART=align finished on Kaggle. Runs in the background: closing SSH / the droplet console does not stop it.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...          # WRITE access to ThunderBlade7773/s2s-checkpoints and s2s-data-pk
#   export DO_TOKEN=dop_v1_...      # optional: the droplet deletes itself when done / failed / after MAX_HOURS
#   bash scripts/mi300x_v3.sh start     # starts in the background and shows the log (Ctrl-C only stops viewing)
#   bash scripts/mi300x_v3.sh log       # view the log again (after reconnecting)
#   bash scripts/mi300x_v3.sh status    # running or not, last training / test lines, GPU use
#   bash scripts/mi300x_v3.sh stop      # stop everything (no self-delete)
#
#   setup     Python 3.12 venv (uv) with ROCm PyTorch + our package                          ~10 min
#   check     checkpoints/speech_llm_v3a/DONE must be on Hugging Face (align finished)
#   polish    download the v3 data, train TRAIN_MIN minutes (batch 32 = the Kaggle 4 x 8),     ~TRAIN_MIN + 20 min
#             checkpoint to HF every 20 min (a new droplet continues from it)
#   test      the Gate 1 speech-vs-text test on the same clips as the Nemotron baseline       ~40 min
# Result: checkpoints/speech_llm_v3 and logs/v3_polish (summaries) on Hugging Face. Defaults: ~3.5 h.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
MAX_HOURS=${MAX_HOURS:-5}                # with DO_TOKEN: the droplet is deleted after this, whatever happens
TRAIN_MIN=${TRAIN_MIN:-150}              # polish training minutes
TEST_MIN=${TEST_MIN:-45}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s_v3 || echo "$HOME/s2s_v3")}
VENV=$HOME/s2s-venv
LOG=$WORK/run.log
mkdir -p "$WORK"

case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$WORK/run.pid")): bash $0 log"; exit 0
        fi
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$WORK/run.pid"
        echo "started in the background (pid $!). Log: $LOG"
        echo "Ctrl-C below only stops VIEWING the log; the run keeps going. Later: bash $0 log"
        sleep 2; exec tail -n 50 -f "$LOG" ;;
    log)
        exec tail -n 100 -f "$LOG" ;;
    status)
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then echo "RUNNING"; else echo "NOT RUNNING"; fi
        grep -E "^\[|step [0-9]+ .*loss|\[eval|same answer" "$LOG" 2>/dev/null | tail -n 10 || true
        (rocm-smi --showuse --showmemuse 2>/dev/null || nvidia-smi 2>/dev/null) | grep -E "GPU|%" | head -6 || true
        exit 0 ;;
    stop)
        if [ -f "$WORK/run.pid" ]; then
            touch "$WORK/stopped"
            pid=$(cat "$WORK/run.pid"); pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            kill -- -"${pgid:-$pid}" 2>/dev/null || kill "$pid" 2>/dev/null || true
            echo "stopped"
        fi
        exit 0 ;;
    run) ;;
    *) sed -n 2,20p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ run
: "${HF_TOKEN:?HF_TOKEN missing}"
echo $$ > "$WORK/run.pid"
rm -f "$WORK/stopped"
export HF_TOKEN PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
say "===== run started on $(hostname), work dir $WORK"

# kill switch: with DO_TOKEN the droplet is deleted when the run ends (done or failed) and after MAX_HOURS
DROPLET_ID=$(curl -s --max-time 3 http://169.254.169.254/metadata/v1/id || true)
destroy() {
    if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ] && [ ! -f "$WORK/stopped" ]; then
        say "deleting droplet $DROPLET_ID in 120 s (run 'stop' to cancel)"; sleep 120
        [ -f "$WORK/stopped" ] || curl -s -X DELETE -H "Authorization: Bearer $DO_TOKEN" \
            "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
    fi
}
if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ]; then
    say "kill switch on: droplet $DROPLET_ID is deleted at the end or after $MAX_HOURS h"
    ( sleep $(( MAX_HOURS * 3600 )); say "MAX_HOURS reached"; destroy ) &
    TIMER_PID=$!
else
    say "no DO_TOKEN: the droplet is NOT deleted automatically - destroy it yourself when the run is done"
fi
cleanup() {
    local code=$?
    kill "${TIMER_PID:-0}" 2>/dev/null || true
    if [ $code -eq 0 ]; then say "===== ALL DONE"; else say "===== FAILED (exit $code): see the lines above"; fi
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" - "$LOG" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo="logs/v3_polish/mi300x_run.log",
                    repo_id=os.environ["CKPT_REPO"], commit_message="mi300x v3 polish: run log")
EOF
    destroy
}
trap cleanup EXIT

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import torch, transformers, peft, pyarrow, librosa, jiwer" 2>/dev/null; then
    say ">>> setup"
    if command -v apt-get >/dev/null 2>&1; then   # optional; broken third-party apt sources must not stop us
        DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl libsndfile1 >/dev/null 2>&1 || true
    fi
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 --managed-python --seed "$VENV"
    # PyTorch for the GPU first, then our package WITHOUT deps (pip never swaps in another torch), then the deps
    uv pip install -p "$VENV/bin/python" torch --index-url "$TORCH_INDEX"
    uv pip install -p "$VENV/bin/python" -q --no-deps -e .
    uv pip install -p "$VENV/bin/python" -q "transformers>=5.13" "peft>=0.11" accelerate numpy scipy soundfile \
        librosa pyarrow pyyaml tqdm jiwer huggingface_hub
fi
export PATH="$VENV/bin:$PATH"   # kaggle_v3.sh's `python` / `pip` are the venv's
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU (check TORCH_INDEX)"
print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item(); print("GPU OK")
EOF

say ">>> check: align finished on Kaggle"
python - <<'EOF'
import os, sys
from huggingface_hub import HfApi
files = HfApi().list_repo_files(os.environ["CKPT_REPO"])
if "checkpoints/speech_llm_v3a/DONE" not in files:
    print("checkpoints/speech_llm_v3a/DONE is not on Hugging Face: PART=align has not finished yet", flush=True)
    sys.exit(1)
print("align is done", flush=True)
EOF

say ">>> polish: $TRAIN_MIN min training + test (scripts/kaggle_v3.sh PART=polish)"
PART=polish WORK="$WORK" OUT="$WORK/out_polish" BUDGET_MIN=$(( 25 + TRAIN_MIN + TEST_MIN )) TEST_MIN="$TEST_MIN" \
    POLISH_BATCH=32 POLISH_ACCUM=1 TEST_PAR=3 bash scripts/kaggle_v3.sh
