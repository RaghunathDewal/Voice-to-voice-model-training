#!/usr/bin/env bash
# WHOLE-PIPELINE load test on one AMD MI300X droplet: guests stream real speech in real time over the /live
# websocket to s2s/cli/batch_live.py (batched Nemotron listener + end-of-turn rule + batched 4B with adapter v4
# + a pool of Kokoro voice servers on the same GPU). Measured per concurrency level by s2s/eval/load_test.py:
# end of the guest's speech -> first reply audio (p50 / p95 / max), whole turn, failed turns. ~40-60 min.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...          # READ access to the checkpoints (write: results upload)
#   export DO_TOKEN=dop_v1_...      # optional: the droplet deletes itself when done / failed / after MAX_HOURS
#   bash scripts/mi300x_pipeline_test.sh start      # then: log | status | stop
#
# Options: MODEL=v4, LEVELS="1 8 16 32 64", TURNS=3, N_TTS=8 (Kokoro servers), MAX_CALLS=64, MAX_HOURS=2.
# Results: $WORK/results/pipeline_<model>.json + server log/stats, uploaded to logs/pipeline_test/ on Hugging Face.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
MODEL=${MODEL:-v4}
LEVELS=${LEVELS:-"1 8 16 32 64"}
TURNS=${TURNS:-3}
N_TTS=${N_TTS:-8}
MAX_CALLS=${MAX_CALLS:-64}
MAX_HOURS=${MAX_HOURS:-2}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s_pipe || echo "$HOME/s2s_pipe")}
VENV=$HOME/s2s-venv
LOG=$WORK/run.log
mkdir -p "$WORK"

case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$WORK/run.pid")): bash $0 log   (or: bash $0 stop)"; exit 0
        fi
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" MODEL="$MODEL" LEVELS="$LEVELS" TURNS="$TURNS" N_TTS="$N_TTS" \
            MAX_CALLS="$MAX_CALLS" MAX_HOURS="$MAX_HOURS" setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$WORK/run.pid"
        echo "started in the background (pid $!). Log: $LOG"
        echo "Ctrl-C below only stops VIEWING the log; the run keeps going. Later: bash $0 log"
        sleep 2; exec tail -n 50 -f "$LOG" ;;
    log)
        exec tail -n 100 -f "$LOG" ;;
    status)
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then echo "RUNNING"; else echo "NOT RUNNING"; fi
        grep -E "^\[|guests \|" "$LOG" 2>/dev/null | tail -n 14 || true
        exit 0 ;;
    stop)
        touch "$WORK/stopped"
        if [ -f "$WORK/run.pid" ]; then
            pid=$(cat "$WORK/run.pid"); pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            kill -- -"${pgid:-$pid}" 2>/dev/null || kill "$pid" 2>/dev/null || true
            rm -f "$WORK/run.pid"
        fi
        pkill -f "s2s.cli.batch_live" 2>/dev/null || true
        pkill -f "kokoro_tts_server.py" 2>/dev/null || true
        echo "stopped"
        exit 0 ;;
    run) ;;
    *) sed -n 2,14p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ run
: "${HF_TOKEN:?HF_TOKEN missing}"
echo $$ > "$WORK/run.pid"
rm -f "$WORK/stopped"
export HF_TOKEN PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
say "===== pipeline test started on $(hostname), work dir $WORK"

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
    pkill -f "s2s.cli.batch_live" 2>/dev/null || true
    pkill -f "kokoro_tts_server.py" 2>/dev/null || true
    if [ $code -eq 0 ]; then say "===== ALL DONE"; else say "===== FAILED (exit $code): see the lines above"; fi
    cp "$LOG" "$WORK/results/run.log" 2>/dev/null || true
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" - "$WORK" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
work = sys.argv[1]
if os.path.isdir(f"{work}/results"):
    HfApi().upload_folder(folder_path=f"{work}/results", path_in_repo="logs/pipeline_test", repo_id=os.environ["CKPT_REPO"],
                          commit_message="pipeline test: results")
    print("results uploaded to logs/pipeline_test", flush=True)
EOF
    rm -f "$WORK/run.pid"
    destroy
}
trap cleanup EXIT
mkdir -p "$WORK"/{checkpoints,results,logs}

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import torch, transformers, peft" 2>/dev/null; then
    say ">>> setup"
    if command -v apt-get >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl libsndfile1 >/dev/null 2>&1 || true
    fi
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 --managed-python --seed "$VENV"
    uv pip install -p "$VENV/bin/python" torch --index-url "$TORCH_INDEX"
    uv pip install -p "$VENV/bin/python" -q --no-deps -e .
    uv pip install -p "$VENV/bin/python" -q "transformers>=5.13" "peft>=0.11" accelerate numpy scipy soundfile \
        librosa pyarrow pyyaml tqdm jiwer huggingface_hub
fi
if ! "$VENV/bin/python" -c "import kokoro, fastapi, uvicorn, websockets, silero_vad" 2>/dev/null; then
    say ">>> setup: voice (Kokoro), server and client packages"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq espeak-ng >/dev/null 2>&1 || true
    uv pip install -p "$VENV/bin/python" -q "kokoro>=0.9.4" "misaki[en]" loguru fastapi uvicorn websockets silero-vad
fi
export PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.__version__, torch.cuda.get_device_name(0))"

say ">>> download: speech_llm_$MODEL"
ln -sfn "$WORK/checkpoints" checkpoints
M="$MODEL" python - <<'EOF'
import os
from huggingface_hub import snapshot_download
name = f"speech_llm_{os.environ['M']}"
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt",
                  f"checkpoints/{name}/*.json", f"checkpoints/{name}/*.yaml", f"checkpoints/{name}/lora/*"])
assert os.path.exists(f"checkpoints/{name}/adapter.pt"), f"{name} not found"
print(f"{name}: downloaded", flush=True)
EOF

say ">>> $N_TTS Kokoro voice servers on the GPU"
pkill -f "kokoro_tts_server.py" 2>/dev/null || true
URLS=""
for i in $(seq 0 $(( N_TTS - 1 ))); do
    port=$(( 7791 + i ))
    nohup python s2s/cli/kokoro_tts_server.py --device cuda:0 --port "$port" > "$WORK/logs/tts_$port.log" 2>&1 &
    URLS="$URLS${URLS:+,}http://127.0.0.1:$port/tts"
done
for i in $(seq 0 $(( N_TTS - 1 ))); do
    port=$(( 7791 + i ))
    for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break; sleep 5; done
    curl -sf "http://127.0.0.1:$port/health" > /dev/null || { tail -20 "$WORK/logs/tts_$port.log"; say "Kokoro on $port did not start"; exit 1; }
done
say "voice servers up"

say ">>> batched /live server (adapter $MODEL, up to $MAX_CALLS calls)"
pkill -f "s2s.cli.batch_live" 2>/dev/null || true
nohup python -m s2s.cli.batch_live --config configs/qwen4b_v3.yaml --speech-llm-dir "checkpoints/speech_llm_$MODEL" \
    --tts-url "$URLS" --max-calls "$MAX_CALLS" --port 7860 > "$WORK/logs/server.log" 2>&1 &
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:7860/health > /dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:7860/health || { tail -40 "$WORK/logs/server.log"; say "the server did not start"; exit 1; }
echo

say ">>> guests: real speech streamed in real time, levels $LEVELS, $TURNS turns each"
# shellcheck disable=SC2086
python -m s2s.eval.load_test --url ws://127.0.0.1:7860/live --concurrency $LEVELS --turns "$TURNS" --timeout 90 \
    --system-prompt "You are the voice concierge of the Marina View Hotel. Checkout is at 11 AM, breakfast is served from 7 to 10:30 AM in the lobby restaurant, the pool is open from 7 AM to 10 PM and the Wi-Fi password is on the key card. Be brief and friendly." \
    --out "$WORK/results/pipeline_$MODEL.json" 2>&1 | grep -v -e "Warning" -e "warn("
curl -sf http://127.0.0.1:7860/health > "$WORK/results/server_stats_$MODEL.json" || true
cp "$WORK/logs/server.log" "$WORK/results/server_$MODEL.log" 2>/dev/null || true
say "results: $WORK/results/pipeline_$MODEL.json, server stats: $WORK/results/server_stats_$MODEL.json"
