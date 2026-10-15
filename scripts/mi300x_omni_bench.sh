#!/usr/bin/env bash
# REFERENCE benchmark on one AMD MI300X droplet: Qwen-Omni (speech in -> speech out, ONE model) served by vLLM-Omni
# (docker image vllm/vllm-omni-rocm), N guests at once sending spoken hotel questions (s2s/eval/omni_bench.py).
# To compare with our own pipeline (scripts/mi300x_pipeline_test.sh) on the same GPU. ~30-60 min per model.
# This does NOT run our 4B / adapter: vLLM-Omni serves Qwen-Omni's own audio encoder + thinker + talker.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...          # model download + results upload (write)
#   export DO_TOKEN=dop_v1_...      # optional: the droplet deletes itself when done / failed / after MAX_HOURS
#   bash scripts/mi300x_omni_bench.sh start      # then: log | status | stop
#
# Options: OMNI_MODELS="Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct", LEVELS="1 8 16 32 64", TURNS=3,
#          OMNI_IMAGE=vllm/vllm-omni-rocm:v0.28.0, MAX_TOKENS=128, MAX_HOURS=3, SERVE_ARGS="" (extra serve flags).
# Results: $WORK/results/omni_<model>.json + server logs, uploaded to logs/omni_bench/ on Hugging Face.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
OMNI_MODELS=${OMNI_MODELS:-"Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct"}
OMNI_IMAGE=${OMNI_IMAGE:-vllm/vllm-omni-rocm:v0.28.0}
LEVELS=${LEVELS:-"1 8 16 32 64"}
TURNS=${TURNS:-3}
MAX_TOKENS=${MAX_TOKENS:-128}
MAX_HOURS=${MAX_HOURS:-3}
SERVE_ARGS=${SERVE_ARGS:-}
PORT=8091
NAME=s2s-omni

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s_omni || echo "$HOME/s2s_omni")}
VENV=$HOME/omni-client-venv
LOG=$WORK/run.log
mkdir -p "$WORK"

case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$WORK/run.pid")): bash $0 log   (or: bash $0 stop)"; exit 0
        fi
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" OMNI_MODELS="$OMNI_MODELS" OMNI_IMAGE="$OMNI_IMAGE" LEVELS="$LEVELS" \
            TURNS="$TURNS" MAX_TOKENS="$MAX_TOKENS" MAX_HOURS="$MAX_HOURS" SERVE_ARGS="$SERVE_ARGS" \
            setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
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
        docker rm -f "$NAME" > /dev/null 2>&1 || true
        echo "stopped"
        exit 0 ;;
    run) ;;
    *) sed -n 2,15p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ run
: "${HF_TOKEN:?HF_TOKEN missing}"
echo $$ > "$WORK/run.pid"
rm -f "$WORK/stopped"
export HF_TOKEN PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
say "===== vLLM-Omni benchmark started on $(hostname), work dir $WORK"

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
    docker logs "$NAME" > "$WORK/results/server_last.log" 2>&1 || true
    docker rm -f "$NAME" > /dev/null 2>&1 || true
    if [ $code -eq 0 ]; then say "===== ALL DONE"; else say "===== FAILED (exit $code): see the lines above"; fi
    cp "$LOG" "$WORK/results/run.log" 2>/dev/null || true
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" - "$WORK" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
work = sys.argv[1]
if os.path.isdir(f"{work}/results"):
    HfApi().upload_folder(folder_path=f"{work}/results", path_in_repo="logs/omni_bench", repo_id=os.environ["CKPT_REPO"],
                          commit_message="vLLM-Omni benchmark: results")
    print("results uploaded to logs/omni_bench", flush=True)
EOF
    rm -f "$WORK/run.pid"
    destroy
}
trap cleanup EXIT
mkdir -p "$WORK"/{results,hf}

command -v docker >/dev/null 2>&1 || { say "docker not found: use a droplet image with docker (e.g. the ROCm / vLLM one-click)"; exit 1; }
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import soundfile, huggingface_hub" 2>/dev/null; then
    say ">>> setup: small client venv"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 --managed-python --seed "$VENV"
    uv pip install -p "$VENV/bin/python" -q numpy soundfile huggingface_hub
fi

say ">>> docker pull $OMNI_IMAGE"
docker pull -q "$OMNI_IMAGE"

for M in $OMNI_MODELS; do
    tag=$(basename "$M")
    say ">>> vLLM-Omni server: $M"
    docker rm -f "$NAME" > /dev/null 2>&1 || true
    # newer images ship a separate `vllm-omni` CLI; older ones take `vllm serve ... --omni`
    docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -p "$PORT:$PORT" -e HF_TOKEN \
        -v "$WORK/hf:/root/.cache/huggingface" --entrypoint bash "$OMNI_IMAGE" -c \
        "if command -v vllm-omni >/dev/null; then exec vllm-omni serve $M --port $PORT $SERVE_ARGS; \
         else exec vllm serve $M --omni --port $PORT $SERVE_ARGS; fi" > /dev/null
    up=0
    for _ in $(seq 1 360); do   # up to 60 min: first start downloads the weights
        if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q -f name="$NAME" | grep -q . || break
        sleep 10
    done
    docker logs "$NAME" > "$WORK/results/server_$tag.log" 2>&1 || true
    if [ "$up" != 1 ]; then
        tail -40 "$WORK/results/server_$tag.log"; say "$M: server did not start - skipped"; continue
    fi
    say "$M: server up"
    say ">>> guests: levels $LEVELS, $TURNS turns each"
    # shellcheck disable=SC2086
    "$VENV/bin/python" s2s/eval/omni_bench.py --url "http://127.0.0.1:$PORT" --model "$M" --concurrency $LEVELS \
        --turns "$TURNS" --max-tokens "$MAX_TOKENS" --out "$WORK/results/omni_$tag.json" || say "$M: client failed"
    docker logs "$NAME" > "$WORK/results/server_$tag.log" 2>&1 || true
    docker rm -f "$NAME" > /dev/null 2>&1 || true
done
say "results: $WORK/results/omni_*.json"
