#!/usr/bin/env bash
# Load test of the batched thinker (s2s/runtime/batch_engine.py) for adapter v4 and v3 on one AMD MI300X droplet
# (DigitalOcean): N simulated hotel calls at once (1, 8, 16, 32, 64), first-token / first-spoken-token latency,
# reply speed per call, batch size, GPU memory. ~45-60 min in all (~$2-3). Runs in the background.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...          # READ access to ThunderBlade7773/s2s-checkpoints and s2s-data-pk (write: results upload)
#   export DO_TOKEN=dop_v1_...      # optional: the droplet deletes itself when done / failed / after MAX_HOURS
#   bash scripts/mi300x_load_test.sh start     # starts in the background and shows the log (Ctrl-C only stops viewing)
#   bash scripts/mi300x_load_test.sh log | status | stop
#
#   setup     Python 3.12 venv (uv) with ROCm PyTorch + our package                          ~10 min
#   download  speech_llm_v4 + speech_llm_v3, the hotel eval turns (manifest + stored listener features)
#   test      per model: one warm-up call, then each level LEVELS (simultaneous calls)    ~10-15 min each
# Results: $WORK/results/load_<model>/summary.md (+ results.json), uploaded to logs/load_test/ on Hugging Face.
# Options: MODELS="v4 v3", LEVELS=1,8,16,32,64, TURNS=4, MAX_TOKENS=4096, MAX_HOURS=2.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints DATA_REPO=$HF_USER/s2s-data-pk
MODELS=${MODELS:-"v4 v3"}
LEVELS=${LEVELS:-1,8,16,32,64}
TURNS=${TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-4096}
MAX_HOURS=${MAX_HOURS:-2}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s_load || echo "$HOME/s2s_load")}
VENV=$HOME/s2s-venv
LOG=$WORK/run.log
mkdir -p "$WORK"

case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$WORK/run.pid")): bash $0 log"; exit 0
        fi
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" MODELS="$MODELS" LEVELS="$LEVELS" TURNS="$TURNS" \
            MAX_TOKENS="$MAX_TOKENS" MAX_HOURS="$MAX_HOURS" setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$WORK/run.pid"
        echo "started in the background (pid $!). Log: $LOG"
        echo "Ctrl-C below only stops VIEWING the log; the run keeps going. Later: bash $0 log"
        sleep 2; exec tail -n 50 -f "$LOG" ;;
    log)
        exec tail -n 100 -f "$LOG" ;;
    status)
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then echo "RUNNING"; else echo "NOT RUNNING"; fi
        grep -E "^\[|RESULT" "$LOG" 2>/dev/null | tail -n 14 || true
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
    *) sed -n 2,18p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ run
: "${HF_TOKEN:?HF_TOKEN missing}"
echo $$ > "$WORK/run.pid"
rm -f "$WORK/stopped"
export HF_TOKEN PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
say "===== load test started on $(hostname), work dir $WORK"

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
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" - "$WORK" "$LOG" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
work, log = sys.argv[1], sys.argv[2]
api = HfApi()
api.upload_file(path_or_fileobj=log, path_in_repo="logs/load_test/mi300x_run.log", repo_id=os.environ["CKPT_REPO"],
                commit_message="load test: run log")
if os.path.isdir(f"{work}/results"):
    api.upload_folder(folder_path=f"{work}/results", path_in_repo="logs/load_test", repo_id=os.environ["CKPT_REPO"],
                      commit_message="load test: results")
print("results uploaded to logs/load_test", flush=True)
EOF
    destroy
}
trap cleanup EXIT

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
export PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU (check TORCH_INDEX)"
print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
EOF

say ">>> download: checkpoints and the hotel eval turns"
mkdir -p "$WORK"/{data,checkpoints,results,tmp}
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
WANT="$MODELS" WORK="$WORK" python - <<'EOF'
import os, subprocess
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
ckpt, data, work = os.environ["CKPT_REPO"], os.environ["DATA_REPO"], os.environ["WORK"]
api = HfApi()
files = api.list_repo_files(ckpt)
for m in os.environ["WANT"].split():
    name = f"speech_llm_{m}"
    if not any(f.startswith(f"checkpoints/{name}/") for f in files):
        raise SystemExit(f"checkpoints/{name} is not on Hugging Face")
    snapshot_download(ckpt, local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt", f"checkpoints/{name}/*.json",
                                                           f"checkpoints/{name}/*.yaml", f"checkpoints/{name}/lora/*"])
    print(f"{name}: downloaded", flush=True)
snapshot_download(data, repo_type="dataset", local_dir=".", allow_patterns=["v3/manifests/hotel5_eval_q4.jsonl"])
os.makedirs("data/manifests", exist_ok=True)
subprocess.run("cp v3/manifests/hotel5_eval_q4.jsonl data/manifests/", shell=True, check=True)
if not os.path.isdir("data/features/hotel5_eval_nm"):
    t = hf_hub_download(data, "v3/feat-hotel5_eval_nm.tar", repo_type="dataset", local_dir=f"{work}/tmp")
    subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
print("hotel eval turns ready", flush=True)
EOF

for M in $MODELS; do
    say ">>> load test speech_llm_$M (levels $LEVELS, $TURNS turns per call)"
    python -m s2s.eval.batch_load_test --config configs/qwen4b_v3.yaml --speech-llm-dir "checkpoints/speech_llm_$M" \
        --manifest data/manifests/hotel5_eval_q4.jsonl --levels "$LEVELS" --turns "$TURNS" --max-tokens "$MAX_TOKENS" \
        --label "adapter $M" --out "$WORK/results/load_$M" 2>&1 | grep -v -e "Warning" -e "warn("
done
say "summaries: $WORK/results/load_*/summary.md"
