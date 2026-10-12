#!/usr/bin/env bash
# Full-duplex experiment on one AMD MI300X droplet (DigitalOcean), from nothing to an evaluated model on
# Hugging Face in ~3 h. Runs in the background: closing the terminal, SSH or the droplet console does not stop it.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...          # write access to ThunderBlade7773/s2s-checkpoints and s2s-data-pk
#   export DO_TOKEN=dop_v1_...      # optional: the droplet deletes itself when done / failed / after MAX_HOURS
#   bash scripts/mi300x_duplex.sh start     # starts in the background and shows the log (Ctrl-C only stops viewing)
#   bash scripts/mi300x_duplex.sh log       # view the log again (after reconnecting)
#   bash scripts/mi300x_duplex.sh status    # finished stages, last training lines, GPU use
#   bash scripts/mi300x_duplex.sh stop      # stop everything (no self-delete)
#
# Stages (each runs once; `start` again on the same droplet skips finished ones):
#   setup     Python 3.12 venv (uv), ROCm PyTorch, our package, Kokoro + espeak-ng       ~10 min
#   data      10k generated two-speaker hotel calls: Kokoro speech (8 workers), Nemotron
#             features of the guest channel, Mimi codes of the agent channel              ~60-75 min
#   upload    the data to the HF data repo (duplex/...), in the background while training
#   train     Qwen3-0.6B LoRA + new layers, bf16, ends itself after TRAIN_MIN minutes       ~100 min
#   eval      50 held-out calls run frame by frame (answers, timing, interruptions, tools,
#             intelligibility) + stereo wavs to listen to -> HF data repo results/duplex  ~15 min
# Checkpoints go to Hugging Face every 15 min (checkpoints/duplex); a new droplet continues from there.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
MAX_HOURS=${MAX_HOURS:-4}                # with DO_TOKEN: the droplet is deleted after this, whatever happens
TRAIN_MIN=${TRAIN_MIN:-100}
N_TRAIN=${N_TRAIN:-10000}
N_EVAL=${N_EVAL:-200}
N_TEST=${N_TEST:-50}
PROCS=${PROCS:-8}                        # Kokoro workers sharing the GPU
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/duplex || echo "$HOME/duplex")}
VENV=$HOME/s2s-venv
LOG=$WORK/run.log
CFG="--config configs/default.yaml"
mkdir -p "$WORK"/{data,checkpoints,logs,done,tmp}

# --------------------------------------------------------------- commands
case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$WORK/run.pid")): bash $0 log"; exit 0
        fi
        # setsid + nohup: own session, no terminal; closing SSH / the console / the window does not reach it
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$WORK/run.pid"
        echo "started in the background (pid $!). Log: $LOG"
        echo "Ctrl-C below only stops VIEWING the log; the run keeps going. Later: bash $0 log"
        sleep 2; exec tail -n 50 -f "$LOG" ;;
    log)
        exec tail -n 100 -f "$LOG" ;;
    status)
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then echo "RUNNING"; else echo "NOT RUNNING"; fi
        echo "finished stages: $(ls "$WORK/done" 2>/dev/null | tr '\n' ' ')"
        grep -E "^\[|step [0-9]+ loss|\[eval|calls$" "$LOG" 2>/dev/null | tail -n 8 || true
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
    *) sed -n 2,24p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ run
: "${HF_TOKEN:?HF_TOKEN missing}"
echo $$ > "$WORK/run.pid"
rm -f "$WORK/stopped"
export HF_TOKEN CKPT_REPO DATA_REPO WORK PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
mkdir -p data/manifests
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
say "===== run started on $(hostname), work dir $WORK"

hf() {  # hf push <folder> <path> [dataset] | hf get-ckpt
    "$VENV/bin/python" - "$@" <<'EOF'
import os, sys
from huggingface_hub import HfApi, snapshot_download
api, cmd, args = HfApi(), sys.argv[1], sys.argv[2:]
ckpt, data = os.environ["CKPT_REPO"], os.environ["DATA_REPO"]
if cmd == "push":
    kind = args[2] if len(args) > 2 else "model"
    try:
        api.upload_folder(repo_id=ckpt if kind == "model" else data, repo_type=kind, folder_path=args[0],
                          path_in_repo=args[1], commit_message=f"mi300x duplex: {args[1] or 'data'}")
        print(f"[hf] {args[0]} -> {args[1]}", flush=True)
    except Exception as e:
        print(f"[hf] upload failed: {e}", flush=True); sys.exit(1)
elif cmd == "get-ckpt":
    if any(f.startswith("checkpoints/duplex/") for f in api.list_repo_files(ckpt)):
        snapshot_download(ckpt, local_dir=".", allow_patterns=["checkpoints/duplex/*"])
        print("found checkpoints/duplex on Hugging Face", flush=True)
EOF
}

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
    ( sleep $(( MAX_HOURS * 3600 )); say "MAX_HOURS reached"; hf push "$WORK/logs" logs/mi300x_duplex || true; destroy ) &
    TIMER_PID=$!
else
    say "no DO_TOKEN: the droplet is NOT deleted automatically - destroy it yourself when the run is done"
fi

cleanup() {
    local code=$?
    kill "${SYNC_PID:-0}" "${TIMER_PID:-0}" 2>/dev/null || true
    [ -n "${UPLOAD_PID:-}" ] && { say "waiting for the data upload to finish"; wait "$UPLOAD_PID" 2>/dev/null || true; }
    [ -x "$VENV/bin/python" ] && { hf push "$WORK/logs" logs/mi300x_duplex || true; }
    if [ $code -eq 0 ]; then say "===== ALL DONE"; else say "===== FAILED (exit $code): see the stage log in $WORK/logs"; fi
    destroy
}
trap cleanup EXIT

stage() {
    local name=$1; shift
    if [ -f "$WORK/done/$name" ]; then say "skip $name (done)"; return; fi
    say ">>> $name"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" -e "it/s]" || true; } | tee "$WORK/logs/$name.log"
    touch "$WORK/done/$name"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}

# ------------------------------------------------------------------ setup
setup() {
    if command -v apt-get >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl libsndfile1 espeak-ng >/dev/null 2>&1 || true
    fi
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 --managed-python --seed "$VENV"
    uv pip install -p "$VENV/bin/python" torch torchaudio --index-url "$TORCH_INDEX"
    uv pip install -p "$VENV/bin/python" -q --no-deps -e .
    uv pip install -p "$VENV/bin/python" -q "transformers>=5.13" "peft>=0.11" "accelerate>=0.30" numpy scipy \
        soundfile librosa pyyaml jiwer tqdm huggingface_hub "kokoro>=0.9" "misaki[en]"
    uv pip install -p "$VENV/bin/python" torch torchaudio --index-url "$TORCH_INDEX"   # in case kokoro pulled another torch
    "$VENV/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU (check TORCH_INDEX)"
print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item(); print("GPU OK")
from transformers import AutoModelForRNNT; print("transformers OK")
# first use downloads Kokoro, its voices and the spaCy English model: do it once here, not in 8 workers at once
from huggingface_hub import snapshot_download
snapshot_download("hexgrad/Kokoro-82M")
from kokoro import KPipeline
for lang, voice in (("a", "af_heart"), ("b", "bf_emma")):
    n = sum(len(r[2]) if isinstance(r, tuple) else len(r.audio) for r in KPipeline(lang_code=lang)("Hello there.", voice=voice))
    print(f"kokoro {lang} OK ({n} samples)")
EOF
}
stage setup setup
export PATH="$VENV/bin:$PATH"

# --------------------------------------------------------------- data
data() {
    python -m s2s.prep.duplex_data $CFG script --train "$N_TRAIN" --eval "$N_EVAL"
    python -m s2s.prep.duplex_data $CFG audio --split eval --procs "$PROCS"
    python -m s2s.prep.duplex_data $CFG audio --split train --procs "$PROCS"
    python -m s2s.prep.duplex_data $CFG features --split eval --batch-size 8
    python -m s2s.prep.duplex_data $CFG features --split train --batch-size 8 --augment-prob 0.5 --phone-prob 0.4 \
        --drop-audio
    du -sh data/duplex; df -h "$WORK" | tail -1
}
stage data data

upload_data() {  # background: the data stays reusable (e.g. Kaggle) without generating it again
    mkdir -p "$WORK/up/duplex/manifests"
    cp data/manifests/duplex_*.jsonl "$WORK/up/duplex/manifests/"
    tar -C data -cf "$WORK/up/duplex/train.tar" duplex/train
    tar -C data -cf "$WORK/up/duplex/eval.tar" duplex/eval
    hf push "$WORK/up" "" dataset && rm -rf "${WORK:?}/up" && touch "$WORK/done/upload"
}
if [ ! -f "$WORK/done/upload" ]; then
    upload_data > "$WORK/logs/upload.log" 2>&1 &
    UPLOAD_PID=$!
    say "data upload running in the background (log: $WORK/logs/upload.log)"
fi

# --------------------------------------------------------------- train
( while sleep 900; do
    [ -f checkpoints/duplex/duplex.pt ] && { hf push checkpoints/duplex checkpoints/duplex || true; }
    hf push "$WORK/logs" logs/mi300x_duplex || true
  done ) > "$WORK/logs/sync.log" 2>&1 &
SYNC_PID=$!

train() {
    hf get-ckpt || true
    local init=null
    if [ -f checkpoints/duplex/duplex.pt ] && [ ! -f checkpoints/duplex/DONE ]; then
        rm -rf checkpoints/duplex_prev && cp -r checkpoints/duplex checkpoints/duplex_prev
        init=checkpoints/duplex_prev
        echo "continuing from the last saved checkpoint"
    fi
    if [ -f checkpoints/duplex/DONE ]; then echo "checkpoints/duplex is DONE: skipped"; return; fi
    # 192 GB: no gradient checkpointing, 4 calls per micro-batch, 16 per step
    python -m s2s.train.duplex $CFG --set train_duplex.init_from=$init train_duplex.max_minutes="$TRAIN_MIN" \
        train_duplex.max_steps=12000 train_duplex.batch_size=4 train_duplex.grad_accum=4 \
        train_duplex.gradient_checkpointing=false train_duplex.num_workers=8 train_duplex.eval_every=500 \
        train_duplex.save_every=250 train_duplex.log_every=20
    touch checkpoints/duplex/DONE
    hf push checkpoints/duplex checkpoints/duplex
}
stage train train

# --------------------------------------------------------------- eval
evaluate() {
    python -m s2s.eval.duplex_eval --ckpt checkpoints/duplex --manifest data/manifests/duplex_eval.jsonl \
        --n "$N_TEST" --out "$WORK/eval"
    mkdir -p "$WORK/res" && cp "$WORK"/eval/summary.* "$WORK/res/"
    for w in $(ls "$WORK"/eval/*.wav | head -8); do cp "$w" "$WORK/res/"; done
    hf push "$WORK/res" results/duplex dataset || true
    cat "$WORK/eval/summary.md"
}
stage eval evaluate
say "results: $WORK/eval/summary.md and on Hugging Face ($DATA_REPO results/duplex, 8 stereo wavs: left = guest, right = agent)"
