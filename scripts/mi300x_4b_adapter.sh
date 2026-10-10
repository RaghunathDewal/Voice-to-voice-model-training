#!/usr/bin/env bash
# Continue the Qwen3-4B speech adapter on a fresh AMD MI300X droplet (DigitalOcean), from nothing to a
# tested checkpoint on Hugging Face. Runs in the background: closing the terminal / SSH does not stop it.
#
#   git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training.git ~/s2s && cd ~/s2s
#   export HF_TOKEN=hf_...          # write access to ThunderBlade7773/s2s-checkpoints (+ s2s-data-pk)
#   export DO_TOKEN=dop_v1_...      # optional: the droplet deletes itself when done / failed / after MAX_HOURS
#   bash scripts/mi300x_4b_adapter.sh start     # starts in the background and shows the log (Ctrl-C only stops viewing)
#   bash scripts/mi300x_4b_adapter.sh log       # view the log again later (after reconnecting)
#   bash scripts/mi300x_4b_adapter.sh status    # current stage, last training line, GPU use
#   bash scripts/mi300x_4b_adapter.sh stop      # stop everything (no self-delete)
#
# Stages (each runs once; re-running `start` on the same droplet skips finished ones):
#   setup     Python 3.12 venv (uv) with the ROCm build of PyTorch + our package
#   download  speech features + manifests (incl. the 4B replies made on Kaggle) + the Kaggle adapter
#   replies   the 4B's own replies, only for sets that do not have them yet (normally skipped)
#   phase_a   short real speech, 50% exact transcription, higher learning rate: teaches the new
#             connection to the 4B the exact words (fast: short sequences)
#   phase_b   the full mix incl. the long hotel prompts with tools
#   test      speech vs text tool accuracy (hotel) and transcription (accented speech), as on Kaggle
# Progress goes to Hugging Face every 15 minutes; on a NEW droplet, `start` resumes each phase from the
# last uploaded checkpoint. Result: checkpoints/speech_llm_4b_v2 (configs/qwen4b.yaml uses it).
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
MAX_HOURS=${MAX_HOURS:-10}               # with DO_TOKEN: the droplet is deleted after this, whatever happens
PHASE_A_MIN=${PHASE_A_MIN:-120}          # training minutes per phase (each phase ends itself on time)
PHASE_B_MIN=${PHASE_B_MIN:-240}
INIT=${INIT:-speech_llm_4b}              # the Kaggle adapter (falls back to the 0.6B adapter speech_llm_pk4)
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}
REPLIES=${REPLIES:-"cv_train_pk:14000 svarah_train_pk:5000 voxpop_acc_pk:5000 peoples_train_pk:6000"}

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s4b || echo "$HOME/s2s4b")}
VENV=$HOME/s2s-venv
LOG=$WORK/run.log
CFG="--config configs/qwen4b.yaml"
M=data/manifests
mkdir -p "$WORK"/{data,checkpoints,logs,done,tmp}

# --------------------------------------------------------------- commands
case "${1:-}" in
    start)
        : "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$WORK/run.pid")): bash $0 log"; exit 0
        fi
        # setsid + nohup: own session, no terminal; SSH disconnects and closing the window do not reach it
        HF_TOKEN="$HF_TOKEN" DO_TOKEN="${DO_TOKEN:-}" setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$WORK/run.pid"   # replaced by the run's own pid as soon as it starts
        echo "started in the background (pid $!). Log: $LOG"
        echo "Ctrl-C below only stops VIEWING the log; training keeps running. Later: bash $0 log"
        sleep 2; exec tail -n 50 -f "$LOG" ;;
    log)
        exec tail -n 100 -f "$LOG" ;;
    status)
        if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then echo "RUNNING"; else echo "NOT RUNNING"; fi
        echo "finished stages: $(ls "$WORK/done" 2>/dev/null | tr '\n' ' ')"
        grep -E "^\[|step [0-9]+ loss|\[eval" "$LOG" 2>/dev/null | tail -n 6 || true
        (rocm-smi --showuse --showmemuse 2>/dev/null || nvidia-smi 2>/dev/null) | grep -E "GPU|%" | head -6 || true
        exit 0 ;;
    stop)
        if [ -f "$WORK/run.pid" ]; then
            touch "$WORK/stopped"   # tells the exit handler not to delete the droplet
            pid=$(cat "$WORK/run.pid"); pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            kill -- -"${pgid:-$pid}" 2>/dev/null || kill "$pid" 2>/dev/null || true
            echo "stopped"
        fi
        exit 0 ;;
    run) ;;
    *) sed -n 2,23p "$0"; exit 1 ;;
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

hf() {  # hf push <folder> <path> [dataset] | hf get-ckpt <name>... | hf data
    "$VENV/bin/python" - "$@" <<'EOF'
import os, sys, subprocess
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api, cmd, args = HfApi(), sys.argv[1], sys.argv[2:]
ckpt, data, work = os.environ["CKPT_REPO"], os.environ["DATA_REPO"], os.environ["WORK"]
if cmd == "push":
    kind = args[2] if len(args) > 2 else "model"
    try:
        api.upload_folder(repo_id=ckpt if kind == "model" else data, repo_type=kind, folder_path=args[0],
                          path_in_repo=args[1], commit_message=f"mi300x 4b adapter: {args[1]}")
        print(f"[hf] {args[0]} -> {args[1]}", flush=True)
    except Exception as e:
        print(f"[hf] upload failed: {e}", flush=True); sys.exit(1)
elif cmd == "get-ckpt":          # download every listed checkpoint folder that exists; print the found names
    files = api.list_repo_files(ckpt)
    for name in args:
        if any(f.startswith(f"checkpoints/{name}/") for f in files):
            snapshot_download(ckpt, local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt", f"checkpoints/{name}/*.json",
                                                                   f"checkpoints/{name}/*.yaml", f"checkpoints/{name}/DONE"])
            print(name, flush=True)
elif cmd == "data":
    snapshot_download(data, repo_type="dataset", local_dir="data", allow_patterns=["manifests/*"])
    tars = sorted(f for f in api.list_repo_files(data, repo_type="dataset")
                  if f.endswith(".tar") and (f.startswith("pk-") or f.startswith("more-")))
    print(f"{len(tars)} feature archives", flush=True)
    for name in tars:
        t = hf_hub_download(data, name, repo_type="dataset", local_dir=work + "/tmp")
        subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
        print(f"  {name} extracted", flush=True)
EOF
}

# kill switch: with DO_TOKEN the droplet is deleted when the run ends (done or failed) and after MAX_HOURS
DROPLET_ID=$(curl -s --max-time 3 http://169.254.169.254/metadata/v1/id || true)
destroy() {
    if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ] && [ ! -f "$WORK/stopped" ]; then
        say "deleting droplet $DROPLET_ID in 120 s (Ctrl-C does nothing here; run 'stop' to cancel)"; sleep 120
        [ -f "$WORK/stopped" ] || curl -s -X DELETE -H "Authorization: Bearer $DO_TOKEN" \
            "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
    fi
}
if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ]; then
    say "kill switch on: droplet $DROPLET_ID is deleted at the end or after $MAX_HOURS h"
    ( sleep $(( MAX_HOURS * 3600 )); say "MAX_HOURS reached"; hf push "$WORK/logs" logs/mi300x_4b || true; destroy ) &
    TIMER_PID=$!
else
    say "no DO_TOKEN: the droplet is NOT deleted automatically - destroy it yourself when the run is done"
fi

cleanup() {
    local code=$?
    kill "${SYNC_PID:-0}" "${TIMER_PID:-0}" 2>/dev/null || true
    [ -x "$VENV/bin/python" ] && { hf push "$WORK/logs" logs/mi300x_4b || true; }
    if [ $code -eq 0 ]; then say "===== ALL DONE"; else say "===== FAILED (exit $code): see the stage log in $WORK/logs"; fi
    destroy
}
trap cleanup EXIT

stage() {  # stage <name> <command...>: once per droplet, logged to $WORK/logs/<name>.log
    local name=$1; shift
    if [ -f "$WORK/done/$name" ]; then say "skip $name (done)"; return; fi
    say ">>> $name"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$WORK/logs/$name.log"
    touch "$WORK/done/$name"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}

# ------------------------------------------------------------------ setup
setup() {
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
    uv pip install -p "$VENV/bin/python" -q "transformers>=4.51" "peft>=0.11" "accelerate>=0.30" numpy scipy \
        soundfile pyyaml jiwer tqdm huggingface_hub
    "$VENV/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU (check TORCH_INDEX)"
n = torch.cuda.device_count()
print(f"torch {torch.__version__} hip {torch.version.hip} | {n} x {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item(); print("GPU OK")
EOF
}
stage setup setup
export PATH="$VENV/bin:$PATH"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$NGPU" -gt 1 ]; then TRAIN="torchrun --standalone --nproc_per_node=$NGPU -m"; else TRAIN="python -m"; fi

# progress to Hugging Face every 15 minutes (a new droplet resumes from there)
( while sleep 900; do
    for d in speech_llm_4b_v2a speech_llm_4b_v2; do
        [ -f "checkpoints/$d/adapter.pt" ] && { hf push "checkpoints/$d" "checkpoints/$d" || true; }
    done
    hf push "$WORK/logs" logs/mi300x_4b || true
  done ) > "$WORK/logs/sync.log" 2>&1 &
SYNC_PID=$!

# --------------------------------------------------------------- download
download() {
    hf data
    found=$(hf get-ckpt speech_llm_4b_v2a speech_llm_4b_v2 "$INIT" speech_llm_pk4)
    echo "checkpoints found: $found"
    du -sh data/features; df -h "$WORK" | tail -1
}
stage download download
START=$INIT; [ -f "checkpoints/$INIT/adapter.pt" ] || START=speech_llm_pk4
[ -f "checkpoints/$START/adapter.pt" ] || { say "no starting adapter ($INIT / speech_llm_pk4) on Hugging Face"; exit 1; }
say "starting adapter: $START"

replies() {   # normally all present from the Kaggle run (manifests/*_q4f.jsonl)
    for pair in $REPLIES; do
        name=${pair%%:*}; n=${pair##*:}
        [ -f "$M/${name}_q4f.jsonl" ] && { echo "replies $name: present"; continue; }
        [ -f "$M/$name.jsonl" ] || { echo "replies $name: no manifest, skipped"; continue; }
        python -m s2s.prep.distill $CFG --gpus "$NGPU" --batch-size 128 --max-new-tokens 80 --max-utts "$n" \
            --in "$M/$name.jsonl" --out "$M/${name}_q4.jsonl" | tail -2
        python -m s2s.prep.reply_texts strip --in "$M/${name}_q4.jsonl" --out "$M/${name}_q4f.jsonl"
        mkdir -p "$WORK/up/manifests" && cp "$M/${name}_q4.jsonl" "$M/${name}_q4f.jsonl" "$WORK/up/manifests/"
    done
    if [ -d "$WORK/up" ]; then hf push "$WORK/up" "" dataset && rm -rf "$WORK/up"; fi
}
stage replies replies

mix() { [ -f "$1" ] && printf '{path: %s, weight: %s}, ' "$1" "$2"; return 0; }
REAL="$(mix $M/cv_train_pk_q4f.jsonl 0.20)$(mix $M/svarah_train_pk_q4f.jsonl 0.13)\
$(mix $M/voxpop_acc_pk_q4f.jsonl 0.09)$(mix $M/peoples_train_pk_q4f.jsonl 0.08)\
$(mix $M/cv_india_train_pk.jsonl 0.06)$(mix $M/ami_ihm_train_pk.jsonl 0.05)$(mix $M/ami_sdm_train_pk.jsonl 0.04)\
$(mix $M/peoples_dirty_train_pk.jsonl 0.05)$(mix $M/mls_train_pk.jsonl 0.04)"
MIX_A="[${REAL%, }]"
MIX_B="[$(mix $M/hotel4_train_pk.jsonl 0.23)$(mix $M/hotel3_train_pk.jsonl 0.03)${REAL%, }]"
VALID=$M/hotel4_eval_noisy_pk.jsonl; [ -f "$VALID" ] || VALID=$M/hotel4_eval_clean_pk.jsonl
VALID_A=$M/svarah_eval_pk.jsonl; [ -f "$VALID_A" ] || VALID_A=$VALID

train_phase() {  # train_phase <out name> <from> <steps> <minutes> <lr> <warmup> <batch> <accum> <transcribe> <mix> <valid>
    local out=$1 from=$2 steps=$3 minutes=$4 lr=$5 warmup=$6 batch=$7 accum=$8 tp=$9 mixv=${10} valid=${11}
    if [ -f "checkpoints/$out/DONE" ]; then echo "$out finished earlier (Hugging Face): skipped"; return; fi
    if [ -f "checkpoints/$out/adapter.pt" ]; then   # resume an interrupted phase from its last checkpoint
        local done_steps
        done_steps=$(python -c "import json; print(json.load(open('checkpoints/$out/state.json'))['step'])" 2>/dev/null || echo 0)
        echo "resuming $out after step $done_steps"
        from=$out; steps=$(( steps > done_steps + 100 ? steps - done_steps : 100 )); warmup=20
    fi
    echo "mix: $mixv"
    # shellcheck disable=SC2086
    $TRAIN s2s.train.speech_llm $CFG --set \
        train_speech_llm.train_lora=false train_speech_llm.init_from="checkpoints/$from" \
        train_speech_llm.output_dir="checkpoints/$out" train_speech_llm.max_steps="$steps" \
        train_speech_llm.max_minutes="$minutes" train_speech_llm.adapter_lr="$lr" \
        train_speech_llm.warmup_steps="$warmup" train_speech_llm.batch_size="$batch" \
        train_speech_llm.grad_accum="$accum" train_speech_llm.transcribe_prob="$tp" \
        train_speech_llm.num_workers=12 train_speech_llm.log_every=20 train_speech_llm.eval_every=500 \
        train_speech_llm.eval_batches=20 train_speech_llm.save_every=250 \
        "train_speech_llm.train_manifests=$mixv" train_speech_llm.valid_manifest="$valid"
    touch "checkpoints/$out/DONE"
    hf push "checkpoints/$out" "checkpoints/$out"
}

# phase A: short real speech, half exact transcription, higher learning rate (batch 64)
stage phase_a train_phase speech_llm_4b_v2a "$START" 6000 "$PHASE_A_MIN" 5e-4 200 32 2 0.5 "$MIX_A" "$VALID_A"
# phase B: everything incl. long hotel prompts with tools (batch 64)
stage phase_b train_phase speech_llm_4b_v2 speech_llm_4b_v2a 8000 "$PHASE_B_MIN" 3e-4 150 16 4 0.35 "$MIX_B" "$VALID"

test_adapter() {  # same tests as the Kaggle run (Kaggle adapter: tool accuracy 18%, transcription WER 142%)
    for m in hotel4_eval_clean_pk svarah_eval_pk; do
        [ -f "$M/$m.jsonl" ] || continue
        echo "===== $m"
        python -m s2s.eval.speech_llm $CFG --speech-llm-dir checkpoints/speech_llm_4b_v2 \
            --manifest "$M/$m.jsonl" --max 300 --out "$WORK/logs/test_$m.json" | tail -25
    done
}
stage test test_adapter
say "result: checkpoints/speech_llm_4b_v2 on Hugging Face ($CKPT_REPO); test results in $WORK/logs/test.log"
