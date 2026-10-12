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
#   BRAIN=1 (default): the 4B learns the hotel behaviour first, as the 0.6B did (thinker_merged_v3):
#   brain     text-only hotel LoRA on the 4B (hotel_v3 + tool-use data), at most BRAIN_MIN min ~BRAIN_MIN
#             (training data: hotel rules from templates + the 4B's OWN correct hotel answers + replay of its
#             own general answers, so it keeps its wording and personality)
#   merge     merged -> checkpoints/thinker_4b_hotel; checks, base vs tuned: typed hotel requests, variety of
#             wording, general requests, the voice (talker_4b) - the tuned 4B is used only if all pass ~35 min
#   answers   every training answer again, from the tuned 4B (<set>_q4h.jsonl, to HF)        ~40 min
#   polish    download the v3 data, train TRAIN_MIN minutes (batch 32 = the Kaggle 4 x 8),     ~TRAIN_MIN + 20 min
#             checkpoint to HF every 20 min (a new droplet continues from it)
#   test      the Gate 1 speech-vs-text test on the same clips as the Nemotron baseline       ~40 min
# Result: checkpoints/speech_llm_v3 and logs/v3_polish (summaries) on Hugging Face.
# Defaults: ~6.5 h with BRAIN=1, ~3.5 h with BRAIN=0 (polish only, base 4B).
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
BRAIN=${BRAIN:-1}                        # 1: hotel-tune the 4B before polish (see above)
MAX_HOURS=${MAX_HOURS:-$( [ "$BRAIN" = 1 ] && echo 8 || echo 5 )}  # with DO_TOKEN: the droplet is deleted after this
TRAIN_MIN=${TRAIN_MIN:-150}              # polish training minutes
TEST_MIN=${TEST_MIN:-45}
BRAIN_MIN=${BRAIN_MIN:-60}               # hotel LoRA: training stops after this (the last save is used)
BRAIN_STEPS=${BRAIN_STEPS:-1200}         # batch 16 x 2 = 32 conversations per step
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
    *) sed -n 2,26p "$0"; exit 1 ;;
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

POLISH_CFG="--config configs/qwen4b_v3.yaml"
ANS=q4
if [ "$BRAIN" = 1 ]; then
    CFG="--config configs/qwen4b_v3.yaml"
    BASE=Qwen/Qwen3-4B-Instruct-2507
    M=data/manifests
    mkdir -p "$WORK"/{data,checkpoints,done} "$WORK/out_polish/logs"
    ln -sfn "$WORK/data" data
    ln -sfn "$WORK/checkpoints" checkpoints
    mkdir -p $M
    LOGS="$WORK/out_polish/logs"
    stage() {  # stage <name> <command...>: runs once per droplet (marker in $WORK/done)
        local name=$1; shift
        if [ -f "$WORK/done/$name" ]; then say "skip $name (done)"; return; fi
        say ">>> $name"; local t0=$SECONDS
        "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$LOGS/$name.log"
        touch "$WORK/done/$name"; say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
    }
    hfpy() { python - "$@"; }

    fetch() {  # the v3 manifests, and the tuned 4B / its answers if an earlier droplet made them
        hfpy <<'PY'
import os, subprocess
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
ck, data = os.environ["CKPT_REPO"], os.environ["CKPT_REPO"].split("/")[0] + "/s2s-data-pk"
snapshot_download(data, repo_type="dataset", local_dir=".", allow_patterns=["v3/manifests/*"])
subprocess.run("cp v3/manifests/*.jsonl data/manifests/", shell=True, check=True)
if "checkpoints/thinker_4b_hotel/DONE" in api.list_repo_files(ck):
    snapshot_download(ck, local_dir=".", allow_patterns=["checkpoints/thinker_4b_hotel/*"])
    print("the tuned 4B is already on Hugging Face", flush=True)
PY
    }
    stage fetch fetch
    if [ -f checkpoints/thinker_4b_hotel/DONE ]; then
        for f in brain_data brain merge; do touch "$WORK/done/$f"; done
    fi

    brain_data() {  # template rows teach the rules; the 4B's own right answers + replay keep its voice
        python -m s2s.prep.thinker_sft_data $CFG --version 3 --hotel 8000 --hotel-v2 1000 --xlam 4000 --hermes 1500 ||
            python -m s2s.prep.thinker_sft_data $CFG --version 3 --hotel 8000 --hotel-v2 1000 --xlam 0 --hermes 1500
        python -m s2s.prep.brain_sft $CFG --hotel $M/hotel5_train --n-replay 6000 \
            --replay $M/cv_train_q4.jsonl $M/cv_india_q4.jsonl $M/peoples_train_q4.jsonl $M/ami_ihm_q4.jsonl \
                     $M/mls_train_q4.jsonl $M/spokenwoz_train_q4.jsonl $M/slurp_train_q4.jsonl
    }
    stage brain_data brain_data

    brain() {  # stops itself after BRAIN_MIN minutes; the LoRA saved every 100 steps is used
        local resume=""; [ -f checkpoints/thinker_text_4b/state.json ] && resume=--resume
        timeout "${BRAIN_MIN}m" python -m s2s.train.thinker_text $CFG --base $BASE --out checkpoints/thinker_text_4b \
            --steps "$BRAIN_STEPS" --batch 16 --accum 2 --lr 1e-4 --warmup 50 --eval-every 300 --save-every 100 \
            --eval-rows 100 --data-version 3 $resume || true
        [ -d checkpoints/thinker_text_4b/lora ] || { echo "the hotel LoRA saved nothing"; return 1; }
    }
    stage brain brain

    merge() {  # merge, then the checks; the tuned 4B is used only if all of them pass
        python -m s2s.prep.merge_lora $CFG --set thinker.model=$BASE --speech-llm-dir checkpoints/thinker_text_4b \
            --out checkpoints/thinker_4b_hotel
        python -m s2s.eval.brain_check $CFG --base $BASE --tuned checkpoints/thinker_4b_hotel \
            --hotel $M/hotel5_eval_nm.jsonl --general $M/brain_general_check.jsonl --out "$LOGS/brain_check.json"
        python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=["checkpoints/talker_4b/*.pt", "checkpoints/talker_4b/*.json"])
PY
        for t in base tuned; do  # the voice: talker_4b reads the 4B's hidden states, so check it with both
            local th=""; [ $t = tuned ] && th="--thinker checkpoints/thinker_4b_hotel"
            python -m s2s.eval.talker $CFG --talker-dir checkpoints/talker_4b $th --max 40 \
                --manifest $M/brain_talker_check.jsonl --out-dir "$LOGS/talker_$t" | tail -3
        done
        python - "$LOGS" <<'PY'
import json, os, sys
d = sys.argv[1]
check = json.load(open(os.path.join(d, "brain_check.json")))
wb = json.load(open(os.path.join(d, "talker_base", "result.json")))["talker_wer"]
wt = json.load(open(os.path.join(d, "talker_tuned", "result.json")))["talker_wer"]
voice_ok = wt <= wb + 0.05
print(f"voice (talker_4b, words wrong): with base 4B {wb * 100:.1f}%  with tuned 4B {wt * 100:.1f}%"
      f"  -> {'ok' if voice_ok else 'WORSE: the talker would need retraining'}", flush=True)
if check["use_tuned"] and voice_ok:
    open("checkpoints/thinker_4b_hotel/DONE", "w").close()
    print("DECISION: polish uses the hotel-tuned 4B", flush=True)
else:
    print("DECISION: polish uses the base 4B (the tuned one did not pass every check)", flush=True)
PY
        [ -f checkpoints/thinker_4b_hotel/DONE ] && python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=os.environ["CKPT_REPO"], folder_path="checkpoints/thinker_4b_hotel",
                      path_in_repo="checkpoints/thinker_4b_hotel", commit_message="v3: hotel-tuned 4B (merged)")
print("checkpoints/thinker_4b_hotel -> Hugging Face", flush=True)
PY
        return 0
    }
    stage merge merge

    if [ -f checkpoints/thinker_4b_hotel/DONE ]; then
        POLISH_CFG="--config configs/qwen4b_v3_hotel.yaml"
        ANS=q4h
        answers() {  # the same rows as the base 4B's answers, now from the tuned 4B
            local HCFG="--config configs/qwen4b_v3_hotel.yaml"
            for f in $M/*_q4.jsonl; do
                local s; s=$(basename "$f" _q4.jsonl)
                [ -f "$M/${s}_q4h.jsonl" ] && continue
                local n; n=$(wc -l < "$f")
                if [ "${s#hotel5}" != "$s" ]; then
                    python -m s2s.prep.distill $HCFG --gpus 1 --batch-size 48 --max-new-tokens 120 --max-utts 100000 \
                        --row-prompts --in "$M/${s}_nm.jsonl" --out "$M/${s}_q4h.jsonl" | tail -1
                else
                    python -m s2s.prep.distill $HCFG --gpus 1 --batch-size 128 --max-new-tokens 80 --max-utts "$n" \
                        --in "$M/${s}_nm.jsonl" --out "$M/${s}_q4h.jsonl" | tail -1
                fi
            done
            python - <<'PY'
import glob, os
from huggingface_hub import HfApi
api, data = HfApi(), os.environ["CKPT_REPO"].split("/")[0] + "/s2s-data-pk"
for f in sorted(glob.glob("data/manifests/*_q4h.jsonl")):
    api.upload_file(path_or_fileobj=f, path_in_repo="v3/manifests/" + os.path.basename(f), repo_id=data,
                    repo_type="dataset", commit_message="v3: answers from the hotel-tuned 4B")
print("q4h answers -> Hugging Face", flush=True)
PY
        }
        stage answers answers
    fi
fi

say ">>> polish: $TRAIN_MIN min training + test (scripts/kaggle_v3.sh PART=polish, $POLISH_CFG, answers $ANS)"
PART=polish WORK="$WORK" OUT="$WORK/out_polish" BUDGET_MIN=$(( 25 + TRAIN_MIN + TEST_MIN )) TEST_MIN="$TEST_MIN" \
    POLISH_BATCH=32 POLISH_ACCUM=1 TEST_PAR=3 CFG="$POLISH_CFG" ANS="$ANS" bash scripts/kaggle_v3.sh
