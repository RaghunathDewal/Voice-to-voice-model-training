#!/usr/bin/env bash
# =====================================================================================================
#  Qwen3-VL-4B room-inventory fine-tune on Kaggle (GPU T4 x2, Internet ON): data preparation -> training -> eval.
#  Trains on EVERY prepared image (HomeObjects train + all ~30k usable LVIS images), inside the session time.
#
#  Kaggle: Add-ons -> Secrets -> HF_TOKEN (write token). Then, in a notebook (GPU T4 x2, Internet on):
#     !git clone -q -b vision-qwen3vl https://github.com/RaghunathDewal/Voice-to-voice-model-training.git /tmp/repo
#     !bash /tmp/repo/vision/run_kaggle.sh all      (eval set: fetched from HF, or EVAL_DIR=/kaggle/input/...)
#  The cells, step by step: vision/README.md
#  Phases: setup | smoke | data | train | eval | all   (smoke = 3 real updates in ~10 min, before the long data phase)
#          bench: no training; the same eval set (built alone in ~6 min) for several models / ways of counting
#
#  Like the speech training (max_minutes):
#     - the run measures its own speed and fits the cosine learning rate into the time it has (time_budget.py)
#     - a checkpoint every SAVE_STEPS updates, each one uploaded to HF (HF_CKPT_REPO/last-checkpoint, with optimizer)
#     - if the session dies, run the same cell again: it downloads that checkpoint and continues
#     - TOTAL_HOURS=18 spreads one run over two sessions (the second commit continues and finishes)
#  Every value below can be changed from the command line, e.g.  SESSION_HOURS=11 bash run_kaggle.sh all
# =====================================================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-/kaggle/working}
DATA=${DATA:-/tmp/vl_data}              # images live outside /kaggle/working so the saved output stays small
RUN=${RUN:-v3}                          # run name -> output folder and HF repo qwen3vl-room-lora-<RUN> (README: Runs)
OUT=${OUT:-$WORK/out/qwen3vl-room-$RUN}
mkdir -p "$WORK" && cd "$WORK"
# session clock: started by the first phase run in this session (cells are separate processes)
[ -f "$WORK/.session_start" ] || date +%s > "$WORK/.session_start"
SESSION_START=$(cat "$WORK/.session_start")

MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
HF_USER=${HF_USER:-ThunderBlade7773}
export HF_CKPT_REPO=${HF_CKPT_REPO:-$HF_USER/qwen3vl-room-lora-$RUN}   # private repo of THIS run (v1 stays untouched)
# start from an earlier run's LoRA (its last-checkpoint on HF) instead of from zero; empty = from zero
INIT_FROM=${INIT_FROM-$HF_USER/qwen3vl-room-lora-v2}
# earlier runs to evaluate next to this one, on the SAME eval set (the only fair comparison)
COMPARE=${COMPARE-"$HF_USER/qwen3vl-room-lora $HF_USER/qwen3vl-room-lora-v2"}
INIT_DIR=/tmp/vl_init
INIT_ARGS=()
# image size: Qwen3-VL makes 1 token per 32x32 pixels. 768 tokens = ~0.79 MP (e.g. 1024x768). The SAME value is
# used in training and in eval. 512 = ~30% faster (more images seen per hour), less detail on small objects.
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-768}
# 4-bit base (QLoRA): QUANT=1. Off by default: on T4s (no native bf16) 4-bit ran ~165 s per update vs ~25-30 s
# expected; the 4B in bf16 fits a 16 GB T4 with LoRA (~12 GB). Turn it on only on a GPU that runs out of memory.
QUANT=${QUANT:-0}
# precision, as in the voice trainer (s2s/utils.py resolve_dtype): bf16 only on GPUs with native bf16 (A100, L4,
# RTX 30xx+); T4 / P100 / V100 emulate bf16, which is ~10x slower (the smoke test: 308 s per update) -> fp16 there
DTYPE=${DTYPE:-$(python -c "import torch; print('bfloat16' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else 'float16')" 2>/dev/null || echo float16)}
export DTYPE

# ----------------------------------------------------------------------------------------- time
SESSION_HOURS=${SESSION_HOURS:-10}      # what one Kaggle session may run (commit limit is 12 h; keep a margin)
EVAL_BASE=${EVAL_BASE:-0}               # 1 = also the base model (its frozen-room numbers are known: 73.1% exact)
EVAL_BATCH=${EVAL_BATCH:-4}             # questions per generate call in the eval
EVAL_JOB_H=${EVAL_JOB_H:-0.8}           # hours one model needs for the eval on one GPU (batched; 1.1 h unbatched)
# kept free at the end of the session for the eval: rounds of models on the GPUs x EVAL_JOB_H
N_EVAL_JOBS=$(( 1 + $(echo $COMPARE | wc -w) + EVAL_BASE ))
TOTAL_HOURS=${TOTAL_HOURS:-}            # empty = finish in this session; 18 = plan for two sessions

# ----------------------------------------------------------------------------------------- data settings
LVIS_IMAGES=${LVIS_IMAGES:-0}           # 0 = every usable LVIS image (~30k, ~5 GB download)
LVIS_GROUPS=${LVIS_GROUPS:-"bedroom bathroom living kitchen building"}   # areas of prep_lvis.py GROUPS to keep
TOTAL_ROWS=${TOTAL_ROWS:-0}             # 0 = one counting row per image (+ grounding rows below)
GROUND_SHARE=${GROUND_SHARE:-0.2}       # share of "locate every X" rows (boxes help counting)
EXTRA=${EXTRA:-}                        # optional: your own rows, "path.jsonl:1.0 other.jsonl:1.0" (same format)
EVAL_PER_SOURCE=${EVAL_PER_SOURCE:-200} # eval images per source (frozen 79 always all; HomeObjects val; LVIS held-out)
EVAL_DIR=${EVAL_DIR:-}                  # folder with eval_frozen.json + images/; empty = fetched from HF (below)
EVAL_ZIP_REPO=${EVAL_ZIP_REPO:-$HF_USER/s2s-checkpoints}             # where your fine-tune zip with eval_data/ is
EVAL_ZIP=${EVAL_ZIP:-Yesterday_Qwen3VL_FineTune_2026-10-21.zip}

# ----------------------------------------------------------------------------------------- training settings
EPOCHS=${EPOCHS:-2}                     # up to two passes; the time budget fits the cosine to what the session allows
LR=${LR:-$([ -n "$INIT_FROM" ] && echo 5e-5 || echo 1e-4)}   # continuing from a trained LoRA: half the step size
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
ACCUM=${ACCUM:-8}                       # 1 image per GPU x 8 accumulated x 2 GPUs = 16 rows per update
SAVE_STEPS=${SAVE_STEPS:-50}            # ~20-25 min of training between checkpoints
NGPU=${NGPU:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}

say() { echo "[$(date '+%H:%M:%S')] $*"; }
EVAL_RESERVE_H=${EVAL_RESERVE_H:-$(python -c "import math; print(round(math.ceil($N_EVAL_JOBS / $NGPU) * $EVAL_JOB_H + 0.2, 2))")}
export EVAL_BATCH

hf_token() {
    if [ -z "${HF_TOKEN:-}" ]; then
        HF_TOKEN=$(python -c "from kaggle_secrets import UserSecretsClient as U; print(U().get_secret('HF_TOKEN'))" \
                   2>/dev/null || true)
    fi
    export HF_TOKEN
    [ -n "$HF_TOKEN" ] || say "WARNING: no HF_TOKEN (Kaggle secret): checkpoints are NOT uploaded, a dead session loses them"
    [ -n "$HF_TOKEN" ] || export HF_CKPT_REPO=""
}

setup() {
    say ">>> setup"
    # ms-swift pinned to the version whose flags were checked (4.6 renamed --train_type to --tuner_type)
    pip install -q "ms-swift==4.6.0" "transformers>=4.57" "qwen-vl-utils>=0.0.14" accelerate peft bitsandbytes requests
    # Kaggle ships torchao 0.10; peft refuses any torchao < 0.16 when it adds LoRA layers (ImportError), and we do
    # not use torchao at all: remove it
    pip uninstall -q -y torchao 2>/dev/null || true
    python -c "
import swift, transformers, peft, qwen_vl_utils
from peft.import_utils import is_torchao_available
print('swift', swift.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__,
      '| torchao usable:', is_torchao_available())"
}

# download a run's LoRA (its last-checkpoint on HF, adapter files only) once:  fetch_lora <repo> <dir>
fetch_lora() {
    [ -f "$2/adapter_config.json" ] && return 0
    say ">>> LoRA: $1/last-checkpoint -> $2"
    python - "$1" "$2" <<'EOF'
import os, shutil, sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1:]
d = snapshot_download(repo, allow_patterns=["last-checkpoint/adapter_*", "last-checkpoint/*.json"], local_dir=out + "_dl")
os.makedirs(out, exist_ok=True)
for f in os.listdir(f"{d}/last-checkpoint"):            # the LoRA only: no optimizer, no trainer state
    if f.startswith("adapter_") or f in ("args.json", "additional_config.json"):
        shutil.copy(f"{d}/last-checkpoint/{f}", out)
print("LoRA files:", sorted(os.listdir(out)))
EOF
}

# eval name of a run's repo: qwen3vl-room-lora -> v1, qwen3vl-room-lora-v2 -> v2
run_tag() { local n; n=$(basename "$1"); case "$n" in *-lora) echo v1 ;; *-lora-*) echo "${n##*-lora-}" ;; *) echo "$n" ;; esac; }

fetch_init() {
    INIT_ARGS=()
    [ -n "$INIT_FROM" ] || return 0
    fetch_lora "$INIT_FROM" "$INIT_DIR"
    INIT_ARGS=(--adapters "$INIT_DIR")
}

# the one swift command, shared by smoke and train:  swift_sft <train.jsonl> <val.jsonl or ""> <out dir> [extra flags]
swift_sft() {
    local train_rows=$1 val_rows=$2 out=$3; shift 3
    local VAL=() QUANT_ARGS=()
    [ -n "$val_rows" ] && VAL=(--val_dataset "$val_rows")
    [ "$QUANT" = 1 ] && QUANT_ARGS=(--quant_method bnb --quant_bits 4 --bnb_4bit_compute_dtype "$DTYPE")
    NPROC_PER_NODE=$NGPU swift sft \
        --model "$MODEL" \
        --dataset "$train_rows" "${VAL[@]}" \
        --model_type qwen3_vl --tuner_type lora --use_hf true \
        --torch_dtype "$DTYPE" "${QUANT_ARGS[@]}" \
        --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --target_modules all-linear \
        --freeze_vit true \
        --num_train_epochs "$EPOCHS" \
        --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps "$ACCUM" \
        --learning_rate "$LR" --warmup_ratio 0.03 --weight_decay 0.01 \
        --max_length 2048 \
        --gradient_checkpointing true \
        --dataloader_num_workers 2 \
        --external_plugins "$HERE/time_budget.py" --callbacks time_budget \
        --add_version false --output_dir "$out" \
        "${INIT_ARGS[@]}" "$@"
}

smoke() {
    # ~10 min: 3 real updates on 64 HomeObjects rows with the exact training setup (also downloads the model once).
    # Catches install / model / LoRA / memory problems BEFORE the 30 min data phase. Nothing is uploaded.
    say ">>> smoke test: 3 updates, same settings as training ($DTYPE, image tokens $IMAGE_MAX_TOKEN_NUM, 4-bit=$QUANT)"
    mkdir -p "$DATA" /tmp/vl_smoke
    if [ ! -d "$DATA/homeobjects-3K/images/train" ]; then
        curl -sSL -o "$DATA/ho.zip" https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip
        mkdir -p "$DATA/homeobjects-3K" && unzip -q "$DATA/ho.zip" -d "$DATA/homeobjects-3K" && rm "$DATA/ho.zip"
    fi
    python "$HERE/prep_homeobjects.py" --root "$DATA/homeobjects-3K" --out /tmp/vl_smoke/home > /dev/null
    { head -48 /tmp/vl_smoke/home/count_sft.jsonl; head -16 /tmp/vl_smoke/home/ground_sft.jsonl; } > /tmp/vl_smoke/rows.jsonl
    rm -rf /tmp/vl_smoke/out
    hf_token
    fetch_init                                         # also tests loading the starting LoRA
    HF_CKPT_REPO="" TRAIN_HOURS=5 swift_sft /tmp/vl_smoke/rows.jsonl "" /tmp/vl_smoke/out \
        --max_steps 3 --save_steps 1000 --logging_steps 1 2>&1 | tee /tmp/vl_smoke/log.txt \
        | grep -E "Error|error|'loss'" || true
    python "$HERE/smoke_check.py" /tmp/vl_smoke/log.txt
    rm -rf /tmp/vl_smoke/out
}

data() {
    mkdir -p "$DATA"
    say ">>> HomeObjects-3K (train split only; the eval set is built from its val split)"
    if [ ! -d "$DATA/homeobjects-3K/images/train" ]; then
        curl -sSL -o "$DATA/ho.zip" https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip
        mkdir -p "$DATA/homeobjects-3K" && unzip -q "$DATA/ho.zip" -d "$DATA/homeobjects-3K" && rm "$DATA/ho.zip"
    fi
    python "$HERE/prep_homeobjects.py" --root "$DATA/homeobjects-3K" --out "$DATA/home"
    python "$HERE/prep_homeobjects.py" --root "$DATA/homeobjects-3K" --out "$DATA/home_val" --split val   # test only

    say ">>> LVIS: official train annotations (for VERIFIED zeros) + Voxel51 images"
    if [ ! -f "$DATA/lvis_v1_train.json" ]; then
        curl -sSL -o "$DATA/lvis.zip" https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
        unzip -q "$DATA/lvis.zip" -d "$DATA" && rm "$DATA/lvis.zip"
    fi
    # shellcheck disable=SC2086
    python "$HERE/prep_lvis.py" --out "$DATA/lvis" --groups $LVIS_GROUPS --lvis-json "$DATA/lvis_v1_train.json" \
        --max-images "$LVIS_IMAGES" --download
    rm -f "$DATA/lvis_v1_train.json"     # 1 GB, not needed any more

    say ">>> mix -> $DATA/sft  (same seed every time: a resumed session rebuilds the same rows)"
    # shellcheck disable=SC2086
    python "$HERE/mix_data.py" --out "$DATA/sft" --total "$TOTAL_ROWS" --ground-share "$GROUND_SHARE" \
        --count "$DATA/home/count_sft.jsonl" "$DATA/lvis/count_sft.jsonl" $EXTRA \
        --ground "$DATA/home/ground_sft.jsonl" "$DATA/lvis/ground_sft.jsonl"
    mkdir -p "$OUT" && cp "$DATA/lvis/stats.json" "$OUT/lvis_stats.json"

    say ">>> eval set: your 79 frozen rooms + HomeObjects val + held-out LVIS (none of them trained on)"
    hf_token
    fetch_eval || say "frozen set not available: eval set without it"
    python "$HERE/build_eval_set.py" --out "$DATA/eval/eval_set.jsonl" --test "$DATA/sft/test.jsonl" \
        --home-val "$DATA/home_val/count_sft.jsonl" --max-per-source "$EVAL_PER_SOURCE" \
        ${EVAL_DIR:+--frozen "$EVAL_DIR"}
    cp "$DATA/eval/eval_set.jsonl" "$OUT/eval_set.jsonl"
    say ">>> data check (stops here if the mix repeats a known mistake)"
    python "$HERE/check_data.py" "$DATA/sft/train.jsonl" --eval "$DATA/eval/eval_set.jsonl"
}

latest_ckpt() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true; }

train() {
    [ -f "$DATA/sft/train.jsonl" ] || { echo "no $DATA/sft/train.jsonl: run the data phase first"; exit 1; }
    python "$HERE/check_data.py" "$DATA/sft/train.jsonl" > /dev/null || { echo "data check failed: see the data phase"; exit 1; }
    hf_token
    # continue from the last checkpoint: local one, else the one on Hugging Face (a previous session)
    if [ -z "$(latest_ckpt)" ] && [ -n "$HF_CKPT_REPO" ]; then
        python - "$HF_CKPT_REPO" "$OUT" <<'EOF' || true
import json, os, sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1:]
try:
    d = snapshot_download(repo, allow_patterns=["last-checkpoint/*"], local_dir=f"{out}/_hf")
except Exception as e:
    print("no checkpoint on HF yet:", type(e).__name__); sys.exit(0)
step = json.load(open(f"{d}/last-checkpoint/trainer_state.json"))["global_step"]
os.rename(f"{d}/last-checkpoint", f"{out}/checkpoint-{step}")
print(f"downloaded checkpoint-{step} from {repo}")
EOF
    fi
    RESUME=()
    last=$(latest_ckpt)
    fetch_init
    # resuming THIS run: its checkpoint already holds the LoRA, so the starting LoRA is not loaded again
    [ -n "$last" ] && { say "resuming from $last"; RESUME=(--resume_from_checkpoint "$last"); INIT_ARGS=(); }

    used_h=$(python -c "import time; print((time.time() - $SESSION_START) / 3600)")
    export TRAIN_HOURS=${TRAIN_HOURS:-$(python -c "print(round($SESSION_HOURS - $used_h - $EVAL_RESERVE_H - 0.15, 2))")}
    export TOTAL_HOURS=${TOTAL_HOURS:-$TRAIN_HOURS}
    rows=$(wc -l < "$DATA/sft/train.jsonl")
    say ">>> train: $rows rows, $NGPU GPU(s), image tokens $IMAGE_MAX_TOKEN_NUM, $TRAIN_HOURS h in this session" \
        "(run planned for $TOTAL_HOURS h, $EVAL_RESERVE_H h kept for $N_EVAL_JOBS eval jobs), run $RUN from ${INIT_FROM:-zero}, lr $LR, $EPOCHS epochs max, $DTYPE, 4-bit=$QUANT, checkpoints every $SAVE_STEPS updates -> ${HF_CKPT_REPO:-local only}"
    swift_sft "$DATA/sft/train.jsonl" "$DATA/sft/val.jsonl" "$OUT" \
        --eval_steps 200 --save_steps "$SAVE_STEPS" --save_total_limit 2 --logging_steps 10 \
        "${RESUME[@]}"
    say "checkpoints: $(ls -d "$OUT"/checkpoint-* | tr '\n' ' ')"
}

fetch_eval() {
    # no EVAL_DIR given: download the zip from Hugging Face and use the eval_data/ folder inside it
    [ -n "$EVAL_DIR" ] && return
    mkdir -p /tmp/vl_eval
    if [ -z "$(find /tmp/vl_eval -name eval_frozen.json | head -1)" ]; then
        say ">>> eval set: $EVAL_ZIP from $EVAL_ZIP_REPO"
        python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$EVAL_ZIP_REPO', '$EVAL_ZIP', local_dir='/tmp/vl_eval'))"
        unzip -q -o "/tmp/vl_eval/$EVAL_ZIP" -d /tmp/vl_eval
    fi
    f=$(find /tmp/vl_eval -name eval_frozen.json | head -1)
    [ -n "$f" ] || { EVAL_DIR=""; return 1; }
    EVAL_DIR=$(dirname "$f")
    say "EVAL_DIR=$EVAL_DIR ($(ls "$EVAL_DIR/images" | wc -l) images)"
}

evaluate() {
    ES=$DATA/eval/eval_set.jsonl
    [ -f "$ES" ] || { echo "no $ES: run the data phase first (it builds the eval set)"; exit 1; }
    ck=${ADAPTER:-$(latest_ckpt)}
    [ -n "$ck" ] || { echo "no checkpoint in $OUT"; exit 1; }
    # jobs: this run, the runs in COMPARE, optionally the base; spread over the GPUs, each GPU one after the other
    JOBS=("$RUN|$ck")
    for r in $COMPARE; do
        t=$(run_tag "$r")
        if [ "$r" = "$INIT_FROM" ]; then fetch_lora "$r" "$INIT_DIR"; d=$INIT_DIR; else d=/tmp/vl_cmp/$t; fetch_lora "$r" "$d"; fi
        JOBS+=("$t|$d")
    done
    [ "$EVAL_BASE" = 1 ] && JOBS+=("base|")
    Q4=$([ "$QUANT" = 1 ] && echo --quant4 || true)
    say ">>> eval on $(wc -l < "$ES") images: ${JOBS[*]%%|*} on $NGPU GPU(s)"
    PIDS=(); PREDS=()
    for g in $(seq 0 $((NGPU - 1))); do
        (
            for i in "${!JOBS[@]}"; do
                [ $((i % NGPU)) = "$g" ] || continue
                tag=${JOBS[$i]%%|*}; ad=${JOBS[$i]#*|}
                CUDA_VISIBLE_DEVICES=$g python "$HERE/eval_counts.py" run --eval-set "$ES" --model "$MODEL" --tag "$tag" \
                    ${ad:+--adapter "$ad"} $Q4 --batch "$EVAL_BATCH" --out "$WORK/pred_$tag.json" > "$WORK/eval_$tag.log" 2>&1 \
                    && echo "  [$tag] done" || { echo "  [$tag] FAILED:"; tail -15 "$WORK/eval_$tag.log"; }
            done
        ) &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    for j in "${JOBS[@]}"; do t=${j%%|*}; [ -f "$WORK/pred_$t.json" ] && PREDS+=("$WORK/pred_$t.json"); done
    [ "${#PREDS[@]}" -gt 0 ] || exit 1
    python "$HERE/eval_counts.py" report "${PREDS[@]}" --out "$WORK/eval_results.json"
    [ -z "${HF_CKPT_REPO:-}" ] || python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$WORK/eval_results.json', path_in_repo='eval_results.json', repo_id='$HF_CKPT_REPO')
print('eval_results.json uploaded to $HF_CKPT_REPO')" || true
}

# ===================================================================================================== bench
# BENCH: one job per entry "tag|base model|LoRA repo on HF (or empty)|boxes", ';'-separated. boxes = main counts
# pillow/towel/lamp/chair also by "locate every X" (report rows <tag>+boxes and <tag>+hybrid), none = counting only.
BENCH=${BENCH:-"v3|Qwen/Qwen3-VL-4B-Instruct|$HF_USER/qwen3vl-room-lora-v3|main;qwen3.5-4b|Qwen/Qwen3.5-4B||main;qwen3vl-4b-base|Qwen/Qwen3-VL-4B-Instruct||none"}
BENCH_REPO=${BENCH_REPO:-$HF_USER/qwen3vl-room-lora-v3}      # where bench_results.json is uploaded
BENCH_FLA=${BENCH_FLA:-0}               # 1 = install flash-linear-attention (fast Qwen3.5 kernels; T4 support unknown)

data_eval() {
    # only what the eval set needs: HomeObjects val, the frozen rooms, and the held-out LVIS images (not all 30k)
    [ -f "$DATA/eval/eval_set.jsonl" ] && { say "eval set already built: $DATA/eval/eval_set.jsonl"; return 0; }
    mkdir -p "$DATA"
    if [ ! -d "$DATA/homeobjects-3K/images/val" ]; then
        curl -sSL -o "$DATA/ho.zip" https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip
        mkdir -p "$DATA/homeobjects-3K" && unzip -q "$DATA/ho.zip" -d "$DATA/homeobjects-3K" && rm "$DATA/ho.zip"
    fi
    python "$HERE/prep_homeobjects.py" --root "$DATA/homeobjects-3K" --out "$DATA/home_val" --split val
    if [ ! -f "$DATA/lvis_v1_train.json" ]; then
        curl -sSL -o "$DATA/lvis.zip" https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
        unzip -q "$DATA/lvis.zip" -d "$DATA" && rm "$DATA/lvis.zip"
    fi
    # shellcheck disable=SC2086
    python "$HERE/prep_lvis.py" --out "$DATA/lvis" --groups $LVIS_GROUPS --lvis-json "$DATA/lvis_v1_train.json" \
        --max-images "$LVIS_IMAGES" > /dev/null      # no --download: rows point at the Hub, held-out ones fetched below
    python "$HERE/mix_data.py" --out "$DATA/sft" --count "$DATA/lvis/count_sft.jsonl" > /dev/null
    hf_token
    fetch_eval || say "frozen set not available: eval set without it"
    python "$HERE/build_eval_set.py" --out "$DATA/eval/eval_set.jsonl" --test "$DATA/sft/test.jsonl" \
        --home-val "$DATA/home_val/count_sft.jsonl" --max-per-source "$EVAL_PER_SOURCE" ${EVAL_DIR:+--frozen "$EVAL_DIR"}
    python - "$DATA/eval/eval_set.jsonl" "$DATA/lvis/images" <<'EOF'
import json, os, sys, requests
from concurrent.futures import ThreadPoolExecutor
es, d = sys.argv[1:]
os.makedirs(d, exist_ok=True)
rows = [json.loads(l) for l in open(es)]
h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
def get(url):
    p = os.path.join(d, os.path.basename(url))
    if not os.path.exists(p):
        r = requests.get(url, headers=h, timeout=60); r.raise_for_status(); open(p, "wb").write(r.content)
    return p
urls = sorted({r["images"][0] for r in rows if r["images"][0].startswith("http")})
local = dict(zip(urls, ThreadPoolExecutor(16).map(get, urls)))
with open(es, "w") as f:
    for r in rows:
        r["images"] = [local.get(r["images"][0], r["images"][0])]
        f.write(json.dumps(r) + "\n")
print(f"eval images downloaded: {len(urls)} LVIS")
EOF
}

bench() {
    data_eval
    hf_token
    [ "$BENCH_FLA" = 1 ] && pip install -q flash-linear-attention || true
    ES=$DATA/eval/eval_set.jsonl
    Q4=$([ "$QUANT" = 1 ] && echo --quant4 || true)
    IFS=';' read -r -a JOBS <<< "$BENCH"
    say ">>> bench on $(wc -l < "$ES") images: $(for j in "${JOBS[@]}"; do printf '%s ' "${j%%|*}"; done)on $NGPU GPU(s)"
    for j in "${JOBS[@]}"; do                          # LoRAs first, so the GPU jobs start together
        IFS='|' read -r tag model repo boxes <<< "$j"
        [ -n "$repo" ] && fetch_lora "$repo" "/tmp/vl_cmp/$tag"
    done
    PIDS=()
    for g in $(seq 0 $((NGPU - 1))); do
        (
            for i in "${!JOBS[@]}"; do
                [ $((i % NGPU)) = "$g" ] || continue
                IFS='|' read -r tag model repo boxes <<< "${JOBS[$i]}"
                CUDA_VISIBLE_DEVICES=$g python "$HERE/eval_counts.py" run --eval-set "$ES" --model "$model" --tag "$tag" \
                    ${repo:+--adapter "/tmp/vl_cmp/$tag"} --boxes "${boxes:-none}" --batch "$EVAL_BATCH" $Q4 \
                    --out "$WORK/bench_$tag.json" > "$WORK/bench_$tag.log" 2>&1 \
                    && echo "  [$tag] done" || { echo "  [$tag] FAILED:"; tail -25 "$WORK/bench_$tag.log"; }
            done
        ) &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    PREDS=()
    for j in "${JOBS[@]}"; do t=${j%%|*}; [ -f "$WORK/bench_$t.json" ] && PREDS+=("$WORK/bench_$t.json"); done
    [ "${#PREDS[@]}" -gt 0 ] || exit 1
    python "$HERE/eval_counts.py" report "${PREDS[@]}" --out "$WORK/bench_results.json"
    [ -z "${HF_TOKEN:-}" ] || python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$WORK/bench_results.json', path_in_repo='bench_results.json', repo_id='$BENCH_REPO')
print('bench_results.json uploaded to $BENCH_REPO')" || true
}

case "${1:-}" in
    setup) setup ;;
    smoke) smoke ;;
    data) data ;;
    train) train ;;
    eval) hf_token; evaluate ;;
    bench) bench ;;
    all) setup; smoke; data; train; evaluate ;;
    *) sed -n 2,19p "$0"; exit 1 ;;
esac
say "done ($(( ($(date +%s) - SESSION_START) / 60 )) min since the session started)"
