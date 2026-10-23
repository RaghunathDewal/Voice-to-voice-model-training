#!/usr/bin/env bash
# =====================================================================================================
#  Qwen3-VL-4B room-inventory fine-tune on Kaggle (GPU T4 x2, Internet ON): data preparation -> training -> eval.
#  Trains on EVERY prepared image (HomeObjects train + all ~30k usable LVIS images), inside the session time.
#
#  Kaggle: Add-ons -> Secrets -> HF_TOKEN (write token). Then, in a notebook (GPU T4 x2, Internet on):
#     !git clone -q -b vision-qwen3vl https://github.com/RaghunathDewal/Voice-to-voice-model-training.git /tmp/repo
#     !bash /tmp/repo/vision/run_kaggle.sh all      (eval set: fetched from HF, or EVAL_DIR=/kaggle/input/...)
#  The cells, step by step: vision/README.md
#  Phases: setup | data | train | eval | all
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
OUT=${OUT:-$WORK/out/qwen3vl-room}
mkdir -p "$WORK" && cd "$WORK"
# session clock: started by the first phase run in this session (cells are separate processes)
[ -f "$WORK/.session_start" ] || date +%s > "$WORK/.session_start"
SESSION_START=$(cat "$WORK/.session_start")

MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
HF_USER=${HF_USER:-ThunderBlade7773}
export HF_CKPT_REPO=${HF_CKPT_REPO:-$HF_USER/qwen3vl-room-lora}   # private model repo, created on the first save
# image size: Qwen3-VL makes 1 token per 32x32 pixels. 768 tokens = ~0.79 MP (e.g. 1024x768). The SAME value is
# used in training and in eval. 512 = ~30% faster (more images seen per hour), less detail on small objects.
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-768}

# ----------------------------------------------------------------------------------------- time
SESSION_HOURS=${SESSION_HOURS:-10}      # what one Kaggle session may run (commit limit is 12 h; keep a margin)
EVAL_RESERVE_H=${EVAL_RESERVE_H:-0.8}   # kept free at the end for the eval phase (~580 images, base || finetuned)
TOTAL_HOURS=${TOTAL_HOURS:-}            # empty = finish in this session; 18 = plan for two sessions

# ----------------------------------------------------------------------------------------- data settings
LVIS_IMAGES=${LVIS_IMAGES:-0}           # 0 = every usable LVIS image (~30k, ~5 GB download)
LVIS_GROUPS=${LVIS_GROUPS:-"bedroom bathroom living kitchen building"}   # areas of prep_lvis.py GROUPS to keep
TOTAL_ROWS=${TOTAL_ROWS:-0}             # 0 = one counting row per image (+ grounding rows below)
GROUND_SHARE=${GROUND_SHARE:-0.2}       # share of "locate every X" rows (boxes help counting)
EXTRA=${EXTRA:-}                        # optional: your own rows, "path.jsonl:1.0 other.jsonl:1.0" (same format)
EVAL_PER_SOURCE=${EVAL_PER_SOURCE:-250} # eval images per source (frozen 79 always all; HomeObjects val; LVIS held-out)
EVAL_DIR=${EVAL_DIR:-}                  # folder with eval_frozen.json + images/; empty = fetched from HF (below)
EVAL_ZIP_REPO=${EVAL_ZIP_REPO:-$HF_USER/s2s-checkpoints}             # where your fine-tune zip with eval_data/ is
EVAL_ZIP=${EVAL_ZIP:-Yesterday_Qwen3VL_FineTune_2026-10-21.zip}

# ----------------------------------------------------------------------------------------- training settings
EPOCHS=${EPOCHS:-1}                     # one pass over all images; the time budget may end it earlier
LR=${LR:-1e-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
ACCUM=${ACCUM:-8}                       # 1 image per GPU x 8 accumulated x 2 GPUs = 16 rows per update
SAVE_STEPS=${SAVE_STEPS:-50}            # ~20-25 min of training between checkpoints
NGPU=${NGPU:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}

say() { echo "[$(date '+%H:%M:%S')] $*"; }

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
    pip install -q "ms-swift>=4.5" "transformers>=4.57" "qwen-vl-utils>=0.0.14" accelerate peft bitsandbytes requests
    python -c "import swift, transformers, qwen_vl_utils; print('swift', swift.__version__, 'transformers', transformers.__version__)"
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
}

latest_ckpt() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true; }

train() {
    [ -f "$DATA/sft/train.jsonl" ] || { echo "no $DATA/sft/train.jsonl: run the data phase first"; exit 1; }
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
    [ -n "$last" ] && { say "resuming from $last"; RESUME=(--resume_from_checkpoint "$last"); }

    used_h=$(python -c "import time; print((time.time() - $SESSION_START) / 3600)")
    export TRAIN_HOURS=${TRAIN_HOURS:-$(python -c "print(round($SESSION_HOURS - $used_h - $EVAL_RESERVE_H - 0.15, 2))")}
    export TOTAL_HOURS=${TOTAL_HOURS:-$TRAIN_HOURS}
    rows=$(wc -l < "$DATA/sft/train.jsonl")
    say ">>> train: $rows rows, $NGPU GPU(s), image tokens $IMAGE_MAX_TOKEN_NUM, $TRAIN_HOURS h in this session" \
        "(run planned for $TOTAL_HOURS h), checkpoints every $SAVE_STEPS updates -> ${HF_CKPT_REPO:-local only}"
    NPROC_PER_NODE=$NGPU swift sft \
        --model "$MODEL" \
        --dataset "$DATA/sft/train.jsonl" \
        --val_dataset "$DATA/sft/val.jsonl" \
        --train_type lora \
        --quant_method bnb --quant_bits 4 \
        --bnb_4bit_compute_dtype bfloat16 --torch_dtype bfloat16 \
        --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --target_modules all-linear \
        --freeze_vit true \
        --num_train_epochs "$EPOCHS" \
        --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps "$ACCUM" \
        --learning_rate "$LR" --warmup_ratio 0.03 --weight_decay 0.01 \
        --max_length 2048 \
        --gradient_checkpointing true \
        --eval_steps 200 --save_steps "$SAVE_STEPS" --save_total_limit 2 --logging_steps 10 \
        --dataloader_num_workers 2 \
        --external_plugins "$HERE/time_budget.py" --callbacks time_budget \
        --add_version false --output_dir "$OUT" \
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
    say ">>> eval on $(wc -l < "$ES") images: base (GPU 0) and $ck (GPU 1) at the same time"
    G1=$([ "$NGPU" -gt 1 ] && echo 1 || echo 0)
    CUDA_VISIBLE_DEVICES=0 python "$HERE/eval_counts.py" run --eval-set "$ES" --model "$MODEL" --tag base \
        --out "$WORK/pred_base.json" > "$WORK/eval_base.log" 2>&1 &
    P0=$!
    [ "$G1" = 0 ] && wait $P0                         # one GPU: one after the other
    CUDA_VISIBLE_DEVICES=$G1 python "$HERE/eval_counts.py" run --eval-set "$ES" --model "$MODEL" --tag finetuned \
        --adapter "$ck" --out "$WORK/pred_ft.json" 2>&1 | tee "$WORK/eval_ft.log" | grep -E "done|/[0-9]+$" || true
    wait $P0 || { tail -20 "$WORK/eval_base.log"; exit 1; }
    [ -f "$WORK/pred_ft.json" ] || { tail -20 "$WORK/eval_ft.log"; exit 1; }
    python "$HERE/eval_counts.py" report "$WORK/pred_base.json" "$WORK/pred_ft.json" --out "$WORK/eval_results.json"
    [ -z "${HF_CKPT_REPO:-}" ] || python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$WORK/eval_results.json', path_in_repo='eval_results.json', repo_id='$HF_CKPT_REPO')
print('eval_results.json uploaded')" || true
}

case "${1:-}" in
    setup) setup ;;
    data) data ;;
    train) train ;;
    eval) hf_token; evaluate ;;
    all) setup; data; train; evaluate ;;
    *) sed -n 2,19p "$0"; exit 1 ;;
esac
say "done ($(( ($(date +%s) - SESSION_START) / 60 )) min since the session started)"
