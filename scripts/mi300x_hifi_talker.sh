#!/usr/bin/env bash
# A clearer, more natural voice: retrain the talker on real studio speech, on the MI300X droplet after
# scripts/mi300x_train.sh (it reuses that run's Python, thinker_merged_v3, talker_v3 and hotel data).
#
#   export HF_TOKEN=hf_...
#   bash scripts/mi300x_hifi_talker.sh 2>&1 | tee -a ~/hifi.log
#
#   1. LibriTTS-R (CC BY 4.0): ~200 h of studio speech from ~900 voices. Pre-training on it teaches
#      the talker how English words are pronounced in general, not only the ~40k Kokoro sentences
#      it has seen so far. Our hotel sentences (Kokoro) stay in the mix (40%) for numbers, room
#      numbers, times and Wi-Fi passwords.
#   2. Hi-Fi TTS (CC BY 4.0), one speaker (default 92, female, ~28 h): fine-tune to that one voice.
#   3. Evaluation on the same held-out hotel replies for talker_v3 and the new talker: Whisper word
#      error rate (lower = clearer) and the audio files to listen to, in $WORK/logs/hifi_eval/.
#
# Same size (small.yaml, 47M) and same architecture, so streaming is unchanged: the talker starts
# speaking from the first text tokens and audio is sent every 160 ms. Every stage is resumable.
# Note: a Hi-Fi TTS voice is a real (volunteer) narrator; check you are fine using it in a product.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
SPEAKER=${SPEAKER:-92}                 # Hi-Fi TTS speakers: 92, 6670, 6671, 8051, 9136, 11614, 11697, 12787 (F/M mix), 6097, 9017
LIBRI_HOURS=${LIBRI_HOURS:-200}
PRE_STEPS=${PRE_STEPS:-40000}          # batch 32
FT_STEPS=${FT_STEPS:-12000}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s || echo "$HOME/s2s_work")}
CFG="--config configs/small.yaml"
M=data/manifests
TH=checkpoints/thinker_merged_v3
HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
: "${HF_TOKEN:?export HF_TOKEN first (or log in once: huggingface-cli login)}"
export HF_TOKEN CKPT_REPO PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30 TOKENIZERS_PARALLELISM=false
cd "$REPO_DIR"
mkdir -p "$WORK"/{data,checkpoints,logs,done}
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
[ -x "$WORK/bin/python" ] && export PATH="$WORK/bin:$PATH" PIP_BREAK_SYSTEM_PACKAGES=1
python -c "import torch, s2s; assert torch.cuda.is_available()" || {
    echo "run this inside the rocm container, after scripts/mi300x_train.sh set up Python"; exit 1; }
for need in $TH/config.json checkpoints/talker_v3/talker.pt $M/talker_v3_train.jsonl $M/talker_v3_valid.jsonl; do
    [ -e "$need" ] || { echo "missing $need (from scripts/mi300x_train.sh)"; exit 1; }
done
pip install -q pyarrow librosa
LOGS=$WORK/logs
say() { echo "[$(date +%H:%M:%S)] $*"; }
stage() {  # stage <name> <command...>: run once, log to $LOGS/<name>.log
    local name=$1; shift
    if [ -f "$WORK/done/$name" ]; then say "skip $name (done)"; return; fi
    say ">>> $name"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$LOGS/$name.log"
    touch "$WORK/done/$name"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}
hf_push() {  # hf_push <local folder> <path in repo>
    python - "$1" "$2" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=os.environ["CKPT_REPO"], folder_path=sys.argv[1], path_in_repo=sys.argv[2],
                      commit_message=f"mi300x: {sys.argv[2]}")
print(f"[hf] {sys.argv[1]} -> {sys.argv[2]}")
EOF
}
( while sleep 600; do for d in talker_hifi_pre talker_hifi; do
    [ -d "checkpoints/$d" ] && hf_push "checkpoints/$d" "checkpoints/$d"; done; done ) > "$LOGS/hifi_sync.log" 2>&1 &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

# ------------------------------------------------------------------ 1. data (24 kHz, original text)
libri() {
    python -m s2s.prep.hf_asr $CFG --preset libritts_r --split train.clean.360 --max-hours "$LIBRI_HOURS" \
        --sample-rate 24000 --max-seconds 25 --min-words 3 --name libritts_r_train | tail -3
    python -m s2s.prep.extract_mimi $CFG --mode codes --in $M/libritts_r_train_raw.jsonl \
        --out $M/libritts_r_train.jsonl | tail -1
    rm -rf data/hf_asr/libritts_r_train      # the codes are what we train on
}
stage hifi_libri libri

hifi() {
    python -m s2s.prep.hf_asr $CFG --preset hifitts --split train.clean --accent-match "^${SPEAKER}\$" \
        --sample-rate 24000 --max-seconds 25 --min-words 3 --name hifi${SPEAKER} | tail -3
    head -200 $M/hifi${SPEAKER}_raw.jsonl > $M/hifi${SPEAKER}_valid_raw.jsonl
    tail -n +201 $M/hifi${SPEAKER}_raw.jsonl > $M/hifi${SPEAKER}_train_raw.jsonl
    for n in hifi${SPEAKER}_train hifi${SPEAKER}_valid; do
        python -m s2s.prep.extract_mimi $CFG --mode codes --in $M/${n}_raw.jsonl --out $M/$n.jsonl | tail -1
    done
}
stage hifi_voice hifi

# ------------------------------------------------- 2. pre-train: many voices + hotel sentences
TRAIN_COMMON=(train_talker.thinker_dir=$TH train_talker.batch_size=32 train_talker.grad_accum=1
              train_talker.num_workers=12 train_talker.eval_every=2000 train_talker.save_every=2000)
stage hifi_pretrain python -m s2s.train.talker $CFG --set "${TRAIN_COMMON[@]}" \
    train_talker.init_from=checkpoints/talker_v3 train_talker.output_dir=checkpoints/talker_hifi_pre \
    train_talker.max_steps="$PRE_STEPS" train_talker.lr=0.0003 train_talker.warmup_steps=1000 \
    "train_talker.train_manifests=[{path: $M/libritts_r_train.jsonl, weight: 0.6}, {path: $M/talker_v3_train.jsonl, weight: 0.4}]" \
    train_talker.valid_manifest=$M/talker_v3_valid.jsonl
hf_push checkpoints/talker_hifi_pre checkpoints/talker_hifi_pre

# ------------------------------------------------- 3. fine-tune to the one voice
# lower learning rate: keep what pre-training learned about words, change the voice
stage hifi_finetune python -m s2s.train.talker $CFG --set "${TRAIN_COMMON[@]}" \
    train_talker.init_from=checkpoints/talker_hifi_pre train_talker.output_dir=checkpoints/talker_hifi \
    train_talker.max_steps="$FT_STEPS" train_talker.lr=0.0001 train_talker.warmup_steps=300 \
    "train_talker.train_manifests=[{path: $M/hifi${SPEAKER}_train.jsonl, weight: 1.0}]" \
    train_talker.valid_manifest=$M/hifi${SPEAKER}_valid.jsonl
hf_push checkpoints/talker_hifi checkpoints/talker_hifi

# ------------------------------------------------- 4. compare on hotel replies (Whisper WER + audio)
compare() {
    for t in talker_v3 talker_hifi_pre talker_hifi; do
        echo "== $t"
        python -m s2s.eval.talker $CFG --set train_talker.thinker_dir=$TH --talker-dir checkpoints/$t \
            --manifest $M/talker_v3_valid.jsonl --max 100 --out-dir "$LOGS/hifi_eval/$t" | grep -e WER -e runaway
    done
}
stage hifi_compare compare
hf_push "$LOGS/hifi_eval" logs/hifi_eval
say "HIFI TALKER DONE: checkpoints/talker_hifi (listen: $LOGS/hifi_eval/*). Destroy the droplet when finished."
