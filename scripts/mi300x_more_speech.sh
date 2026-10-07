#!/usr/bin/env bash
# More real English speech for the adapter, run next to scripts/mi300x_train.sh on the same droplet
# (a second tmux window, a second clone of the repo, the same $WORK). It does not change the model's
# size, only what the adapter hears in training:
#
#   AMI meetings (CC BY 4.0)         spontaneous talk, "um"s and interruptions; headset + distant room mic
#   MLS English (CC BY 4.0)          read speech from thousands of speakers (capped per speaker)
#   People's Speech "dirty" (CC BY)  noisier real recordings
#   Common Voice, Indian accents     real volunteers, consumer microphones (CC0)
#
#   export HF_TOKEN=hf_...
#   bash scripts/mi300x_more_speech.sh 2>&1 | tee -a ~/more_speech.log
#
# Each set: download -> Parakeet features (half with noise / phone / room augmentation) -> once
# checkpoints/thinker_merged_v3 exists, the new thinker's replies. mi300x_train.sh adds every set
# that finished (marker $WORK/done/more_<name>) to the adapter mix, and waits for this script while
# $WORK/more_running exists.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
DATA_REPO=$HF_USER/s2s-data-pk
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s || echo "$HOME/s2s_work")}
CFG="--config configs/small.yaml"
E=parakeet:nvidia/parakeet-ctc-0.6b
M=data/manifests
: "${HF_TOKEN:?export HF_TOKEN first}"
export HF_TOKEN PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30 TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK"/{data,checkpoints,logs,done,tmp}
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
# the wrappers mi300x_train.sh made around the Python that has torch
[ -x "$WORK/bin/python" ] && export PATH="$WORK/bin:$PATH" PIP_BREAK_SYSTEM_PACKAGES=1
[ -x "$WORK/venv/bin/python" ] && [ ! -x "$WORK/bin/python" ] && export PATH="$WORK/venv/bin:$PATH"
python -c "import torch, s2s; assert torch.cuda.is_available()" || {
    echo "run scripts/mi300x_train.sh first (it sets up Python), and run this inside the rocm container"; exit 1; }

touch "$WORK/more_running"
trap 'rm -f "$WORK/more_running"' EXIT
say() { echo "[$(date +%H:%M:%S)] $*"; }
log() { grep --line-buffered -v -e Warning -e warnings.warn || true; }

# name | preset | hf_asr options (hours are of speech kept after filtering)
JOBS=(
  "ami_ihm_train|ami_ihm|--max-hours 20 --min-words 4 --min-seconds 1.5"
  "ami_sdm_train|ami_sdm|--max-hours 15 --min-words 4 --min-seconds 1.5"
  "mls_train|mls|--max-hours 25 --max-per-accent 0.1"
  "peoples_dirty_train|peoples_dirty|--max-hours 25"
  "cv_india_train|commonvoice|--max-hours 15 --accent-match India --max-shards 120"
)

# 1. download + Parakeet features (runs while the thinker trains)
for job in "${JOBS[@]}"; do
    IFS='|' read -r name preset opts <<< "$job"
    [ -f "$WORK/done/more_feat_$name" ] && { say "skip features $name"; continue; }
    say ">>> $name: download"
    # shellcheck disable=SC2086
    python -m s2s.prep.hf_asr $CFG --preset "$preset" --split train --name "$name" $opts 2>&1 | log | tail -3
    say ">>> $name: Parakeet features"
    python -m s2s.prep.extract_mimi $CFG --mode latents --encoder $E --augment-prob 0.5 \
        --in $M/${name}_raw.jsonl --out $M/${name}_pk.jsonl 2>&1 | log | tail -1
    rm -rf "data/hf_asr/$name"          # the FLAC copies are no longer needed
    touch "$WORK/done/more_feat_$name"
done

# 2. replies from the new thinker (needs thinker_merged_v3 from mi300x_train.sh)
until [ -f "$WORK/done/thinker_merge" ]; do say "waiting for thinker_merged_v3 ..."; sleep 120; done
TH="thinker.model=checkpoints/thinker_merged_v3"
for job in "${JOBS[@]}"; do
    IFS='|' read -r name _ _ <<< "$job"
    [ -f "$WORK/done/more_$name" ] && { say "skip replies $name"; continue; }
    say ">>> $name: replies"
    python -m s2s.prep.distill $CFG --set $TH --gpus 1 --batch-size 192 --max-new-tokens 80 --max-utts 20000 \
        --in $M/${name}_pk.jsonl --out $M/${name}_pk_d3.jsonl 2>&1 | log | tail -2
    python -m s2s.prep.reply_texts strip --in $M/${name}_pk_d3.jsonl --out $M/${name}_pk_d3f.jsonl
    touch "$WORK/done/more_$name"
done

# 3. keep a copy on Hugging Face (features + manifests), so a new droplet does not redo this
say ">>> upload"
mkdir -p "$WORK/up_more/manifests"
for job in "${JOBS[@]}"; do
    IFS='|' read -r name _ _ <<< "$job"
    tar -C data -cf "$WORK/up_more/more-$name.tar" "features/${name}_pk"
    cp $M/${name}_pk.jsonl $M/${name}_pk_d3.jsonl "$WORK/up_more/manifests/"
done
python - <<EOF && rm -rf "$WORK/up_more"
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="$DATA_REPO", repo_type="dataset", folder_path="$WORK/up_more",
                      commit_message="more real speech: AMI, MLS, People's Speech dirty, CV India (parakeet)")
EOF
du -sh data/features/*_train_pk 2>/dev/null | tail -8
say "MORE SPEECH DONE"
