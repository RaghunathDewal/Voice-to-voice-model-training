#!/usr/bin/env bash
# Full retraining on one AMD MI300X (192 GB) droplet: thinker v3 -> replies -> adapter v4 -> talker v3.
#
#   export HF_TOKEN=hf_...            # write access to $HF_USER/s2s-checkpoints and $HF_USER/s2s-data-pk
#   export DO_TOKEN=dop_v1_...        # optional: DigitalOcean token (droplet read + delete) for the kill switch
#   bash scripts/mi300x_train.sh 2>&1 | tee -a ~/run.log
#
# Order matters: the adapter and the talker are trained on the thinker's exact behaviour, so the
# thinker is retrained first and frozen. Every stage writes a marker in $WORK/done/ and is skipped
# when the script is run again, so a crash or a new droplet resumes where it stopped (after the
# first stages have uploaded their outputs). Checkpoints and logs are pushed to Hugging Face every
# 10 minutes, so destroying the droplet loses at most 10 minutes of work.
#
# With DO_TOKEN set the droplet destroys itself after MAX_HOURS (default 14), and when the script
# finishes or fails. Destroying (not powering off) is what stops the billing.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
MAX_HOURS=${MAX_HOURS:-14}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$( [ -d /mnt/scratch ] && echo /mnt/scratch/s2s || echo "$HOME/s2s_work")}
CFG="--config configs/small.yaml"
E=parakeet:nvidia/parakeet-ctc-0.6b
M=data/manifests

# sizes for one MI300X (192 GB): bigger batches than the 2xT4 runs, same learning rates
THINKER_STEPS=${THINKER_STEPS:-1500}       # batch 32: ~48k conversations seen
ADAPTER_STEPS=${ADAPTER_STEPS:-8000}       # batch 64
# after an interruption: ADAPTER_INIT=checkpoints/speech_llm_pk4 ADAPTER_STEPS=<steps left> ADAPTER_WARMUP=100
ADAPTER_INIT=${ADAPTER_INIT:-checkpoints/speech_llm_pk3}
ADAPTER_WARMUP=${ADAPTER_WARMUP:-300}
TALKER_STEPS=${TALKER_STEPS:-30000}        # batch 32, small talker (47M) continued from talker_v2
TALKER_TEXTS=${TALKER_TEXTS:-40000}        # sentences spoken in af_heart for the talker

# the token from `export HF_TOKEN=...`, or from a saved `huggingface-cli login` (for background runs)
HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
: "${HF_TOKEN:?export HF_TOKEN first (or log in once: huggingface-cli login)}"
export HF_TOKEN PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30 TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK"/{data,checkpoints,logs,done,tmp}
cd "$REPO_DIR"
# Python: where torch already exists (e.g. inside the image's `rocm` container) use that interpreter
# directly; otherwise a venv on the host, where setup installs the ROCm build of torch.
SYS_PY=""
for c in ${PYTHON:-} $(command -v python3 python 2>/dev/null) /opt/venv/bin/python3 /opt/conda/bin/python3 \
         /opt/conda/envs/*/bin/python3 /opt/*/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null; then SYS_PY=$c; break; fi
done
if [ -n "$SYS_PY" ]; then
    echo "using $SYS_PY ($("$SYS_PY" -c 'import torch; print("torch", torch.__version__)'))"
    mkdir -p "$WORK/bin"
    rm -f "$WORK/bin/python"   # a wrapper, not a symlink: a symlinked venv python loses its venv
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$SYS_PY" > "$WORK/bin/python" && chmod +x "$WORK/bin/python"
    printf '#!/bin/sh\nexec "%s" -m pip "$@"\n' "$SYS_PY" > "$WORK/bin/pip" && chmod +x "$WORK/bin/pip"
    export PATH="$WORK/bin:$PATH" PIP_BREAK_SYSTEM_PACKAGES=1   # Ubuntu 24.04 marks system Python as managed
else
    SYS_PY=$(command -v python3 || command -v python)
    echo "no Python with torch found (inside the rocm container it should exist: try PYTHON=/path/to/python3)"
    if [ ! -x "$WORK/venv/bin/pip" ]; then   # also repairs a venv made before python3-venv was installed
        rm -rf "$WORK/venv"
        "$SYS_PY" -m venv --system-site-packages "$WORK/venv" || {
            echo "python3-venv missing: run  apt-get install -y python3-venv  and start again"; exit 1; }
    fi
    export PATH="$WORK/venv/bin:$PATH"
fi
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
LOGS=$WORK/logs

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ------------------------------------------------------------- kill switch
DROPLET_ID=$(curl -s --max-time 3 http://169.254.169.254/metadata/v1/id || true)
cleanup() {
    kill "${SYNC_PID:-0}" 2>/dev/null || true
    if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ]; then
        say "destroying droplet $DROPLET_ID in 120 s (last uploads)"; sleep 120
        curl -s -X DELETE -H "Authorization: Bearer $DO_TOKEN" "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
    fi
}
if [ -n "${DO_TOKEN:-}" ] && [ -n "$DROPLET_ID" ]; then
    nohup bash -c "sleep $((MAX_HOURS * 3600)); curl -s -X DELETE -H 'Authorization: Bearer $DO_TOKEN' \
        https://api.digitalocean.com/v2/droplets/$DROPLET_ID" > "$LOGS/killswitch.log" 2>&1 &
    say "kill switch armed: droplet $DROPLET_ID is destroyed after $MAX_HOURS h"
else
    say "no DO_TOKEN / droplet id: no kill switch. Destroy the droplet yourself when done."
fi

# ------------------------------------------------------- Hugging Face sync
hf_push() {  # hf_push <local folder> <path in repo> [repo type] [repo]; fails if the upload fails
    python - "$@" <<'EOF'
import sys
from huggingface_hub import HfApi
folder, path = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "model"
repo = sys.argv[4] if len(sys.argv) > 4 else None
import os
repo = repo or os.environ["CKPT_REPO"]
try:
    HfApi().upload_folder(repo_id=repo, repo_type=kind, folder_path=folder, path_in_repo=path,
                          commit_message=f"mi300x: {path}")
    print(f"[hf] {folder} -> {repo}/{path}", flush=True)
except Exception as e:
    print(f"[hf] upload failed: {e}", flush=True)
    sys.exit(1)
EOF
}
export CKPT_REPO
SYNC_DIRS="$WORK/sync_dirs"; : > "$SYNC_DIRS"
( while sleep 600; do
    while read -r d; do [ -d "checkpoints/$d" ] && { hf_push "checkpoints/$d" "checkpoints/$d" || true; }; done < "$SYNC_DIRS"
    hf_push "$LOGS" logs/mi300x || true
  done ) > "$LOGS/sync.log" 2>&1 &
SYNC_PID=$!
trap cleanup EXIT   # stop the uploader; with DO_TOKEN, also destroy the droplet (done or failed)

stage() {  # stage <name> <command...>: run once, log to $LOGS/<name>.log
    local name=$1; shift
    if [ -f "$WORK/done/$name" ]; then say "skip $name (done)"; return; fi
    say ">>> $name"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$LOGS/$name.log"
    touch "$WORK/done/$name"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}

# ------------------------------------------------------------------ setup
setup() {
    # The PyTorch image keeps torch inside its `rocm` container; run this script there. On a bare host,
    # the ROCm build is installed instead
    if ! python -c "import torch" 2>/dev/null; then
        say "torch not found on the host: installing the ROCm build from ${TORCH_INDEX:=https://download.pytorch.org/whl/rocm7.1}"
        pip install -q --upgrade pip
        pip install -q torch --index-url "$TORCH_INDEX"
    fi
    python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('torch', torch.__version__, \
'hip', torch.version.hip, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GB')"
    pip install -q -e ".[demo]" "kokoro>=0.9" "misaki[en]"   # the image's ROCm torch is kept as is
    python -m spacy download en_core_web_sm -q || true       # misaki's English model (else fetched on first use)
    $( [ "$(id -u)" = 0 ] || echo sudo ) apt-get update -q > /dev/null
    $( [ "$(id -u)" = 0 ] || echo sudo ) apt-get install -y -q espeak-ng > /dev/null
    python -c "import kokoro, torch; assert torch.version.hip, 'torch lost its ROCm build'; print('kokoro OK')"
}
stage setup setup
KOKORO=python

download() {
    python - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download
import subprocess
snapshot_download("$CKPT_REPO", local_dir=".", allow_patterns=["checkpoints/thinker_merged_v2/*",
                  "checkpoints/speech_llm_pk3/*"])
snapshot_download("$DATA_REPO", repo_type="dataset", local_dir="data", allow_patterns=["manifests/*"])
for i in range(15):  # Parakeet features of Common Voice, People's Speech, Svarah, VoxPopuli, hotel3
    t = hf_hub_download("$DATA_REPO", f"pk-{i:03d}.tar", repo_type="dataset", local_dir="$WORK/tmp")
    subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
EOF
    du -sh data/features
}
stage download download

# ------------------------------------------------- 1. thinker v3 (text only)
thinker_data() {
    python -m s2s.prep.thinker_sft_data $CFG --version 3 --hotel 24000 --hotel-v2 3000 --xlam 4000 --hermes 1500
    python -m s2s.prep.hotel_data $CFG --version 3 --train 12000 --eval 300 --prefix hotel4
}
stage thinker_data thinker_data

echo thinker_text_v3 >> "$SYNC_DIRS"
stage thinker_train python -m s2s.train.thinker_text $CFG --base checkpoints/thinker_merged_v2 \
    --out checkpoints/thinker_text_v3 --steps "$THINKER_STEPS" --batch 32 --accum 1 --lr 2e-4 --warmup 50 \
    --eval-every 250 --save-every 250 --eval-rows 300 --data-version 3 --upload-repo "$CKPT_REPO"

thinker_merge() {
    python -m s2s.prep.merge_lora $CFG --set thinker.model=checkpoints/thinker_merged_v2 \
        --speech-llm-dir checkpoints/thinker_text_v3 --out checkpoints/thinker_merged_v3
    python -m s2s.eval.text_tools $CFG --model checkpoints/thinker_merged_v3 --manifest $M/hotel4_eval_text.jsonl \
        --out "$LOGS/thinker_v3_eval.json" | tail -25
    hf_push checkpoints/thinker_merged_v3 checkpoints/thinker_merged_v3 || true
}
stage thinker_merge thinker_merge
TH="thinker.model=checkpoints/thinker_merged_v3"

# ------------------------------------------- 2. replies from the new thinker
replies() {
    for pair in svarah_train_pk:6000 voxpop_acc_pk:6000 peoples_train_pk:6000 cv_train_pk:25000; do
        name=${pair%%:*}; n=${pair##*:}
        python -m s2s.prep.distill $CFG --set $TH --gpus 1 --batch-size 192 --max-new-tokens 80 --max-utts "$n" \
            --in $M/$name.jsonl --out $M/${name}_d3.jsonl | tail -2
        python -m s2s.prep.reply_texts strip --in $M/${name}_d3.jsonl --out $M/${name}_d3f.jsonl
    done
}
stage replies replies

# ------------------------------- 3. hotel v3 speech (20 voices) + Parakeet features
VOICES="af_heart af_bella af_sarah af_sky af_nova af_river af_jessica af_kore am_adam am_michael \
am_eric am_liam am_onyx am_echo bf_alice bf_isabella bf_lily bm_daniel bm_lewis bm_fable"
hotel_speech() {
    $KOKORO -m s2s.prep.synth_kokoro $CFG --in $M/hotel4_train_text.jsonl --out $M/hotel4_train_audio.jsonl \
        --voices $VOICES --speed-jitter 0.12 | tail -1
    $KOKORO -m s2s.prep.synth_kokoro $CFG --in $M/hotel4_eval_text.jsonl --out $M/hotel4_eval_audio.jsonl \
        --voices af_nicole am_puck bf_emma bm_george --speed-jitter 0.1 --seed 1 | tail -1
    for job in hotel4_train_audio:hotel4_train_pk:0.5 hotel4_eval_audio:hotel4_eval_clean_pk:0.0 \
               hotel4_eval_audio:hotel4_eval_noisy_pk:1.0; do
        IFS=: read -r src out aug <<< "$job"
        python -m s2s.prep.extract_mimi $CFG --mode latents --encoder $E --augment-prob "$aug" \
            --in $M/$src.jsonl --out $M/$out.jsonl | tail -1
    done
    mkdir -p "$WORK/up/manifests"
    tar -C data -cf "$WORK/up/pk-015.tar" features/hotel4_train_pk features/hotel4_eval_clean_pk features/hotel4_eval_noisy_pk
    cp $M/hotel4_*_pk.jsonl $M/*_d3.jsonl "$WORK/up/manifests/"
    if hf_push "$WORK/up" "" dataset "$DATA_REPO"; then rm -rf "$WORK/up"; fi
}
stage hotel_speech hotel_speech

# ------------------------------------------------------- 4. adapter v4
echo speech_llm_pk4 >> "$SYNC_DIRS"
# ~74% real speakers (accents, real microphones, meetings) for robust hearing, ~26% synthetic hotel speech
# for the domain words, numbers and the long tool prompt; 35% of samples are exact transcription.
# The extra real sets come from scripts/mi300x_more_speech.sh: wait for it while it runs, then use
# every set that finished.
if [ -f "$WORK/more_running" ]; then
    say "waiting for mi300x_more_speech.sh to finish (max 3 h) ..."
    for _ in $(seq 180); do [ -f "$WORK/more_running" ] || break; sleep 60; done
fi
mix() {  # mix <manifest> <weight>: one entry of the adapter mix, only if the manifest exists
    [ -f "$1" ] && printf '{path: %s, weight: %s}, ' "$1" "$2"
    return 0
}
more() { [ -f "$WORK/done/more_$1" ] && mix "$M/${1}_pk_d3f.jsonl" "$2"; return 0; }
TRAIN="[$(mix $M/hotel4_train_pk.jsonl 0.23)$(mix $M/hotel3_train_pk.jsonl 0.03)\
$(mix $M/cv_train_pk_d3f.jsonl 0.20)$(mix $M/svarah_train_pk_d3f.jsonl 0.13)\
$(mix $M/voxpop_acc_pk_d3f.jsonl 0.09)$(mix $M/peoples_train_pk_d3f.jsonl 0.08)\
$(more cv_india_train 0.06)$(more ami_ihm_train 0.05)$(more ami_sdm_train 0.04)\
$(more peoples_dirty_train 0.05)$(more mls_train 0.04)]"
TRAIN=${TRAIN/%, ]/]}
say "adapter mix: $TRAIN"
stage adapter_train python -m s2s.train.speech_llm $CFG --set adapter.encoder=$E $TH \
    train_speech_llm.train_lora=false train_speech_llm.init_from="$ADAPTER_INIT" \
    train_speech_llm.transcribe_prob=0.35 \
    train_speech_llm.output_dir=checkpoints/speech_llm_pk4 train_speech_llm.max_steps="$ADAPTER_STEPS" \
    train_speech_llm.batch_size=64 train_speech_llm.grad_accum=1 train_speech_llm.warmup_steps="$ADAPTER_WARMUP" \
    train_speech_llm.num_workers=12 train_speech_llm.eval_every=500 train_speech_llm.save_every=500 \
    "train_speech_llm.train_manifests=$TRAIN" train_speech_llm.valid_manifest=$M/hotel4_eval_noisy_pk.jsonl

adapter_eval() {
    for m in hotel4_eval_clean_pk hotel4_eval_noisy_pk svarah_eval_pk; do
        echo "===== $m"
        python -m s2s.eval.speech_llm $CFG --set $TH --speech-llm-dir checkpoints/speech_llm_pk4 \
            --manifest $M/$m.jsonl --max 300 | tail -25
    done
    hf_push checkpoints/speech_llm_pk4 checkpoints/speech_llm_pk4 || true
}
stage adapter_eval adapter_eval

# ------------------------------------------------- 5. talker v3 (full size)
talker_data() {
    python -m s2s.prep.reply_texts talker --hotel 20000 --hotel-version 3 --replies "$M/*_d3.jsonl" \
        --out $M/talker_v3_text.jsonl --max "$TALKER_TEXTS"
    $KOKORO -m s2s.prep.synth_kokoro $CFG --in $M/talker_v3_text.jsonl --out $M/talker_v3_audio.jsonl \
        --voices af_heart | tail -1
    head -300 $M/talker_v3_audio.jsonl > $M/talker_v3_valid_raw.jsonl
    tail -n +301 $M/talker_v3_audio.jsonl > $M/talker_v3_train_raw.jsonl
    for n in talker_v3_train talker_v3_valid; do
        python -m s2s.prep.extract_mimi $CFG --mode codes --in $M/${n}_raw.jsonl --out $M/$n.jsonl | tail -1
    done
    mkdir -p "$WORK/up/manifests"
    tar -C data -cf "$WORK/up/talker-001.tar" features/talker_v3_train features/talker_v3_valid
    cp $M/talker_v3_train.jsonl $M/talker_v3_valid.jsonl "$WORK/up/manifests/"
    if hf_push "$WORK/up" "" dataset "$DATA_REPO"; then rm -rf "$WORK/up"; fi
}
stage talker_data talker_data

echo talker_v3 >> "$SYNC_DIRS"
# small.yaml's talker (47M, keeps the model small and fast), continued from talker_v2 on more data
fetch_talker_v2() {
    python -c "from huggingface_hub import snapshot_download as s; s('$CKPT_REPO', local_dir='.', \
allow_patterns=['checkpoints/talker_v2/talker.pt', 'checkpoints/talker_v2/meta.json', 'checkpoints/talker_v2/config.yaml'])"
}
stage fetch_talker_v2 fetch_talker_v2
stage talker_train python -m s2s.train.talker $CFG --set train_talker.init_from=checkpoints/talker_v2 \
    train_talker.thinker_dir=checkpoints/thinker_merged_v3 train_talker.output_dir=checkpoints/talker_v3 \
    train_talker.max_steps="$TALKER_STEPS" train_talker.batch_size=32 train_talker.grad_accum=1 \
    train_talker.lr=0.0003 train_talker.warmup_steps=1000 train_talker.num_workers=12 \
    train_talker.eval_every=2000 train_talker.save_every=2000 \
    "train_talker.train_manifests=[{path: $M/talker_v3_train.jsonl, weight: 1.0}]" \
    train_talker.valid_manifest=$M/talker_v3_valid.jsonl

talker_eval() {
    python -m s2s.eval.talker $CFG --set train_talker.thinker_dir=checkpoints/thinker_merged_v3 \
        --talker-dir checkpoints/talker_v3 --manifest $M/talker_v3_valid.jsonl --max 100 \
        --out-dir "$LOGS/talker_v3_eval" | grep -e WER -e runaway
    hf_push checkpoints/talker_v3 checkpoints/talker_v3 || true
}
stage talker_eval talker_eval

hf_push "$LOGS" logs/mi300x || true
say "ALL DONE: thinker_merged_v3, speech_llm_pk4, talker_v3 are on $CKPT_REPO"
