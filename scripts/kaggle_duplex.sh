#!/usr/bin/env bash
# Full-duplex experiment on Kaggle (2x T4): a Qwen3-0.6B that listens (Nemotron vectors) and speaks (Mimi
# codes) every 80 ms, with its inner text and tool calls, trained on generated two-speaker hotel calls.
#
#   PART=data   (~2-3 h)  call scripts -> Kokoro speech (guest: 23 voices + noise / phone line, agent: one
#                         voice) -> Nemotron features + Mimi codes -> HF data repo (duplex/...)
#   PART=train  (~8 h)    LoRA + new layers, saved to HF every 20 min; a rerun continues from the last save;
#                         ends with the evaluation (PART=eval)
#   PART=eval   (~30 min) 50 held-out calls run frame by frame: answers, response gap, talking over the guest,
#                         stopping when interrupted, tools, intelligibility; stereo wavs to listen to
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN with WRITE access to the two repos below.
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !cd /tmp/repo && PART=data bash scripts/kaggle_duplex.sh
set -euo pipefail

PART=${PART:?set PART=data, train or eval}
HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
BUDGET_MIN=${BUDGET_MIN:-500}
N_TRAIN=${N_TRAIN:-10000}            # calls (~30 s each -> ~85 h of two-channel audio)
N_EVAL=${N_EVAL:-200}
N_TEST=${N_TEST:-50}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_duplex}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/duplex_$PART || echo "$WORK/out_$PART")}
CFG="--config configs/default.yaml"
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN CKPT_REPO DATA_REPO WORK PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 1 ] || NGPU=1
mkdir -p "$WORK"/{data,checkpoints,tmp} "$OUT/logs"
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
mkdir -p data/manifests
T_START=$SECONDS
say() { echo "[$(date +%H:%M:%S)] $*"; }
minutes_left() { echo $(( BUDGET_MIN - (SECONDS - T_START) / 60 )); }
step() {
    local name=$1; shift
    say ">>> $name ($(minutes_left) min left)"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$OUT/logs/$name.log"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}

hf() {  # hf push <folder> <path> [dataset] | hf get_data | hf get_ckpt
    python - "$@" <<'EOF'
import os, subprocess, sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api, cmd, args = HfApi(), sys.argv[1], sys.argv[2:]
ckpt, data, work = os.environ["CKPT_REPO"], os.environ["DATA_REPO"], os.environ["WORK"]
if cmd == "push":
    kind = args[2] if len(args) > 2 else "model"
    try:
        api.upload_folder(repo_id=ckpt if kind == "model" else data, repo_type=kind, folder_path=args[0],
                          path_in_repo=args[1], commit_message=f"kaggle duplex: {args[1] or 'data'}")
        print(f"[hf] {args[0]} -> {args[1]}", flush=True)
    except Exception as e:
        print(f"[hf] upload failed: {e}", flush=True); sys.exit(1)
elif cmd == "get_data":
    snapshot_download(data, repo_type="dataset", local_dir=".", allow_patterns=["duplex/manifests/*"])
    subprocess.run("cp duplex/manifests/*.jsonl data/manifests/", shell=True, check=True)
    for name in ("duplex/train.tar", "duplex/eval.tar"):
        t = hf_hub_download(data, name, repo_type="dataset", local_dir=work + "/tmp")
        subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
        print(f"  {name} extracted", flush=True)
elif cmd == "get_ckpt":
    files = api.list_repo_files(ckpt)
    if any(f.startswith("checkpoints/duplex/") for f in files):
        snapshot_download(ckpt, local_dir=".", allow_patterns=["checkpoints/duplex/*"])
        print("found checkpoints/duplex", flush=True)
EOF
}

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer peft accelerate soundfile librosa huggingface_hub
python -c "from transformers import AutoModelForRNNT" 2>/dev/null || pip install -q -U "transformers>=5.13"
python -c "import torch, transformers; print('transformers', transformers.__version__, '|', torch.cuda.device_count(), 'x', \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
[ "$NGPU" -gt 1 ] && TRAIN="torchrun --standalone --nproc_per_node=$NGPU -m" || TRAIN="python -m"

# =============================================================== PART data
if [ "$PART" = data ]; then
    apt-get -qq install -y espeak-ng > /dev/null 2>&1 || true
    pip install -q "kokoro>=0.9" "misaki[en]"
    step script python -m s2s.prep.duplex_data $CFG script --train "$N_TRAIN" --eval "$N_EVAL"
    step audio_eval python -m s2s.prep.duplex_data $CFG audio --split eval --gpus "$NGPU"
    step audio_train python -m s2s.prep.duplex_data $CFG audio --split train --gpus "$NGPU"
    step features_eval python -m s2s.prep.duplex_data $CFG features --split eval --gpus "$NGPU"
    step features_train python -m s2s.prep.duplex_data $CFG features --split train --gpus "$NGPU" \
        --augment-prob 0.5 --phone-prob 0.4 --drop-audio
    upload() {
        mkdir -p "$WORK/up/duplex/manifests"
        cp data/manifests/duplex_*.jsonl "$WORK/up/duplex/manifests/"
        tar -C data -cf "$WORK/up/duplex/train.tar" duplex/train   # features only (wavs dropped)
        tar -C data -cf "$WORK/up/duplex/eval.tar" duplex/eval                       # + audio, for listening
        du -sh "$WORK/up"
        hf push "$WORK/up" "" dataset && rm -rf "${WORK:?}/up"
    }
    step upload upload
    say "PART data DONE"
fi

# =============================================================== PART train
if [ "$PART" = train ] || [ "$PART" = eval ]; then
    step download bash -c "$(declare -f hf); hf get_data; hf get_ckpt"
fi
if [ "$PART" = train ]; then
    INIT=null
    if [ -f checkpoints/duplex/duplex.pt ]; then
        [ -f checkpoints/duplex/DONE ] && { say "checkpoints/duplex is already DONE; delete DONE on HF to train on"; exit 0; }
        INIT=checkpoints/duplex_prev; rm -rf "$INIT"; cp -r checkpoints/duplex "$INIT"
    fi
    ( while sleep 1200; do [ -f checkpoints/duplex/duplex.pt ] && hf push checkpoints/duplex checkpoints/duplex; done ) \
        > "$OUT/logs/sync.log" 2>&1 &
    SYNC=$!
    MIN=$(( $(minutes_left) - 50 ))   # leave time for the evaluation
    step train $TRAIN s2s.train.duplex $CFG --set train_duplex.init_from=$INIT train_duplex.max_minutes="$MIN" \
        train_duplex.max_steps=20000
    kill $SYNC 2>/dev/null || true
    touch checkpoints/duplex/DONE
    hf push checkpoints/duplex checkpoints/duplex
fi
if [ "$PART" = train ] || [ "$PART" = eval ]; then
    step eval python -m s2s.eval.duplex_eval --ckpt checkpoints/duplex --manifest data/manifests/duplex_eval.jsonl \
        --n "$N_TEST" --out "$OUT/eval"
    mkdir -p "$WORK/res" && cp "$OUT"/eval/summary.* "$WORK/res/"
    for w in $(ls "$OUT"/eval/*.wav | head -5); do cp "$w" "$WORK/res/"; done
    hf push "$WORK/res" results/duplex dataset || true
    cat "$OUT/eval/summary.md"
    say "PART $PART DONE: listen to $OUT/eval/*.wav (left = guest, right = agent)"
fi
