#!/usr/bin/env bash
# Speech adapter + talker for Qwen3-4B-Instruct-2507 (used as is, no fine-tuning) on Kaggle 2x T4,
# in two sessions that each fit Kaggle's 12 h limit. The two parts are independent (any order).
#
#   PART=1  (~10 h)  data download -> the 4B's own replies (distillation targets) -> adapter (time-boxed)
#                    -> adapter test -> upload checkpoints/speech_llm_4b
#   PART=2  (~8 h)   talker data -> talker (time-boxed) -> talker test with Whisper + audio files
#                    -> upload checkpoints/talker_4b
#
# Kaggle notebook: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN (Add-ons > Secrets) with write
# access to $HF_USER/s2s-checkpoints and $HF_USER/s2s-data-pk. Then "Save Version > Save & Run All":
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !cd /tmp/repo && PART=1 bash scripts/kaggle_4b_train.sh
#
# Both parts start from the 0.6B-era checkpoints (adapter speech_llm_pk4, talker talker_v3): the speech
# encoder and the voice are kept, only the layers that connect to the thinker start fresh (the 4B is
# 2560 wide, the 0.6B 1024). Progress is pushed to Hugging Face every 20 minutes; logs and the test
# results are also written to /kaggle/working (the version's Output tab).
set -euo pipefail

PART=${PART:?set PART=1 (adapter) or PART=2 (talker)}
HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
ADAPTER_INIT=${ADAPTER_INIT:-speech_llm_pk4}   # falls back to speech_llm_pk3
TALKER_INIT=${TALKER_INIT:-talker_v3}          # falls back to talker_v2
ADAPTER_MINUTES=${ADAPTER_MINUTES:-450}        # training time inside the 12 h session (rest: download, replies, test)
TALKER_MINUTES=${TALKER_MINUTES:-400}
ADAPTER_BATCH=${ADAPTER_BATCH:-4}              # per GPU; x accum 4 x 2 GPUs = 32 per step. OOM -> 2 (and ACCUM=16)
ADAPTER_ACCUM=${ADAPTER_ACCUM:-8}
TALKER_BATCH=${TALKER_BATCH:-16}               # per GPU; x 2 GPUs = 32 per step
REPLIES=${REPLIES:-"cv_train_pk:14000 svarah_train_pk:5000 voxpop_acc_pk:5000 peoples_train_pk:6000"}

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_work}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || echo "$WORK/out")}
CFG="--config configs/qwen4b.yaml"
M=data/manifests
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets, then os.environ['HF_TOKEN'] = ...)}"
export HF_TOKEN CKPT_REPO DATA_REPO PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
LOGS=$OUT/logs_4b_part$PART
mkdir -p "$WORK"/{data,checkpoints,tmp} "$LOGS"
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
mkdir -p data/manifests

say() { echo "[$(date +%H:%M:%S)] $*"; }
step() {  # step <name> <command...>: run, log to $LOGS/<name>.log, stop the script if it fails
    local name=$1; shift
    say ">>> $name"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$LOGS/$name.log"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}

hf() {  # python helpers around huggingface_hub: hf push <folder> <path in repo> [model|dataset]
    python - "$@" <<'EOF'
import os, sys, subprocess
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api, cmd, args = HfApi(), sys.argv[1], sys.argv[2:]
ckpt, data = os.environ["CKPT_REPO"], os.environ["DATA_REPO"]
if cmd == "push":                       # push <folder> <path in repo> [model|dataset]
    kind = args[2] if len(args) > 2 else "model"
    try:
        api.upload_folder(repo_id=ckpt if kind == "model" else data, repo_type=kind, folder_path=args[0],
                          path_in_repo=args[1], commit_message=f"kaggle 4b: {args[1]}")
        print(f"[hf] {args[0]} -> {args[1]}", flush=True)
    except Exception as e:
        print(f"[hf] upload failed: {e}", flush=True)
        sys.exit(1)
elif cmd == "ckpt":                     # ckpt <name> [fallback]: first checkpoint folder that exists
    files = api.list_repo_files(ckpt)
    for name in args:
        if any(f.startswith(f"checkpoints/{name}/") for f in files):
            snapshot_download(ckpt, local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt", f"checkpoints/{name}/*.json",
                                                                   f"checkpoints/{name}/*.yaml"])
            print(name)
            sys.exit(0)
    sys.exit(f"none of {args} found in {ckpt}")
elif cmd == "data":                     # data <tar prefix>...: manifests + every matching feature tar
    snapshot_download(data, repo_type="dataset", local_dir="data", allow_patterns=["manifests/*"])
    tars = sorted(f for f in api.list_repo_files(data, repo_type="dataset")
                  if f.endswith(".tar") and any(f.startswith(p) for p in args))
    print(f"{len(tars)} feature archives: {tars}", flush=True)
    for name in tars:
        t = hf_hub_download(data, name, repo_type="dataset", local_dir=os.environ["WORK"] + "/tmp")
        subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
        print(f"  {name} extracted", flush=True)
EOF
}
export WORK

# progress to Hugging Face every 20 minutes (a crash or the 12 h limit loses at most 20 minutes)
SYNC=$([ "$PART" = 1 ] && echo speech_llm_4b || echo talker_4b)
( while sleep 1200; do
    [ -d "checkpoints/$SYNC" ] && hf push "checkpoints/$SYNC" "checkpoints/$SYNC" || true
    hf push "$LOGS" "logs/$(basename "$LOGS")" || true
  done ) > "$LOGS/sync.log" 2>&1 &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

setup() {
    pip install -q --no-deps -e .
    pip install -q jiwer peft accelerate soundfile huggingface_hub
    python -c "import torch; n = torch.cuda.device_count(); assert n, 'no GPU: set Accelerator to GPU T4 x2'; \
print('torch', torch.__version__, n, 'x', torch.cuda.get_device_name(0))"
    [ "$NGPU" -ge 2 ] || say "WARNING: only $NGPU GPU; set Accelerator to 'GPU T4 x2' (this will be ~2x slower)"
    df -h "$WORK" | tail -1
}
step setup setup

# ====================================================================== PART 1: adapter
if [ "$PART" = 1 ]; then
    download() {
        hf data pk- more-
        hf ckpt "$ADAPTER_INIT" speech_llm_pk3 > "$WORK/adapter_init"
        du -sh data/features; df -h "$WORK" | tail -1
    }
    step download download
    INIT=$(tail -1 "$WORK/adapter_init")

    # The adapter learns to make the 4B answer *speech* exactly as it answers the *text*: the targets are the
    # 4B's own replies to the transcripts (kept on Hugging Face, so a re-run skips this).
    replies() {
        for pair in $REPLIES; do
            name=${pair%%:*}; n=${pair##*:}
            [ -f "$M/${name}_q4f.jsonl" ] && { say "replies $name: already on Hugging Face"; continue; }
            [ -f "$M/$name.jsonl" ] || { say "replies $name: no manifest, skipped"; continue; }
            python -m s2s.prep.distill $CFG --gpus "$NGPU" --batch-size 48 --max-new-tokens 80 --max-utts "$n" \
                --in "$M/$name.jsonl" --out "$M/${name}_q4.jsonl" | tail -2
            python -m s2s.prep.reply_texts strip --in "$M/${name}_q4.jsonl" --out "$M/${name}_q4f.jsonl"
            mkdir -p "$WORK/up/manifests" && cp "$M/${name}_q4.jsonl" "$M/${name}_q4f.jsonl" "$WORK/up/manifests/"
        done
        if [ -d "$WORK/up" ]; then hf push "$WORK/up" "" dataset && rm -rf "$WORK/up"; fi
        for pair in $REPLIES; do   # a few examples, to check the 4B's replies look right
            f=$M/${pair%%:*}_q4f.jsonl
            [ -f "$f" ] && python -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; \
[print('  Q:', r['text'][:90], '\n  A:', r['response'][:160]) for r in [r for r in rows if r.get('response')][:2]]" "$f"
        done
    }
    step replies replies

    mix() {  # mix <manifest> <weight>: one entry of the training mix, only if the manifest exists
        [ -f "$1" ] && printf '{path: %s, weight: %s}, ' "$1" "$2"
        return 0
    }
    # ~74% real speakers (accents, microphones, meetings), ~26% synthetic hotel speech with tool prompts;
    # the extra real sets have no replies and are used for exact transcription only
    TRAIN="[$(mix $M/hotel4_train_pk.jsonl 0.23)$(mix $M/hotel3_train_pk.jsonl 0.03)\
$(mix $M/cv_train_pk_q4f.jsonl 0.20)$(mix $M/svarah_train_pk_q4f.jsonl 0.13)\
$(mix $M/voxpop_acc_pk_q4f.jsonl 0.09)$(mix $M/peoples_train_pk_q4f.jsonl 0.08)\
$(mix $M/cv_india_train_pk.jsonl 0.06)$(mix $M/ami_ihm_train_pk.jsonl 0.05)$(mix $M/ami_sdm_train_pk.jsonl 0.04)\
$(mix $M/peoples_dirty_train_pk.jsonl 0.05)$(mix $M/mls_train_pk.jsonl 0.04)]"
    TRAIN=${TRAIN/%, ]/]}
    say "adapter mix: $TRAIN"
    VALID=$M/hotel4_eval_noisy_pk.jsonl; [ -f "$VALID" ] || VALID=$M/hotel4_eval_clean_pk.jsonl

    step adapter_train torchrun --standalone --nproc_per_node="$NGPU" -m s2s.train.speech_llm $CFG --set \
        train_speech_llm.train_lora=false train_speech_llm.init_from="checkpoints/$INIT" \
        train_speech_llm.output_dir=checkpoints/speech_llm_4b train_speech_llm.max_steps=20000 \
        train_speech_llm.max_minutes="$ADAPTER_MINUTES" train_speech_llm.batch_size="$ADAPTER_BATCH" \
        train_speech_llm.grad_accum="$ADAPTER_ACCUM" train_speech_llm.warmup_steps=300 \
        train_speech_llm.transcribe_prob=0.35 train_speech_llm.num_workers=4 train_speech_llm.log_every=10 \
        train_speech_llm.eval_every=250 train_speech_llm.eval_batches=10 train_speech_llm.save_every=250 \
        "train_speech_llm.train_manifests=$TRAIN" train_speech_llm.valid_manifest="$VALID"
    hf push checkpoints/speech_llm_4b checkpoints/speech_llm_4b

    adapter_test() {   # speech in -> the 4B's answer, compared with its answer to the typed text
        for m in hotel4_eval_clean_pk svarah_eval_pk; do
            [ -f "$M/$m.jsonl" ] || continue
            echo "===== $m"
            python -m s2s.eval.speech_llm $CFG --speech-llm-dir checkpoints/speech_llm_4b \
                --manifest "$M/$m.jsonl" --max 150 --out "$OUT/adapter_4b_$m.json" | tail -25
        done
    }
    step adapter_test adapter_test
    hf push "$LOGS" "logs/$(basename "$LOGS")" || true
    say "PART 1 DONE: checkpoints/speech_llm_4b is on Hugging Face ($CKPT_REPO)"
fi

# ====================================================================== PART 2: talker
if [ "$PART" = 2 ]; then
    download() {
        hf data talker-
        hf ckpt "$TALKER_INIT" talker_v2 > "$WORK/talker_init"
        ls $M/talker_v3_*.jsonl; du -sh data/features
    }
    step download download
    INIT=$(tail -1 "$WORK/talker_init")

    # The 4B is frozen; for every clip it is run on the reply text and the talker learns to speak from the
    # 4B's token embeddings and hidden states (the same signals it gets live).
    step talker_train torchrun --standalone --nproc_per_node="$NGPU" -m s2s.train.talker $CFG --set \
        train_talker.init_from="checkpoints/$INIT" train_talker.thinker_dir=null \
        train_talker.output_dir=checkpoints/talker_4b train_talker.max_steps=30000 \
        train_talker.max_minutes="$TALKER_MINUTES" train_talker.batch_size="$TALKER_BATCH" train_talker.grad_accum=2 \
        train_talker.lr=0.0003 train_talker.warmup_steps=500 train_talker.num_workers=4 train_talker.log_every=20 \
        train_talker.eval_every=1000 train_talker.eval_batches=10 train_talker.save_every=500 \
        "train_talker.train_manifests=[{path: $M/talker_v3_train.jsonl, weight: 1.0}]" \
        train_talker.valid_manifest=$M/talker_v3_valid.jsonl
    hf push checkpoints/talker_4b checkpoints/talker_4b

    # voice clarity: held-out replies spoken by the new talker, transcribed by Whisper (lower WER = clearer);
    # the audio files are in the Output tab to listen to
    step talker_test python -m s2s.eval.talker $CFG --talker-dir checkpoints/talker_4b \
        --manifest $M/talker_v3_valid.jsonl --max 40 --out-dir "$OUT/talker_4b_eval"
    hf push "$LOGS" "logs/$(basename "$LOGS")" || true
    say "PART 2 DONE: checkpoints/talker_4b is on Hugging Face ($CKPT_REPO)"
fi
