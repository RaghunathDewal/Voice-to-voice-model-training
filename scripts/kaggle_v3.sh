#!/usr/bin/env bash
# Adapter v3 on Kaggle (2x T4), in three sessions of at most ~8.5 h each. Every session saves to Hugging Face
# and the next one continues from there; a session that stops early can simply be run again (it resumes).
#
#   PART=data    (~6 h)  original audio (public English speech + SpokenWOZ/SLURP train + regenerated hotel
#                        speech), Nemotron streaming features (160 ms) clean / noisy / phone line, "unclear"
#                        rows, the 4B's own answers as targets -> HF data repo (v3/...)
#   PART=align   (~8 h)  stage A: adapter aligned to the 4B's word vectors (no 4B layers in the loop)
#                        -> checkpoints/speech_llm_v3a
#   PART=polish  (~8 h)  stage B: KD to the 4B reading the text + answers / repeat / unclear / multi-turn / tools
#                        -> checkpoints/speech_llm_v3, then the public speech-vs-text test (same as the baseline)
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN with WRITE access to
# ThunderBlade7773/s2s-checkpoints and ThunderBlade7773/s2s-data-pk.
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !cd /tmp/repo && PART=data bash scripts/kaggle_v3.sh
# Licences: SpokenWOZ is CC BY-NC 4.0, so this adapter is for non-commercial use.
set -euo pipefail

PART=${PART:?set PART=data, align or polish}
HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
BUDGET_MIN=${BUDGET_MIN:-500}          # minutes this session may use in total (Kaggle stops at ~9-12 h)
SCALE=${SCALE:-1.0}                    # PART=data: multiply every hour / row count (0.05 for a quick trial)
N_TEST=${N_TEST:-150}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_v3}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/v3_$PART || echo "$WORK/out_$PART")}
CFG=${CFG:-"--config configs/qwen4b_v3.yaml"}
ENC=nemotron:nvidia/nemotron-speech-streaming-en-0.6b@1
M=data/manifests
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN CKPT_REPO DATA_REPO WORK PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
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
step() {  # step <name> <command...>: logged to $OUT/logs/<name>.log; stops the script if it fails
    local name=$1; shift
    say ">>> $name ($(minutes_left) min left)"
    local t0=$SECONDS
    "$@" 2>&1 | { grep --line-buffered -v -e "Warning" -e "warnings.warn" || true; } | tee "$OUT/logs/$name.log"
    say "<<< $name done in $(( (SECONDS - t0) / 60 )) min"
}
n() { python -c "import sys; print(max(1, int(float(sys.argv[1]) * float(sys.argv[2]))))" "$1" "$SCALE"; }

hf() {  # hf push <folder> <path> [dataset] | hf ckpt <name>... | hf v3data
    python - "$@" <<'EOF'
import os, subprocess, sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api, cmd, args = HfApi(), sys.argv[1], sys.argv[2:]
ckpt, data, work = os.environ["CKPT_REPO"], os.environ["DATA_REPO"], os.environ["WORK"]
if cmd == "push":
    kind = args[2] if len(args) > 2 else "model"
    try:
        api.upload_folder(repo_id=ckpt if kind == "model" else data, repo_type=kind, folder_path=args[0],
                          path_in_repo=args[1], commit_message=f"kaggle v3: {args[1] or 'data'}")
        print(f"[hf] {args[0]} -> {args[1]}", flush=True)
    except Exception as e:
        print(f"[hf] upload failed: {e}", flush=True); sys.exit(1)
elif cmd == "ckpt":                     # download each listed checkpoint folder that exists
    files = api.list_repo_files(ckpt)
    for name in args:
        if any(f.startswith(f"checkpoints/{name}/") for f in files):
            snapshot_download(ckpt, local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt", f"checkpoints/{name}/*.json",
                                                                   f"checkpoints/{name}/*.yaml", f"checkpoints/{name}/DONE"])
            print(f"found {name}", flush=True)
elif cmd == "v3tars":                   # names of the feature archives already uploaded
    print("\n".join(f for f in api.list_repo_files(data, repo_type="dataset") if f.startswith("v3/feat-")))
elif cmd in ("v3data", "v3manifests"):  # v3 manifests (+ every v3 feature archive for v3data)
    if not any(f.startswith("v3/manifests/") for f in api.list_repo_files(data, repo_type="dataset")):
        print("no v3 data uploaded yet", flush=True); sys.exit(0)
    snapshot_download(data, repo_type="dataset", local_dir=".", allow_patterns=["v3/manifests/*"])
    os.makedirs("data/manifests", exist_ok=True)
    subprocess.run("cp v3/manifests/*.jsonl data/manifests/", shell=True, check=True)
    if cmd == "v3manifests":
        sys.exit(0)
    tars = sorted(f for f in api.list_repo_files(data, repo_type="dataset") if f.startswith("v3/feat-") and f.endswith(".tar"))
    print(f"{len(tars)} v3 feature archives", flush=True)
    for name in tars:
        t = hf_hub_download(data, name, repo_type="dataset", local_dir=work + "/tmp")
        subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
        print(f"  {name} extracted", flush=True)
EOF
}

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer peft accelerate soundfile librosa pyarrow huggingface_hub
python -c "from transformers import AutoModelForRNNT" 2>/dev/null || pip install -q -U "transformers>=5.13"
python -c "import torch, transformers; print('transformers', transformers.__version__, '|', torch.cuda.device_count(), 'x', \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
[ "$NGPU" -gt 1 ] && TRAIN="torchrun --standalone --nproc_per_node=$NGPU -m" || TRAIN="python -m"

# =============================================================== PART data
if [ "$PART" = data ]; then
    # <name>|<hf_asr preset>|<split>|<extra args>|<4B answers for this many rows>
    SETS=(
        "cv_train|commonvoice|train|--max-hours $(n 50) --max-per-accent 0.15|$(n 12000)"
        "cv_india|commonvoice|train|--max-hours $(n 12)|$(n 4000)"
        "peoples_train|peoples|train|--max-hours $(n 25)|$(n 6000)"
        "peoples_dirty|peoples_dirty|train|--max-hours $(n 12)|$(n 3000)"
        "ami_ihm|ami_ihm|train|--max-hours $(n 12) --min-words 4|$(n 3000)"
        "ami_sdm|ami_sdm|train|--max-hours $(n 12) --min-words 4|$(n 3000)"
        "mls_train|mls|train|--max-hours $(n 20) --max-per-accent 0.1|$(n 4000)"
        "spokenwoz_train|spokenwoz|train|--max-utts $(n 20000)|$(n 20000)"
        "slurp_train|slurp|train|--max-utts $(n 15000)|$(n 8000)"
        "slurp_dev|slurp|devel|--max-utts $(n 400)|0"
    )
    step restore hf v3manifests   # a rerun skips everything an earlier data session already uploaded
    download() {
        for job in "${SETS[@]}"; do
            IFS='|' read -r name preset split extra _ <<< "$job"
            [ -f "$M/${name}_raw.jsonl" ] || [ -f "$M/${name}_nm.jsonl" ] && { echo "$name: have it"; continue; }
            accent=()
            [ "$name" = cv_india ] && accent=(--accent-match "India|South Asia")
            # shellcheck disable=SC2086
            python -m s2s.prep.hf_asr $CFG --preset "$preset" --split "$split" --name "$name" $extra "${accent[@]}" \
                2>&1 | tail -1
        done
    }
    step download download

    hotel() {  # hotel speech with tools, regenerated (only features of the old hotel audio were kept)
        [ -f "$M/hotel5_train_audio.jsonl" ] || [ -f "$M/hotel5_eval_nm.jsonl" ] && { echo "hotel audio: have it"; return; }
        apt-get -qq install -y espeak-ng > /dev/null 2>&1 || true
        pip install -q "kokoro>=0.9" "misaki[en]"
        python -m s2s.prep.hotel_data $CFG --version 3 --train "$(n 10000)" --eval 300 --prefix hotel5
        VOICES="af_heart af_bella af_sarah af_sky af_nova af_river af_jessica af_kore am_adam am_michael \
am_eric am_liam am_onyx am_echo bf_alice bf_isabella bf_lily bm_daniel bm_lewis bm_fable"
        python -m s2s.prep.synth_kokoro $CFG --in $M/hotel5_train_text.jsonl --out $M/hotel5_train_audio.jsonl \
            --voices $VOICES --speed-jitter 0.12 | tail -1
        python -m s2s.prep.synth_kokoro $CFG --in $M/hotel5_eval_text.jsonl --out $M/hotel5_eval_audio.jsonl \
            --voices af_nicole am_puck bf_emma bm_george --speed-jitter 0.1 --seed 1 | tail -1
    }
    step hotel hotel

    upload() {  # manifests + the feature archives not uploaded yet
        mkdir -p "$WORK/up/v3/manifests"
        cp $M/*_nm.jsonl "$WORK/up/v3/manifests/"
        ls $M/*_q4.jsonl > /dev/null 2>&1 && cp $M/*_q4.jsonl "$WORK/up/v3/manifests/"
        local have; have=$(hf v3tars)
        for d in data/features/*_nm; do
            [ -d "$d" ] || continue
            local t="v3/feat-$(basename "$d").tar"
            grep -qx "$t" <<< "$have" || tar -C data -cf "$WORK/up/$t" "features/$(basename "$d")"
        done
        du -sh "$WORK/up"
        hf push "$WORK/up" "" dataset && rm -rf "${WORK:?}/up"
    }
    features() {  # Nemotron 160 ms features; ~45% of clips get echo / band / noise, of those 40% a phone line
        feat() {  # feat <raw manifest> <out name> [extract args]
            local inp=$1 out=$2; shift 2
            [ -f "$M/$out.jsonl" ] || python -m s2s.prep.extract_mimi $CFG --mode latents --encoder "$ENC" \
                --gpus "$NGPU" --in "$M/$inp.jsonl" --out "$M/$out.jsonl" "$@" 2>&1 | tail -1
        }
        for job in "${SETS[@]}"; do
            IFS='|' read -r name _ <<< "$job"
            if [ "$name" = slurp_dev ]; then feat ${name}_raw ${name}_nm; else feat ${name}_raw ${name}_nm --augment-prob 0.45 --phone-prob 0.4; fi
        done
        feat hotel5_train_audio hotel5_train_nm --augment-prob 0.6 --phone-prob 0.5
        feat hotel5_eval_audio hotel5_eval_nm
        # "sorry, I didn't catch that": unintelligible versions of other clips
        [ -f $M/unclear_nm.jsonl ] || head -n "$(n 4000)" $M/cv_train_raw.jsonl > $M/unclear_raw.jsonl
        feat unclear_raw unclear_nm --unclear
    }
    step features features
    step upload_features upload

    answers() {  # the 4B's own answers to the correct transcripts (self-generated targets)
        for job in "${SETS[@]}"; do
            IFS='|' read -r name _ _ _ count <<< "$job"
            [ "$count" -gt 0 ] || continue
            [ -f "$M/${name}_q4.jsonl" ] && continue
            python -m s2s.prep.distill $CFG --gpus "$NGPU" --batch-size 48 --max-new-tokens 80 --max-utts "$count" \
                --in "$M/${name}_nm.jsonl" --out "$M/${name}_q4.jsonl" | tail -1
            upload | tail -1   # each finished set is saved, so a rerun only answers the rest
        done
        for s in hotel5_train hotel5_eval; do  # hotel rows: answered with their own prompt and tools
            [ -f "$M/${s}_q4.jsonl" ] && continue
            python -m s2s.prep.distill $CFG --gpus "$NGPU" --batch-size 16 --max-new-tokens 120 --max-utts 100000 \
                --row-prompts --in "$M/${s}_nm.jsonl" --out "$M/${s}_q4.jsonl" | tail -1
            upload | tail -1
        done
    }
    step answers answers

    step upload_answers upload
    say "PART data DONE"
fi

# ============================================================== shared: data for training
train_data() {
    step download_data bash -c "$(declare -f hf); hf v3data; hf ckpt speech_llm_4b_v2 speech_llm_v3a speech_llm_v3"
    du -sh data/features
}
mix() { [ -f "$1" ] && printf '{path: %s, weight: %s}, ' "$1" "$2"; return 0; }
q4() { [ -f "$M/${1}_q4.jsonl" ] && echo "$M/${1}_q4.jsonl" || echo "$M/${1}_nm.jsonl"; }

# ============================================================== PART align
if [ "$PART" = align ]; then
    train_data
    TRAIN="[$(mix "$(q4 cv_train)" 0.18)$(mix "$(q4 cv_india)" 0.06)$(mix "$(q4 peoples_train)" 0.09)\
$(mix "$(q4 peoples_dirty)" 0.05)$(mix "$(q4 ami_ihm)" 0.06)$(mix "$(q4 ami_sdm)" 0.06)$(mix "$(q4 mls_train)" 0.07)\
$(mix "$(q4 spokenwoz_train)" 0.15)$(mix "$(q4 slurp_train)" 0.12)$(mix "$(q4 hotel5_train)" 0.13)$(mix $M/unclear_nm.jsonl 0.03)]"
    TRAIN=${TRAIN/%, ]/]}
    ( while sleep 1200; do [ -f checkpoints/speech_llm_v3a/adapter.pt ] && hf push checkpoints/speech_llm_v3a checkpoints/speech_llm_v3a; done ) \
        > "$OUT/logs/sync.log" 2>&1 &
    SYNC=$!
    MIN=$(( $(minutes_left) - 25 ))
    step align $TRAIN s2s.train.align_adapter $CFG --set "train_align.train_manifests=$TRAIN" \
        train_align.valid_manifest=$M/slurp_dev_nm.jsonl train_align.max_minutes="$MIN" \
        train_align.resume_from=checkpoints/speech_llm_v3a
    kill $SYNC 2>/dev/null || true
    touch checkpoints/speech_llm_v3a/DONE
    hf push checkpoints/speech_llm_v3a checkpoints/speech_llm_v3a
    say "PART align DONE: checkpoints/speech_llm_v3a"
fi

# ============================================================== PART polish
if [ "$PART" = polish ]; then
    train_data
    [ -f checkpoints/speech_llm_v3a/adapter.pt ] || { say "run PART=align first"; exit 1; }
    INIT=speech_llm_v3a
    [ -f checkpoints/speech_llm_v3/adapter.pt ] && [ ! -f checkpoints/speech_llm_v3/DONE ] && INIT=speech_llm_v3  # resume
    TRAIN="[$(mix "$(q4 hotel5_train)" 0.24)$(mix "$(q4 spokenwoz_train)" 0.16)$(mix "$(q4 slurp_train)" 0.12)\
$(mix "$(q4 cv_train)" 0.14)$(mix "$(q4 cv_india)" 0.05)$(mix "$(q4 peoples_train)" 0.07)$(mix "$(q4 peoples_dirty)" 0.04)\
$(mix "$(q4 ami_ihm)" 0.04)$(mix "$(q4 ami_sdm)" 0.04)$(mix "$(q4 mls_train)" 0.05)$(mix $M/unclear_nm.jsonl 0.05)]"
    TRAIN=${TRAIN/%, ]/]}
    if [ ! -f checkpoints/speech_llm_v3/DONE ]; then
        ( while sleep 1200; do [ -f checkpoints/speech_llm_v3/adapter.pt ] && hf push checkpoints/speech_llm_v3 checkpoints/speech_llm_v3; done ) \
            > "$OUT/logs/sync.log" 2>&1 &
        SYNC=$!
        MIN=$(( $(minutes_left) - 110 ))  # leave time for the test
        step polish $TRAIN s2s.train.speech_llm $CFG --set train_speech_llm.init_from="checkpoints/$INIT" \
            "train_speech_llm.train_manifests=$TRAIN" train_speech_llm.valid_manifest="$(q4 hotel5_eval)" \
            train_speech_llm.max_steps=20000 train_speech_llm.max_minutes="$MIN" train_speech_llm.batch_size=4 \
            train_speech_llm.grad_accum=8 train_speech_llm.num_workers=4 train_speech_llm.log_every=10 \
            train_speech_llm.eval_every=250 train_speech_llm.eval_batches=10 train_speech_llm.save_every=250
        kill $SYNC 2>/dev/null || true
        touch checkpoints/speech_llm_v3/DONE
        hf push checkpoints/speech_llm_v3 checkpoints/speech_llm_v3
    fi

    public_test() {  # the same public test as the Nemotron baseline (fair + strict judge), adapter v3 only
        get() { [ -f "$M/${3}_raw.jsonl" ] || python -m s2s.prep.hf_asr $CFG --preset "$1" --split "$2" \
            --max-utts "$N_TEST" --min-words 3 --max-seconds 20 --name "$3" 2>&1 | tail -1; }
        get slurp test slurp_test; get spokenwoz test spokenwoz_test; get edacc test edacc_test
        f() { [ -f "$M/$3.jsonl" ] || python -m s2s.prep.extract_mimi $CFG --mode latents --encoder "$ENC" \
            --augment-prob "$2" --gpus "$NGPU" --in "$M/${1}_raw.jsonl" --out "$M/$3.jsonl" 2>&1 | tail -1; }
        # the same names as the baseline test, so the noisy clips get exactly the same noise
        f slurp_test 0.0 slurp_pk; f slurp_test 1.0 slurp_noisy_pk; f spokenwoz_test 0.0 spokenwoz_pk
        f edacc_test 0.0 edacc_pk; f edacc_test 1.0 edacc_noisy_pk
        gpu=0
        for s in slurp_pk slurp_noisy_pk spokenwoz_pk edacc_pk edacc_noisy_pk "$(basename "$(q4 hotel5_eval)" .jsonl)"; do
            CUDA_VISIBLE_DEVICES=$gpu python -m s2s.eval.speech_vs_text $CFG --speech-llm-dir checkpoints/speech_llm_v3 \
                --manifest "$M/$s.jsonl" --max "$N_TEST" --name "$s" --out-dir "$OUT" > "$OUT/logs/test_$s.log" 2>&1 &
            gpu=$(( (gpu + 1) % NGPU )); [ "$gpu" = 0 ] && wait
        done
        wait
        grep -h "^\[" "$OUT"/logs/test_*.log || true
    }
    step public_test public_test
    hf push "$OUT" "logs/v3_polish" || true
    say "PART polish DONE: checkpoints/speech_llm_v3; results in $OUT"
fi
