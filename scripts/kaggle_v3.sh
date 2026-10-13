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
#   PART=data V4=1 (~2-3 h) v4 data: ~105 h of what Gate 1 lacked: accents (Common Voice, more per accent) and
#                        spontaneous conversation (Emilia-YODAS talk shows), features + the 4B's answers -> HF
#   PART=v4      (~11 h) from the final v3: align on the bigger mix (speech_llm_v4a, ALIGN_MIN), then polish
#                        (speech_llm_v4, with v3's speech LoRA), then the Gate 1 test (results logs/v4_gate1)
#   PART=hybrid  (~10 h) hybrid input: the listener's own transcript is added after the speech (Gate 1 was amber:
#                        text wins on accents / phone calls, speech vectors in noise). Transcripts for the training
#                        rows from the stored features, then a polish from the final v3 (speech_llm_v3, LoRA)
#                        -> checkpoints/speech_llm_v3h, then the same Gate 1 test (results logs/v3h_gate1)
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN with WRITE access to
# ThunderBlade7773/s2s-checkpoints and ThunderBlade7773/s2s-data-pk.
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !cd /tmp/repo && PART=data bash scripts/kaggle_v3.sh
# Licences: SpokenWOZ is CC BY-NC 4.0, so this adapter is for non-commercial use.
set -euo pipefail

PART=${PART:?set PART=data, align, polish, hybrid or v4}
HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
BUDGET_MIN=${BUDGET_MIN:-$(case $PART in hybrid) echo 650;; v4) echo 690;; *) echo 500;; esac)}  # minutes this session may use (Kaggle stops at 12 h)
SCALE=${SCALE:-1.0}                    # PART=data: multiply every hour / row count (0.05 for a quick trial)
N_TEST=${N_TEST:-150}
POLISH_BATCH=${POLISH_BATCH:-4}        # PART=polish: batch x accumulation = 32 (a big GPU: 16 x 2, same result)
POLISH_ACCUM=${POLISH_ACCUM:-8}
TEST_MIN=${TEST_MIN:-110}              # PART=polish: minutes kept for the test after training
TEST_PAR=${TEST_PAR:-1}                # PART=polish: test sets run at the same time on each GPU
TEST_ONLY=${TEST_ONLY:-0}              # PART=polish: 1 = only the Gate 1 test, on the latest speech_llm_v3 on Hugging Face
                                       # (no training, nothing uploaded but the results, under logs/v3_gate1_step<N>)
POLISH_LORA=${POLISH_LORA:-$( [ "$PART" = hybrid ] || [ "$PART" = v4 ] && echo 1 || echo 0 )}  # PART=polish: 1 = also train a LoRA on the 4B so it learns to read the
                                       # adapter's vectors (the KD teacher stays the original 4B)
LORA_LR=${LORA_LR:-1e-4}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_v3}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/v3_$PART || echo "$WORK/out_$PART")}
CFG=${CFG:-"--config configs/qwen4b_v3$( [ "$PART" = hybrid ] && echo h).yaml"}
ALIGN_MIN=${ALIGN_MIN:-210}            # PART=v4: minutes of align (fast: no 4B) before the polish
ALIGN_LR=${ALIGN_LR:-3e-4}             # PART=v4: lower than v3's 1e-3, the adapter is already trained
ASR_ROWS=${ASR_ROWS:-8000}              # PART=hybrid: training rows per data set that get the listener's transcript
ENC=nemotron:nvidia/nemotron-speech-streaming-en-0.6b@1
M=data/manifests
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN CKPT_REPO DATA_REPO WORK PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$( { timeout 20 nvidia-smi -L 2>/dev/null || true; } | grep -c "^GPU" || true)  # NVIDIA
[ "${NGPU:-0}" -ge 1 ] || NGPU=$(timeout 120 python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || true)  # AMD
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1
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
                                                                   f"checkpoints/{name}/*.yaml", f"checkpoints/{name}/DONE",
                                                                   f"checkpoints/{name}/lora/*"])
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
    if os.path.exists("data/.v3_features_done"):  # already extracted on this machine (a second part / a rerun)
        print("v3 features already here", flush=True); sys.exit(0)
    tars = sorted(f for f in api.list_repo_files(data, repo_type="dataset") if f.startswith("v3/feat-") and f.endswith(".tar"))
    print(f"{len(tars)} v3 feature archives", flush=True)
    for name in tars:
        t = hf_hub_download(data, name, repo_type="dataset", local_dir=work + "/tmp")
        subprocess.run(f"tar -xf {t} -C data && rm -f {t}", shell=True, check=True)
        print(f"  {name} extracted", flush=True)
    open("data/.v3_features_done", "w").close()
EOF
}

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer peft accelerate soundfile librosa pyarrow huggingface_hub
python -c "from transformers import AutoModelForRNNT" 2>/dev/null || pip install -q -U "transformers>=5.13"
python -c "import torch, transformers; print('transformers', transformers.__version__, '|', torch.cuda.device_count(), 'x', \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
[ "$NGPU" -gt 1 ] && LAUNCH="torchrun --standalone --nproc_per_node=$NGPU -m" || LAUNCH="python -m"

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
    V4_SETS="cv_accents emilia_en"
    if [ "${V4:-0}" = 1 ]; then  # v4: only the two gaps Gate 1 showed (accents, spontaneous talk), ~2-3 h session
        # (voxpopuli_en / yodas_en presets exist too: slower to fetch, less on target; add them here if wanted)
        SETS+=(
            "cv_accents|commonvoice|train|--max-hours $(n 45) --max-per-accent 1 --seed 7 --min-words 3|$(n 5000)"
            "emilia_en|emilia_en|train|--max-hours $(n 60) --max-per-accent 0.6 --min-words 4|$(n 5000)"
        )
    fi
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
        for name in $V4_SETS; do  # the v4 audio is only needed for its features: free the disk
            [ -f "$M/${name}_nm.jsonl" ] && rm -rf "data/hf_asr/$name"
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
    step download_data bash -c "$(declare -f hf); hf v3data; hf ckpt speech_llm_4b_v2 speech_llm_v3a speech_llm_v3 speech_llm_v3h speech_llm_v4a speech_llm_v4"
    du -sh data/features
}
mix() { [ -f "$1" ] && printf '{path: %s, weight: %s}, ' "$1" "$2"; return 0; }
ANS=${ANS:-q4}  # answers to train on: <set>_$ANS.jsonl (q4 = the base 4B's; mi300x_v3.sh: q4h = the hotel-tuned 4B's)
q4() { [ -f "$M/${1}_$ANS.jsonl" ] && echo "$M/${1}_$ANS.jsonl" || echo "$M/${1}_nm.jsonl"; }

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
    ok=0
    for attempt in 1 2 3 4 5; do  # an error mid-run restarts from the last save (every eval) with a new data order
        MIN=$(( $(minutes_left) - 25 ))
        [ "$MIN" -ge 20 ] || break
        if step align_$attempt $LAUNCH s2s.train.align_adapter $CFG --set "train_align.train_manifests=$TRAIN" \
            train_align.valid_manifest=$M/slurp_dev_nm.jsonl train_align.max_minutes="$MIN" \
            train_align.resume_from=checkpoints/speech_llm_v3a seed=$(( 1233 + attempt )); then ok=1; break; fi
        say "align stopped with an error (attempt $attempt, see $OUT/logs/align_$attempt.log); restarting in 60 s"
        pkill -f s2s.train.align_adapter 2>/dev/null || true; sleep 60
    done
    kill $SYNC 2>/dev/null || true
    [ -f checkpoints/speech_llm_v3a/adapter.pt ] || { say "align saved nothing: see $OUT/logs/align_*.log"; exit 1; }
    if [ "$ok" = 1 ]; then touch checkpoints/speech_llm_v3a/DONE; else say "align did not finish cleanly: NOT marked DONE"; fi
    hf push checkpoints/speech_llm_v3a checkpoints/speech_llm_v3a
    say "PART align DONE: checkpoints/speech_llm_v3a"
fi

# ============================================================== PART polish
if [ "$PART" = polish ] || [ "$PART" = hybrid ] || [ "$PART" = v4 ]; then
    train_data
    if [ "$PART" = v4 ]; then  # from the final v3: align on the bigger mix, then polish (with v3's speech LoRA)
        PDIR=speech_llm_v4; START=speech_llm_v4a; RESULTS=logs/v4_gate1
        [ -f checkpoints/speech_llm_v3/DONE ] || { say "the final speech_llm_v3 (DONE) is not on Hugging Face"; exit 1; }
        [ -f $M/emilia_en_nm.jsonl ] || { say "no v4 data on Hugging Face: run PART=data V4=1 first"; exit 1; }
        nmf() { echo "$M/${1}_nm.jsonl"; }  # align needs only the text: every row (not only the answered ones)
        if [ "$TEST_ONLY" != 1 ] && [ ! -f checkpoints/speech_llm_v4a/DONE ]; then
            ATRAIN="[$(mix "$(nmf cv_train)" 0.05)$(mix "$(nmf cv_india)" 0.05)$(mix "$(nmf cv_accents)" 0.20)\
$(mix "$(nmf emilia_en)" 0.24)$(mix "$(nmf peoples_train)" 0.05)\
$(mix "$(nmf peoples_dirty)" 0.03)$(mix "$(nmf ami_ihm)" 0.03)$(mix "$(nmf ami_sdm)" 0.03)$(mix "$(nmf mls_train)" 0.04)\
$(mix "$(nmf spokenwoz_train)" 0.12)$(mix "$(nmf slurp_train)" 0.08)$(mix "$(nmf hotel5_train)" 0.06)$(mix $M/unclear_nm.jsonl 0.02)]"
            ATRAIN=${ATRAIN/%, ]/]}
            ( while sleep 1200; do [ -f checkpoints/speech_llm_v4a/adapter.pt ] && hf push checkpoints/speech_llm_v4a checkpoints/speech_llm_v4a; done ) \
                > "$OUT/logs/sync_align.log" 2>&1 &
            ASYNC=$!
            aok=0
            for attempt in 1 2 3; do  # an error restarts from the last save
                AFROM=speech_llm_v3; [ -f checkpoints/speech_llm_v4a/adapter.pt ] && AFROM=speech_llm_v4a
                AMIN=$(( ALIGN_MIN - (SECONDS - T_START) / 60 + 30 ))  # setup counted in: ALIGN_MIN from the start
                [ "$AMIN" -ge 20 ] || { aok=1; break; }
                if step align4_$attempt $LAUNCH s2s.train.align_adapter $CFG --set "train_align.train_manifests=$ATRAIN" \
                    train_align.valid_manifest=$M/slurp_dev_nm.jsonl train_align.max_minutes="$AMIN" \
                    train_align.output_dir=checkpoints/speech_llm_v4a train_align.resume_from="checkpoints/$AFROM" \
                    train_align.lr="$ALIGN_LR" train_align.warmup_steps=500 seed=$(( 4233 + attempt )); then aok=1; break; fi
                say "align stopped with an error (attempt $attempt); restarting in 60 s"
                pkill -f s2s.train.align_adapter 2>/dev/null || true; sleep 60
            done
            kill $ASYNC 2>/dev/null || true
            [ -f checkpoints/speech_llm_v4a/adapter.pt ] || { say "align saved nothing: see $OUT/logs/align4_*.log"; exit 1; }
            [ "$aok" = 1 ] && touch checkpoints/speech_llm_v4a/DONE
            hf push checkpoints/speech_llm_v4a checkpoints/speech_llm_v4a
        fi
        # the polish starts from the v4 adapter with v3's speech LoRA (align does not touch the 4B)
        [ -d checkpoints/speech_llm_v4a/lora ] || cp -r checkpoints/speech_llm_v3/lora checkpoints/speech_llm_v4a/lora
    elif [ "$PART" = hybrid ]; then  # from the final v3 (adapter + speech LoRA), the listener's transcript added
        PDIR=speech_llm_v3h; START=speech_llm_v3; RESULTS=logs/v3h_gate1
        [ -f checkpoints/speech_llm_v3/DONE ] || { say "the final speech_llm_v3 (DONE) is not on Hugging Face"; exit 1; }
    else
        PDIR=speech_llm_v3; START=speech_llm_v3a; RESULTS=logs/v3_polish
        [ -f checkpoints/speech_llm_v3a/adapter.pt ] || { say "run PART=align first"; exit 1; }
    fi
    INIT=$START
    if [ "${POLISH_FRESH:-0}" = 1 ]; then  # start again from the start checkpoint (an earlier run went wrong)
        say "POLISH_FRESH=1: discarding any earlier $PDIR checkpoint, starting from $START"
        rm -rf "checkpoints/$PDIR"
    fi
    [ -f "checkpoints/$PDIR/adapter.pt" ] && [ ! -f "checkpoints/$PDIR/DONE" ] && INIT=$PDIR  # resume
    if [ "$PART" = hybrid ] && [ "$TEST_ONLY" != 1 ]; then
        hybrid_asr() {  # the listener's transcript ("asr") for a random subset of each training set, both GPUs
            local files=()
            for set in hotel5_train spokenwoz_train slurp_train cv_train cv_india peoples_train peoples_dirty ami_ihm \
                       ami_sdm mls_train hotel5_eval; do
                f=$(q4 $set); [ -f "$f" ] && files+=("$f")
            done
            [ -f $M/unclear_nm.jsonl ] && files+=("$M/unclear_nm.jsonl")
            for g in $(seq 0 $(( NGPU - 1 ))); do
                part=(); for j in "${!files[@]}"; do [ $(( j % NGPU )) = "$g" ] && part+=("${files[$j]}"); done
                [ ${#part[@]} -gt 0 ] && CUDA_VISIBLE_DEVICES=$g python -m s2s.prep.asr_transcripts --in "${part[@]}" \
                    --max-rows "$ASR_ROWS" > "$OUT/logs/asr_$g.log" 2>&1 &
            done
            wait
            cat "$OUT"/logs/asr_*.log | grep -e "rows in" -e "Error" || true
        }
        step hybrid_asr hybrid_asr
        for f in "$(q4 hotel5_train)" "$(q4 hotel5_eval)"; do  # the main sets must have their transcripts
            [ -f "${f%.jsonl}_asr.jsonl" ] || { say "no transcripts for $f: see $OUT/logs/asr_*.log"; exit 1; }
        done
        q4() { local f; f=$( [ -f "$M/${1}_$ANS.jsonl" ] && echo "$M/${1}_$ANS.jsonl" || echo "$M/${1}_nm.jsonl"); echo "${f%.jsonl}_asr.jsonl"; }
        UNCLEAR=$M/unclear_nm_asr.jsonl
    else
        UNCLEAR=$M/unclear_nm.jsonl
    fi
    if [ "$PART" = v4 ]; then
        TRAIN="[$(mix "$(q4 hotel5_train)" 0.20)$(mix "$(q4 spokenwoz_train)" 0.14)$(mix "$(q4 slurp_train)" 0.10)\
$(mix "$(q4 cv_train)" 0.04)$(mix "$(q4 cv_india)" 0.04)$(mix "$(q4 cv_accents)" 0.14)\
$(mix "$(q4 emilia_en)" 0.19)$(mix "$(q4 peoples_train)" 0.04)$(mix "$(q4 peoples_dirty)" 0.02)\
$(mix "$(q4 ami_ihm)" 0.02)$(mix "$(q4 ami_sdm)" 0.02)$(mix "$(q4 mls_train)" 0.02)$(mix $UNCLEAR 0.03)]"
    else
    TRAIN="[$(mix "$(q4 hotel5_train)" 0.24)$(mix "$(q4 spokenwoz_train)" 0.16)$(mix "$(q4 slurp_train)" 0.12)\
$(mix "$(q4 cv_train)" 0.14)$(mix "$(q4 cv_india)" 0.05)$(mix "$(q4 peoples_train)" 0.07)$(mix "$(q4 peoples_dirty)" 0.04)\
$(mix "$(q4 ami_ihm)" 0.04)$(mix "$(q4 ami_sdm)" 0.04)$(mix "$(q4 mls_train)" 0.05)$(mix $UNCLEAR 0.05)]"
    fi
    TRAIN=${TRAIN/%, ]/]}
    if [ "$TEST_ONLY" = 1 ]; then
        [ -f "checkpoints/$PDIR/adapter.pt" ] || { say "no $PDIR on Hugging Face to test"; exit 1; }
        STEP=$(python -c "import json; print(json.load(open('checkpoints/$PDIR/state.json')).get('step', 'x'))" 2>/dev/null || echo x)
        say "TEST_ONLY=1: Gate 1 on $PDIR step $STEP (no training, the checkpoint is not uploaded)"
    elif [ ! -f "checkpoints/$PDIR/DONE" ]; then
        ( while sleep 1200; do [ -f "checkpoints/$PDIR/adapter.pt" ] && hf push "checkpoints/$PDIR" "checkpoints/$PDIR"; done ) \
            > "$OUT/logs/sync.log" 2>&1 &
        SYNC=$!
        ok=0
        for attempt in 1 2 3 4 5; do  # an error mid-run restarts from the last save (every 250 steps)
            MIN=$(( $(minutes_left) - TEST_MIN ))  # leave time for the test
            [ "$MIN" -ge 15 ] || break
            [ "$attempt" -gt 1 ] && [ -f "checkpoints/$PDIR/adapter.pt" ] && INIT=$PDIR
            if step polish_$attempt $LAUNCH s2s.train.speech_llm $CFG --set train_speech_llm.init_from="checkpoints/$INIT" \
                "train_speech_llm.train_manifests=$TRAIN" train_speech_llm.valid_manifest="$(q4 hotel5_eval)" \
                train_speech_llm.output_dir="checkpoints/$PDIR" train_speech_llm.max_steps=20000 train_speech_llm.max_minutes="$MIN" train_speech_llm.batch_size="$POLISH_BATCH" \
                train_speech_llm.grad_accum="$POLISH_ACCUM" train_speech_llm.num_workers=4 train_speech_llm.log_every=10 \
                train_speech_llm.eval_every=250 train_speech_llm.eval_batches=10 train_speech_llm.save_every=250 \
                train_speech_llm.train_lora=$( [ "$POLISH_LORA" = 1 ] && echo true || echo false ) \
                train_speech_llm.lora_lr="$LORA_LR" seed=$(( 1233 + attempt )); then ok=1; break; fi
            say "polish stopped with an error (attempt $attempt, see $OUT/logs/polish_$attempt.log); restarting in 60 s"
            pkill -f s2s.train.speech_llm 2>/dev/null || true; sleep 60
        done
        kill $SYNC 2>/dev/null || true
        [ -f "checkpoints/$PDIR/adapter.pt" ] || { say "polish saved nothing: see $OUT/logs/polish_*.log"; exit 1; }
        [ "$ok" = 1 ] && touch "checkpoints/$PDIR/DONE"
        hf push "checkpoints/$PDIR" "checkpoints/$PDIR"
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
        i=0
        HOTEL_EVAL=$(basename "$(q4 hotel5_eval)" .jsonl)
        for s in slurp_pk slurp_noisy_pk spokenwoz_pk edacc_pk edacc_noisy_pk "$HOTEL_EVAL"; do
            gpu=$(( i % NGPU ))
            CUDA_VISIBLE_DEVICES=$gpu python -m s2s.eval.speech_vs_text $CFG --speech-llm-dir "checkpoints/$PDIR" \
                --manifest "$M/$s.jsonl" --max "$N_TEST" --name "$s" --out-dir "$OUT" > "$OUT/logs/test_$s.log" 2>&1 &
            i=$(( i + 1 )); [ $(( i % (NGPU * TEST_PAR) )) = 0 ] && wait
        done
        wait
        grep -h "^\[" "$OUT"/logs/test_*.log || true
    }
    step public_test public_test
    if [ "$TEST_ONLY" = 1 ]; then hf push "$OUT" "${RESULTS/_polish/_gate1}_step$STEP" || true; else hf push "$OUT" "$RESULTS" || true; fi
    say "PART $PART DONE: checkpoints/$PDIR; results in $OUT"
fi
