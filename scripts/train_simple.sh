#!/usr/bin/env bash
# =====================================================================================================
#  SIMPLE TRAINING SCRIPT: the two training stages of the speech adapter, every setting written out.
#
#  What gets trained:
#    - the ADAPTER (21M parameters): turns listener features (.npy) into input vectors the 4B can read
#    - a LoRA on Qwen3-4B (33M parameters): helps the frozen 4B read those vectors
#    The 4B itself (4 billion parameters) stays FROZEN. It writes every answer.
#
#  Before running, the data must be prepared (see the file tour in the walkthrough doc):
#    *_nm.jsonl  = clips + transcript + listener features      (s2s/prep/extract_mimi.py)   -> Stage A
#    *_q4.jsonl  = the same rows + the 4B's reply to the text   (s2s/prep/distill.py)        -> Stage B
#
#  Run:   bash scripts/train_simple.sh            (both stages)
#         STAGE=A bash scripts/train_simple.sh    (Stage A only)
#         STAGE=B bash scripts/train_simple.sh    (Stage B only, after Stage A)
#
#  Every value below can be changed from the command line, e.g.:  STEPS_B=3000 LORA_LR=5e-5 bash ...
# =====================================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:-AB}
CONFIG=configs/qwen4b_v3.yaml      # base settings (model names, adapter shape, loss weights); --set lines override it

# ----------------------------------------------------------------------------------------- starting point
# Continue from v4 instead of starting from zero: faster, and the model keeps what it already learned.
START=${START:-checkpoints/speech_llm_v4}          # finished model to start from (adapter.pt + lora/)
OUT_A=${OUT_A:-checkpoints/speech_llm_new_a}       # Stage A result (adapter only)
OUT_B=${OUT_B:-checkpoints/speech_llm_new}         # Stage B result (adapter + LoRA) = the final model

# ----------------------------------------------------------------------------------------- data
# A list of {path, weight}. Weight = share of the clips in each batch drawn from that file.
# KEEP the old sets in the list and ADD yours: training on the new set alone makes the model
# forget accents, noise and hotel replies. Weights are relative (they do not need to add up to 1).
M=data/manifests
TRAIN_A=${TRAIN_A:-"[
  {path: $M/mydata_nm.jsonl,          weight: 0.15},
  {path: $M/hotel5_train_nm.jsonl,    weight: 0.20},
  {path: $M/emilia_en_nm.jsonl,       weight: 0.19},
  {path: $M/spokenwoz_train_nm.jsonl, weight: 0.14},
  {path: $M/cv_accents_nm.jsonl,      weight: 0.14},
  {path: $M/slurp_train_nm.jsonl,     weight: 0.10},
  {path: $M/cv_train_nm.jsonl,        weight: 0.04},
  {path: $M/cv_india_nm.jsonl,        weight: 0.04},
  {path: $M/peoples_train_nm.jsonl,   weight: 0.04}]"}
TRAIN_B=${TRAIN_B:-"[
  {path: $M/mydata_q4.jsonl,          weight: 0.15},
  {path: $M/hotel5_train_q4.jsonl,    weight: 0.20},
  {path: $M/emilia_en_q4.jsonl,       weight: 0.19},
  {path: $M/spokenwoz_train_q4.jsonl, weight: 0.14},
  {path: $M/cv_accents_q4.jsonl,      weight: 0.14},
  {path: $M/slurp_train_q4.jsonl,     weight: 0.10},
  {path: $M/cv_train_q4.jsonl,        weight: 0.04},
  {path: $M/cv_india_q4.jsonl,        weight: 0.04},
  {path: $M/peoples_train_q4.jsonl,   weight: 0.04},
  {path: $M/unclear_nm.jsonl,         weight: 0.03}]"}
VALID_A=${VALID_A:-$M/slurp_dev_nm.jsonl}          # checked during Stage A (word accuracy of the adapter)
VALID_B=${VALID_B:-$M/hotel5_eval_q4.jsonl}        # checked during Stage B (reply loss on held-out hotel calls)

# ----------------------------------------------------------------------------------------- GPUs
# One process per GPU. Effective batch = BATCH x ACCUM x number of GPUs.
NGPU=${NGPU:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}
if [ "$NGPU" -gt 1 ]; then RUN="torchrun --standalone --nproc_per_node=$NGPU -m"; else RUN="python -m"; fi

# ===================================================================================================== STAGE A
# Stage A: train the ADAPTER alone. The 4B is NOT loaded, only its word-vector table, so it is fast.
# The adapter learns to output, at the right frames, vectors that point like the 4B's own word vectors
# for the words in the transcript (word-level CTC). No 4B answers are involved yet.
STEPS_A=${STEPS_A:-20000}
STAGE_A_SETTINGS=(
  "train_align.train_manifests=$TRAIN_A"
  train_align.valid_manifest="$VALID_A"
  train_align.resume_from="$START"        # start from v4's adapter (its output layers already speak to this 4B)
  train_align.output_dir="$OUT_A"
  train_align.batch_size=${BATCH_A:-32}   # clips per GPU per update: large, because the 4B is not in memory
  train_align.lr=${LR_A:-3e-4}            # step size; lower than a fresh start (1e-3) because the adapter is already trained
  train_align.warmup_steps=500            # learning rate rises from 0 to full over the first 500 updates
  train_align.max_steps=$STEPS_A          # number of updates (then a cosine decay to 10% of the lr at the end)
  train_align.max_minutes=0               # 0 = run exactly max_steps (no time limit)
  train_align.weight_decay=0.01           # mild pull of the weights toward zero (against memorising)
  train_align.max_grad_norm=1.0           # cap the size of any single update (guard against blow-ups)
  train_align.tok_weight=1.0              # loss 1: word alignment to the 4B's word vectors (the main one)
  train_align.ctc_weight=0.3              # loss 2: spell the transcript (gives the server a confidence score)
  train_align.eot_weight=0.1              # loss 3: detect the end of the guest's turn
  train_align.eot_early_frames=3          # end-of-turn target starts 3 frames (240 ms) before speech ends
  train_align.eval_every=2000             # check on VALID_A every 2000 updates
  train_align.save_every=2000             # save a checkpoint every 2000 updates
  seed=1234
)

# ===================================================================================================== STAGE B
# Stage B: run the full frozen 4B on the adapter's output and correct the adapter (and the LoRA) until the
# 4B answers from SPEECH the way it answers from the TYPED transcript. The 4B writes every answer.
STEPS_B=${STEPS_B:-2000}
STAGE_B_SETTINGS=(
  "train_speech_llm.train_manifests=$TRAIN_B"
  train_speech_llm.valid_manifest="$VALID_B"
  train_speech_llm.init_from="$OUT_A"          # the Stage A adapter (+ the LoRA copied in below)
  train_speech_llm.output_dir="$OUT_B"
  # --- batch: 4 clips per GPU x 8 accumulated = 32 per GPU per update (x2 GPUs = 64)
  train_speech_llm.batch_size=${BATCH_B:-4}    # what fits in a 16 GB GPU with the 4B loaded; use 16 on a big GPU
  train_speech_llm.grad_accum=${ACCUM_B:-8}    # add up 8 batches before each update (steady direction, no extra memory); use 2 with batch 16
  train_speech_llm.max_steps=$STEPS_B          # updates (v4 had only 483 because of the Kaggle time cap)
  train_speech_llm.max_minutes=0               # 0 = run exactly max_steps
  # --- what is trained, and how fast
  train_speech_llm.train_lora=true             # also train the LoRA on the 4B (its rank r=16 / alpha=32 come from START's lora/)
  train_speech_llm.adapter_lr=${ADAPTER_LR:-5e-5}  # kept low: 2e-4 blew up in an earlier run
  train_speech_llm.lora_lr=${LORA_LR:-1e-4}        # gentle, so the 4B keeps its language and tool-calling skill
  train_speech_llm.warmup_steps=300            # learning rate rises from 0 to full over 300 updates, then cosine decay
  train_speech_llm.weight_decay=0.01
  # --- safety guards
  train_speech_llm.max_grad_norm=1.0           # cap the size of any single update
  train_speech_llm.skip_spike_factor=10        # skip an update whose gradient is >10x the recent median
  # --- losses (what the model is scored on)
  train_speech_llm.kd_weight=1.0               # match the 4B's predictions when it READS the transcript (main signal)
  train_speech_llm.ce_weight=0.5               # the correct reply (the 4B's own answer from *_q4.jsonl)
  train_speech_llm.tok_weight=1.0              # keep Stage A's word alignment
  train_speech_llm.ctc_weight=0.3              # spell the transcript
  train_speech_llm.eot_weight=0.1              # end of turn
  # --- tasks and lengths
  train_speech_llm.transcribe_prob=0.35        # 35% of clips train "write what was said" instead of "answer"
  train_speech_llm.max_target_tokens=160       # replies are cut to 160 tokens for training
  train_speech_llm.eval_every=250              # check on VALID_B every 250 updates
  train_speech_llm.save_every=250              # save a checkpoint every 250 updates
  seed=1233
)

# ----------------------------------------------------------------------------------------- run
for f in $(echo "$TRAIN_A $TRAIN_B" | grep -o "$M/[^,}]*\.jsonl" | sort -u) "$VALID_A" "$VALID_B"; do
  [ -f "$f" ] || { echo "missing data file: $f  (prepare it first, or remove it from the list)"; exit 1; }
done
[ -f "$START/adapter.pt" ] || { echo "missing $START/adapter.pt: download the v4 checkpoint first"; exit 1; }
echo "GPUs: $NGPU   effective batch: Stage A $(( ${BATCH_A:-32} * NGPU )), Stage B $(( ${BATCH_B:-4} * ${ACCUM_B:-8} * NGPU )) clips per update"

if [[ "$STAGE" == *A* ]]; then
  echo "===== Stage A: adapter alone, $STEPS_A updates -> $OUT_A"
  $RUN s2s.train.align_adapter --config "$CONFIG" --set "${STAGE_A_SETTINGS[@]}"
fi

if [[ "$STAGE" == *B* ]]; then
  # Stage A does not touch the 4B, so it saves no LoRA: carry START's LoRA over to continue training it
  [ -d "$OUT_A/lora" ] || cp -r "$START/lora" "$OUT_A/lora"
  echo "===== Stage B: frozen 4B in the loop, adapter + LoRA, $STEPS_B updates -> $OUT_B"
  $RUN s2s.train.speech_llm --config "$CONFIG" --set "${STAGE_B_SETTINGS[@]}"
fi
echo "done: final model in $OUT_B (adapter.pt + lora/ + config.yaml with the exact settings used)"
