#!/usr/bin/env bash
# Fine-tune LFM2.5-Audio-1.5B on the hotel task and evaluate it with the same test as the base model.
# Run inside the venv from scripts/lfm_setup.sh, detached so an SSH drop does not stop it:
#
#   source ~/lfm-venv/bin/activate
#   nohup bash scripts/lfm_finetune.sh > ~/lfm_ft.log 2>&1 &
#   tail -f ~/lfm_ft.log
#
# Every stage is resumable: re-running skips finished audio clips. Extra args go to every python stage, e.g.
#   bash scripts/lfm_finetune.sh --n-ghb 4000 --epochs 4
# Second round with Qwen-reworded data in a fresh folder:
#   WORK=~/lfm_ft2 STAGES="data rewrite voice build train eval" nohup bash scripts/lfm_finetune.sh > ~/lfm_ft2.log 2>&1 &
set -euo pipefail
export PYTHONUNBUFFERED=1   # show progress in the log immediately (nohup writes to a file)
cd "$(dirname "$0")/.."
WORK=${WORK:-$HOME/lfm_ft}
STAGES=${STAGES:-"data voice build train eval"}
args=("$@")

for s in $STAGES; do
  echo "================ $s ($(date +%H:%M:%S))"
  case $s in
    data)
      if [ -f "$WORK/convs_train.jsonl" ]; then echo "conversations exist, keeping them (delete $WORK to regenerate)";
      else python s2s/train/lfm_finetune.py data --work "$WORK" "${args[@]}"; fi ;;
    rewrite)  # reword with an open Qwen model on vLLM (Docker); only conversations not yet rewritten
      if [ -f "$WORK/convs_train.orig.jsonl" ]; then echo "already rewritten, skipping";
      else bash scripts/lfm_qwen_rewrite.sh "$WORK"; fi ;;
    voice|build|train)
      python s2s/train/lfm_finetune.py "$s" --work "$WORK" "${args[@]}" ;;
    eval)
      python s2s/eval/lfm_audio_probe.py --model-dir "$WORK/model" --placement system \
        --tests placement tts asr s2s text noisy multiturn whisper --out "$WORK/eval" ;;
  esac
done
echo "================ done ($(date +%H:%M:%S)): compare $WORK/eval/summary.md with ~/lfm_results/summary.md"
