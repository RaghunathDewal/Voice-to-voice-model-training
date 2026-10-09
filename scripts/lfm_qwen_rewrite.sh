#!/usr/bin/env bash
# Reword the generated conversations with an open Qwen model on vLLM (s2s/data/llm_rewrite.py), in place.
# Runs vLLM in Docker (AMD: rocm/vllm, NVIDIA: vllm/vllm-openai), so the LFM venv is not touched.
#
#   bash scripts/lfm_qwen_rewrite.sh ~/lfm_ft2                 # rewrites convs_train.jsonl + convs_val.jsonl
#   MODEL=Qwen/Qwen3-14B bash scripts/lfm_qwen_rewrite.sh ~/lfm_ft2 --frac-prompt 0.3
#
# The model is downloaded once into ~/.cache/huggingface (Qwen3-30B-A3B-Instruct-2507: ~60 GB).
set -euo pipefail
WORK=$(realpath "${1:?usage: lfm_qwen_rewrite.sh WORK_DIR [extra args]}"); shift
REPO=$(realpath "$(dirname "$0")/..")
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
mkdir -p "$HOME/.cache/huggingface"
ARGS=(python /repo/s2s/data/llm_rewrite.py --in /work/convs_train.jsonl /work/convs_val.jsonl --model "$MODEL" "$@")
COMMON=(--rm --ipc=host --shm-size 16g -e PYTHONUNBUFFERED=1 -e HF_HOME=/hf
        -v "$HOME/.cache/huggingface:/hf" -v "$REPO:/repo" -v "$WORK:/work")
if [ -e /dev/kfd ]; then
  # same flags that made vLLM start on the MI300X droplet (without them it hangs at start-up)
  docker run "${COMMON[@]}" --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    --device=/dev/kfd --device=/dev/dri --group-add video -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    rocm/vllm:latest "${ARGS[@]}"
else
  docker run "${COMMON[@]}" --gpus all --entrypoint "" vllm/vllm-openai:latest "${ARGS[@]}"
fi
echo "== rewritten in place (originals: $WORK/convs_*.orig.jsonl); examples: $WORK/convs_train.samples.txt"
