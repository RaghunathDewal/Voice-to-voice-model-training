#!/usr/bin/env bash
# Compare the 0.6B system, the 4B system and the full-duplex model on the SAME new spoken hotel requests
# (s2s/eval/compare_systems.py). Kaggle 2x T4, about 1.5-2 h for the defaults.
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints
# (with WRITE access the results are also uploaded to the data repo under results/compare).
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_compare.sh
# Options (before `bash`): N=80 (requests; x2 with the noisy copies), SYSTEMS=0.6b,4b,duplex,
#   PARAPHRASE=1 (re-word the requests with Qwen3-4B), NOISY=1 (also echo / noise / phone-line copies).
# Results: /kaggle/working/compare/summary.md (+ summary.json with every answer, testset.jsonl, wavs/).
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints DATA_REPO=$HF_USER/s2s-data-pk
N=${N:-80}
SYSTEMS=${SYSTEMS:-0.6b,4b,duplex}
PARAPHRASE=${PARAPHRASE:-1}
NOISY=${NOISY:-1}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_compare}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/compare || echo "$WORK/compare")}
: "${HF_TOKEN:?HF_TOKEN is not set}"
export HF_TOKEN PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/checkpoints" "$OUT"
cd "$REPO_DIR"
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
apt-get -qq install -y espeak-ng > /dev/null 2>&1 || true
pip install -q --no-deps -e .
pip install -q jiwer peft accelerate soundfile librosa huggingface_hub
python -c "from transformers import AutoModelForRNNT, ParakeetForCTC" 2>/dev/null || pip install -q -U "transformers>=5.13"
# Kokoro (it speaks the test requests) supports Python <= 3.12; newer Kaggle images run 3.13
if python -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3, 12) else 1)"; then
    pip install -q "kokoro>=0.9" "misaki[en]"
    TTS_PY=python
else
    say "$(python -V) is too new for Kokoro: a small Python 3.12 environment speaks the test requests"
    pip install -q uv
    uv venv -q -p 3.12 --seed /tmp/tts312
    uv pip install -p /tmp/tts312/bin/python -q torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install -p /tmp/tts312/bin/python -q "kokoro>=0.9" "misaki[en]" soundfile scipy numpy pyyaml jiwer \
        huggingface_hub
    uv pip install -p /tmp/tts312/bin/python -q --no-deps -e .
    TTS_PY=/tmp/tts312/bin/python
fi
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

say ">>> download the three systems"
python - <<'EOF'
import os, shutil
from huggingface_hub import snapshot_download
names = ["speech_llm_pk4", "thinker_merged_v3", "talker_v3", "speech_llm_4b_v2", "talker_4b", "duplex"]
pats = [p for n in names for p in (f"checkpoints/{n}/*", f"checkpoints/{n}/last/*")]
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=pats)
for t in ("talker_v3", "talker_4b"):  # the final training step, not the lowest-loss early checkpoint
    if os.path.exists(f"checkpoints/{t}/last/talker.pt"):
        shutil.copy(f"checkpoints/{t}/last/talker.pt", f"checkpoints/{t}/talker.pt")
for n in names:
    print(n, "ok" if os.path.isdir(f"checkpoints/{n}") else "MISSING")
EOF

ARGS=(--n "$N" --systems "$SYSTEMS" --out "$OUT")
[ "$PARAPHRASE" = 1 ] && ARGS+=(--paraphrase)
[ "$NOISY" = 1 ] && ARGS+=(--noisy)
say ">>> test requests (${ARGS[*]})"
python -m s2s.eval.compare_systems "${ARGS[@]}" --stage texts 2>&1 | grep -v -e "Warning" -e "warnings.warn" || true
say ">>> speaking them (Kokoro)"
$TTS_PY -m s2s.eval.compare_systems "${ARGS[@]}" --stage audio --device cpu 2>&1 | grep -v -e "Warning" -e "warnings.warn" || true
[ -f "$OUT/testset.jsonl" ] || { say "the test audio was not created: see the lines above"; exit 1; }
say ">>> compare"
python -m s2s.eval.compare_systems "${ARGS[@]}" 2>&1 | grep --line-buffered -v -e "Warning" -e "warnings.warn" | tee "$OUT/compare.log"

say ">>> upload the results (needs a write token; skipped otherwise)"
python - "$OUT" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
out = sys.argv[1]
HfApi().upload_folder(repo_id=os.environ["DATA_REPO"], repo_type="dataset", folder_path=out, path_in_repo="results/compare",
                      allow_patterns=["summary.*", "testset.jsonl", "compare.log"], commit_message="compare 0.6b / 4b / duplex")
print("uploaded to", os.environ["DATA_REPO"], "results/compare")
EOF
cat "$OUT/summary.md"
