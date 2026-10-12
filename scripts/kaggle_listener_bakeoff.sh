#!/usr/bin/env bash
# Listener bake-off on Kaggle (2x T4, about 45-60 min): Parakeet (the v2 adapter's listener) vs Nemotron
# streaming (v3's), on the same public recordings - accuracy (each model's own transcript) and speed (how soon
# after the guest's last word the listener output is ready, compute per second of audio). s2s/eval/listener_bakeoff.py
#
#   SLURP (+ noisy)      voice-assistant commands          SpokenWOZ   customer-service phone calls
#   EdAcc (+ noisy)      accented conversational English   Indian English: Common Voice (India) + Svarah
# "noisy" = the same echo / phone band / noise as the earlier tests (s2s/augment.py).
#
# Kaggle: Accelerator "GPU T4 x2", Internet on. No Hugging Face token needed (all public); with a WRITE token
# (secret HF_TOKEN) the results are also uploaded to the data repo under results/listeners.
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2 (optional):  import os; from kaggle_secrets import UserSecretsClient
#                       os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_listener_bakeoff.sh
# Options (before `bash`): N=150 (recordings per set).
# Results: /kaggle/working/listeners/summary.md (+ <set>.json with every transcript and timing).
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export DATA_REPO=$HF_USER/s2s-data-pk
N=${N:-150}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_listeners}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/listeners || echo "$WORK/out")}
CFG="--config configs/qwen4b_v3.yaml"
M=data/manifests
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
mkdir -p "$WORK"/{data,checkpoints} "$OUT/logs"
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
mkdir -p $M
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer soundfile librosa pyarrow huggingface_hub
python -c "from transformers import AutoModelForRNNT, ParakeetForCTC" 2>/dev/null || pip install -q -U "transformers>=5.13"
python -c "import torch, transformers; print('transformers', transformers.__version__, '|', torch.cuda.device_count(), 'x', \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

say ">>> test recordings ($N per set)"
get() {  # get <preset> <split> <name> [extra args]: a failed download only skips that set
    local preset=$1 split=$2 name=$3; shift 3
    [ -f "$M/${name}_raw.jsonl" ] && return 0
    python -m s2s.prep.hf_asr $CFG --preset "$preset" --split "$split" --max-utts "$N" --min-words 3 --max-seconds 20 \
        --name "$name" "$@" 2>&1 | tail -1 || say "could not get $name: skipped"
}
get slurp test slurp_test
get spokenwoz test spokenwoz_test
get edacc test edacc_test
get commonvoice test cv_india_test --accent-match "India|South Asia"
get svarah test svarah_test

# <name>|<raw manifest>|<noise name (the earlier tests' noise) or empty>
RUNS=("slurp|slurp_test|" "slurp_noisy|slurp_test|slurp_noisy_pk" "spokenwoz|spokenwoz_test|"
      "edacc|edacc_test|" "edacc_noisy|edacc_test|edacc_noisy_pk" "cv_india|cv_india_test|" "svarah|svarah_test|")
say ">>> bake-off (both listeners on the same GPU, clip by clip)"
i=0
for run in "${RUNS[@]}"; do
    IFS='|' read -r name raw noise <<< "$run"
    [ -s "$M/${raw}_raw.jsonl" ] || { say "skip $name (no recordings)"; continue; }
    extra=(); [ -n "$noise" ] && extra=(--noisy-name "$noise")
    CUDA_VISIBLE_DEVICES=$(( i % NGPU )) python -m s2s.eval.listener_bakeoff $CFG --manifest "$M/${raw}_raw.jsonl" \
        --max "$N" --name "$name" "${extra[@]}" --out-dir "$OUT" > "$OUT/logs/$name.log" 2>&1 &
    i=$(( i + 1 )); [ $(( i % NGPU )) = 0 ] && wait
done
wait
grep -h "^\[" "$OUT"/logs/*.log || true

python - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
rows = [json.load(open(f))["summary"] for f in sorted(glob.glob(os.path.join(out, "*.json")))]
if not rows:
    sys.exit("no results: see the logs in " + os.path.join(out, "logs"))
order = ["slurp", "slurp_noisy", "spokenwoz", "edacc", "edacc_noisy", "cv_india", "svarah"]
rows.sort(key=lambda r: order.index(r["set"]) if r["set"] in order else 99)
L = [f"# Listener bake-off: Parakeet vs Nemotron streaming ({rows[0]['gpu']})", "",
     "## Accuracy: words wrong in each listener's own transcript (lower is better)", "",
     "| set | n | Parakeet (whole utterance) | Nemotron streaming 160 ms | Nemotron full context |", "|---|---|---|---|---|"]
for r in rows:
    off = f"{r['wer_nemotron_offline'] * 100:.1f}%" if "wer_nemotron_offline" in r else ""
    L.append(f"| {r['set']} | {r['n']} | {r['wer_parakeet'] * 100:.1f}% | {r['wer_nemotron_160ms'] * 100:.1f}% | {off} |")
L += ["", "## Speed: listener output ready after the guest's last word (median / 90th percentile)", "",
      "| set | clip length (median) | Parakeet | Nemotron streaming | compute per second of audio: Parakeet / Nemotron | Nemotron per 160 ms chunk (median / 95th) |",
      "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['set']} | {r['seconds_median']:.1f} s | {r['parakeet_ready_ms_p50']:.0f} / {r['parakeet_ready_ms_p90']:.0f} ms | "
             f"{r['nemotron_ready_ms_p50']:.0f} / {r['nemotron_ready_ms_p90']:.0f} ms | {r['parakeet_rtf']:.3f} / {r['nemotron_rtf']:.3f} | "
             f"{r['nemotron_chunk_ms_p50']:.0f} / {r['nemotron_chunk_ms_p95']:.0f} ms |")
L += ["", "Parakeet can only start once the guest has stopped (it reads the whole utterance), so its wait is its whole "
      "encode time. Nemotron runs while the guest speaks (fed in 80 ms pieces as live), so its wait is the last "
      "160 ms chunk's look-ahead plus that chunk's compute. Both exclude the end-of-turn detection, which is the same "
      "for both. Two sets run at a time (one per GPU), so timings include some shared-CPU noise."]
open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
EOF

if [ -n "${HF_TOKEN:-}" ]; then
    python - "$OUT" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=os.environ["DATA_REPO"], repo_type="dataset", folder_path=sys.argv[1], path_in_repo="results/listeners",
                      allow_patterns=["summary.md", "*.json"], commit_message="listener bake-off: Parakeet vs Nemotron")
print("uploaded to", os.environ["DATA_REPO"], "results/listeners")
EOF
fi
say "DONE: $OUT/summary.md"
