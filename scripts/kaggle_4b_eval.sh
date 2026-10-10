#!/usr/bin/env bash
# Test the Qwen3-4B speech adapter on public recordings it never saw, on Kaggle (2x T4, about 2.5 h).
# No human answer labels are needed: for every recording the 4B answers the TYPED transcript and the
# SPOKEN audio, and the 4B itself judges whether the two answers are the same.
#
#   SLURP       real people giving voice-assistant commands, close-talk and far-field mics (CC BY 4.0)
#   SpokenWOZ   real customer-service PHONE calls, with the dialogue so far: tests multi-turn (CC BY-NC 4.0:
#               evaluation only)
#   EdAcc       accented conversational English, many countries (CC BY-SA 4.0)
#   hotel       our synthetic hotel test with tools (clean and noisy), from the private data repo
# SLURP and EdAcc are tested twice: as recorded, and with added room echo, phone-band filtering and
# background noise (s2s/augment.py), to see how much noise costs (Parakeet does NOT remove noise).
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints
# (+ s2s-data-pk for the hotel test). Then:
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !cd /tmp/repo && bash scripts/kaggle_4b_eval.sh
# Results: /kaggle/working/speech_vs_text/summary.md (+ one .md with examples and one .json per test set).
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
DATA_REPO=$HF_USER/s2s-data-pk
ADAPTER=${ADAPTER:-speech_llm_4b_v2}
N=${N:-200}                       # recordings per test set
HOTEL=${HOTEL:-1}                 # 0: skip the hotel test (no access to the data repo needed)
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_eval}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/speech_vs_text || echo "$WORK/out")}
CFG="--config configs/qwen4b.yaml"
E=parakeet:nvidia/parakeet-ctc-0.6b
M=data/manifests
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN CKPT_REPO DATA_REPO WORK ADAPTER PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
mkdir -p "$WORK"/{data,checkpoints,tmp} "$OUT/logs"
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
mkdir -p data/manifests
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer peft accelerate soundfile librosa pyarrow huggingface_hub
python -c "from transformers import ParakeetForCTC" 2>/dev/null || pip install -q -U "transformers>=4.57"
python -c "import torch; assert torch.cuda.device_count(), 'no GPU'; print(torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"

say ">>> download the adapter (and the hotel test)"
python - "$ADAPTER" "$HOTEL" <<'EOF'
import os, subprocess, sys
from huggingface_hub import hf_hub_download, snapshot_download
name, hotel = sys.argv[1], sys.argv[2] == "1"
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt", f"checkpoints/{name}/*.yaml",
                                                                         f"checkpoints/{name}/*.json"])
assert os.path.exists(f"checkpoints/{name}/adapter.pt"), f"checkpoints/{name} not found in {os.environ['CKPT_REPO']}"
if hotel:
    data = os.environ["DATA_REPO"]
    snapshot_download(data, repo_type="dataset", local_dir="data",
                      allow_patterns=["manifests/hotel4_eval_clean_pk.jsonl", "manifests/hotel4_eval_noisy_pk.jsonl"])
    t = hf_hub_download(data, "pk-015.tar", repo_type="dataset", local_dir=os.environ["WORK"] + "/tmp")
    subprocess.run(f"tar -xf {t} -C data --wildcards 'features/hotel4_eval_*' && rm -f {t}", shell=True, check=True)
print("ok")
EOF

say ">>> public recordings ($N each) + Parakeet features (clean and with added noise)"
get() {  # get <preset> <split> <name> [extra hf_asr args]
    local preset=$1 split=$2 name=$3; shift 3
    [ -f "$M/${name}_raw.jsonl" ] || python -m s2s.prep.hf_asr $CFG --preset "$preset" --split "$split" \
        --max-utts "$N" --min-words 3 --max-seconds 20 --name "$name" "$@" 2>&1 | tail -2
}
get slurp test slurp_test
get spokenwoz test spokenwoz_test
get edacc test edacc_test
features() {  # features <name> <augment prob> <out>
    python -m s2s.prep.extract_mimi $CFG --mode latents --encoder $E --augment-prob "$2" --gpus "$NGPU" \
        --in "$M/${1}_raw.jsonl" --out "$M/$3.jsonl" 2>&1 | tail -1
}
features slurp_test 0.0 slurp_pk
features slurp_test 1.0 slurp_noisy_pk
features spokenwoz_test 0.0 spokenwoz_pk
features edacc_test 0.0 edacc_pk
features edacc_test 1.0 edacc_noisy_pk

say ">>> speech vs text (two test sets at a time, one per GPU)"
SETS="slurp_pk slurp_noisy_pk spokenwoz_pk edacc_pk edacc_noisy_pk"
[ "$HOTEL" = 1 ] && SETS="$SETS hotel4_eval_clean_pk hotel4_eval_noisy_pk"
run_set() {  # run_set <gpu> <manifest name>
    CUDA_VISIBLE_DEVICES=$1 python -m s2s.eval.speech_vs_text $CFG --speech-llm-dir "checkpoints/$ADAPTER" \
        --manifest "$M/$2.jsonl" --max "$N" --name "$2" --out-dir "$OUT" > "$OUT/logs/$2.log" 2>&1 \
        && tail -1 "$OUT/logs/$2.log" || { echo "FAILED: $2 (see $OUT/logs/$2.log)"; tail -5 "$OUT/logs/$2.log"; }
}
gpu=0
for s in $SETS; do
    [ -f "$M/$s.jsonl" ] || { say "skip $s (no manifest)"; continue; }
    run_set "$gpu" "$s" &
    gpu=$(( (gpu + 1) % (NGPU > 0 ? NGPU : 1) ))
    [ "$gpu" = 0 ] && wait     # one set per GPU at a time
done
wait

python - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    s = json.load(open(f)).get("summary")
    if s:
        rows.append(s)
lines = ["| test set | recordings | same answer as typed | identical answer | 4B heard (WER) | adapter CTC (WER) | multi-turn rows |",
         "|---|---|---|---|---|---|---|"]
for s in rows:
    lines.append(f"| {s['name']} | {s['n']} | {s['same_answer'] * 100:.1f}% | {s['identical_answer'] * 100:.1f}% | "
                 f"{s['heard_wer'] * 100:.1f}% | {s['ctc_wer'] * 100:.1f}% | {s['multi_turn_rows']} |")
text = "\n".join(lines)
open(os.path.join(out, "summary.md"), "w").write(f"# Speech vs text, adapter {os.environ.get('ADAPTER', '')}\n\n{text}\n")
print(text)
EOF
say "DONE: $OUT/summary.md (examples per test set in $OUT/<name>.md)"
