#!/usr/bin/env bash
# Baseline before adapter v3, on Kaggle (2x T4, about 2.5-3 h): how good is NVIDIA's streaming recogniser
# nemotron-speech-streaming-en-0.6b on our public test recordings, and how does "its transcript -> the same 4B"
# compare with "our adapter -> the 4B"? Same clips, same 4B, same judges.
#
#   SLURP (+ noisy)   voice-assistant commands, close and distant mics
#   SpokenWOZ         real customer-service phone calls with the dialogue so far (multi-turn)
#   EdAcc (+ noisy)   accented conversational English
# "noisy" = the same clips with the same added echo / phone band / noise as before (s2s/augment.py).
#
# Reported per test set:
#   recogniser WER at 160 / 560 / 1120 ms streaming chunks and full-context ("offline")
#   same answer as typed text: our adapter vs recogniser-160ms vs recogniser-560ms, each with a strict judge
#   (wording-level, as in the earlier test) and a fair judge (same request understood, same action, same details)
# The hotel test is not included: its original audio is not stored (only adapter features), so the
# recogniser cannot hear it.
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints.
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !cd /tmp/repo && bash scripts/kaggle_nemotron_baseline.sh
# Results: /kaggle/working/nemotron_baseline/summary.md (+ <set>.md with 25 examples each, <set>.json, asr/*.jsonl)
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
CKPT_REPO=$HF_USER/s2s-checkpoints
ADAPTER=${ADAPTER:-speech_llm_4b_v2}
ASR_MODEL=${ASR_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}
N=${N:-150}                       # recordings per test set
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_baseline}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/nemotron_baseline || echo "$WORK/out")}
CFG="--config configs/qwen4b.yaml"
E=parakeet:nvidia/parakeet-ctc-0.6b
M=data/manifests
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN CKPT_REPO ADAPTER PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
mkdir -p "$WORK"/{data,checkpoints} "$OUT"/{logs,asr}
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
mkdir -p data/manifests
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer peft accelerate soundfile librosa pyarrow huggingface_hub
# the Nemotron streaming model needs transformers >= 5.13 (our code runs on 5.x too)
python -c "from transformers import AutoModelForRNNT" 2>/dev/null || pip install -q -U "transformers>=5.13"
python -c "import torch, transformers; assert torch.cuda.device_count(), 'no GPU'; \
print('transformers', transformers.__version__, '|', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"

say ">>> download our adapter ($ADAPTER)"
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
name = os.environ["ADAPTER"]
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=[f"checkpoints/{name}/*.pt", f"checkpoints/{name}/*.yaml",
                                                                         f"checkpoints/{name}/*.json"])
assert os.path.exists(f"checkpoints/{name}/adapter.pt"), f"checkpoints/{name} not found"
print("ok")
EOF

say ">>> public recordings ($N each) + our adapter's input features (clean and noisy)"
get() {  # get <preset> <split> <name>
    [ -f "$M/${3}_raw.jsonl" ] || python -m s2s.prep.hf_asr $CFG --preset "$1" --split "$2" \
        --max-utts "$N" --min-words 3 --max-seconds 20 --name "$3" 2>&1 | tail -1
}
get slurp test slurp_test
get spokenwoz test spokenwoz_test
get edacc test edacc_test
features() {  # features <raw name> <augment prob> <feature manifest name>
    [ -f "$M/$3.jsonl" ] || python -m s2s.prep.extract_mimi $CFG --mode latents --encoder $E --augment-prob "$2" \
        --gpus "$NGPU" --in "$M/${1}_raw.jsonl" --out "$M/$3.jsonl" 2>&1 | tail -1
}
features slurp_test 0.0 slurp_pk
features slurp_test 1.0 slurp_noisy_pk
features spokenwoz_test 0.0 spokenwoz_pk
features edacc_test 0.0 edacc_pk
features edacc_test 1.0 edacc_noisy_pk

# test sets: <feature manifest>:<raw manifest>:<noise name or ->
SETS="slurp_pk:slurp_test:- slurp_noisy_pk:slurp_test:slurp_noisy_pk spokenwoz_pk:spokenwoz_test:- \
edacc_pk:edacc_test:- edacc_noisy_pk:edacc_test:edacc_noisy_pk"

parallel() {  # parallel <function> : runs it for every set, one set per GPU at a time
    local fn=$1 gpu=0
    for s in $SETS; do
        "$fn" "$gpu" "$s" &
        gpu=$(( (gpu + 1) % (NGPU > 0 ? NGPU : 1) ))
        [ "$gpu" = 0 ] && wait
    done
    wait
}

say ">>> Nemotron streaming transcripts (160 / 560 / 1120 ms + full context)"
asr_set() {  # asr_set <gpu> <feat:raw:noise>
    IFS=: read -r feat raw noise <<< "$2"
    local extra=(); [ "$noise" != "-" ] && extra=(--noisy-name "$noise")
    CUDA_VISIBLE_DEVICES=$1 python -m s2s.eval.asr_nemotron $CFG --model "$ASR_MODEL" --manifest "$M/${raw}_raw.jsonl" \
        --max "$N" "${extra[@]}" --out "$OUT/asr/$feat.jsonl" > "$OUT/logs/asr_$feat.log" 2>&1 \
        && grep "^\[" "$OUT/logs/asr_$feat.log" || { echo "FAILED asr $feat"; tail -5 "$OUT/logs/asr_$feat.log"; }
}
parallel asr_set

say ">>> same 4B: answer from typed text vs our adapter vs Nemotron transcript"
s2t_set() {  # s2t_set <gpu> <feat:raw:noise>
    IFS=: read -r feat _ _ <<< "$2"
    [ -f "$OUT/asr/$feat.jsonl" ] || { echo "skip $feat (no transcripts)"; return 0; }
    CUDA_VISIBLE_DEVICES=$1 python -m s2s.eval.speech_vs_text $CFG --speech-llm-dir "checkpoints/$ADAPTER" \
        --manifest "$M/$feat.jsonl" --max "$N" --name "$feat" --out-dir "$OUT" \
        --asr-json "$OUT/asr/$feat.jsonl" --asr-keys 160,560 > "$OUT/logs/s2t_$feat.log" 2>&1 \
        && grep "^\[" "$OUT/logs/s2t_$feat.log" || { echo "FAILED s2t $feat"; tail -5 "$OUT/logs/s2t_$feat.log"; }
}
parallel s2t_set

python - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
p = lambda v: f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "-"  # noqa: E731
lines = ["## Same answer as typed text (strict / fair judge)", "",
         "| test set | n | our adapter | Nemotron 160 ms + 4B | Nemotron 560 ms + 4B |", "|---|---|---|---|---|"]
wers = ["", "## Hearing: word error rate (numbers spelled out)", "",
        "| test set | our adapter (4B repeats) | our adapter (CTC head) | Nemotron 160 ms | 560 ms | 1120 ms | full context |",
        "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    s = json.load(open(f)).get("summary")
    if not s:
        continue
    lines.append(f"| {s['name']} | {s['n']} | {p(s['same_answer'])} / {p(s['same_answer_fair'])} | "
                 f"{p(s.get('asr_160_same'))} / {p(s.get('asr_160_same_fair'))} | "
                 f"{p(s.get('asr_560_same'))} / {p(s.get('asr_560_same_fair'))} |")
    wers.append(f"| {s['name']} | {p(s['heard_wer'])} | {p(s['ctc_wer'])} | {p(s.get('asr_160_wer'))} | "
                f"{p(s.get('asr_560_wer'))} | {p(s.get('asr_1120_wer'))} | {p(s.get('asr_offline_wer'))} |")
text = "\n".join(lines + wers)
open(os.path.join(out, "summary.md"), "w").write(f"# Nemotron baseline vs adapter {os.environ.get('ADAPTER', '')}\n\n{text}\n")
print(text)
EOF
say "DONE: $OUT/summary.md"
