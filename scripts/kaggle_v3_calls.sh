#!/usr/bin/env bash
# Our voice agent with the LATEST adapter v3 checkpoint (Nemotron listener -> adapter v3 + speech LoRA -> Qwen3-4B
# -> Kokoro voice) on the same four spoken hotel calls as the PersonaPlex test, scored by the same report.
# Kaggle 2x T4, ~25-35 min. The polish run uploads speech_llm_v3 every 20 min, so this takes whatever step is there.
#   1. the hotel calls: reused from the PersonaPlex test if it ran in this session, else made (Kokoro guests)
#   2. each guest turn is fed as live (80 ms pieces), the real end-of-turn detection decides when to answer,
#      the reply is spoken by Kokoro and placed in the call at (detected end of turn + compute time)
#   3. report: reply delay, talking over, right facts, tool calls; stereo files (left guest, right agent)
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints.
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_v3_calls.sh
# Options (before `bash`): ADAPTER=speech_llm_v3 (or speech_llm_4b_v2 with CFG=configs/qwen4b.yaml for the
# earlier adapter), TTS_VOICE=af_heart, NO_TOOLS=1, GRAPHS=1 (CUDA-graph reply step; slower on a T4).
# Results: /kaggle/working/v3_calls/summary.md (+ the PersonaPlex one in /kaggle/working/personaplex if it ran)
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
ADAPTER=${ADAPTER:-speech_llm_v3}
CFG=${CFG:-configs/qwen4b_v3.yaml}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_v3_calls}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/v3_calls || echo "$WORK/out")}
PP_CALLS=${PP_CALLS:-/kaggle/working/personaplex/calls/calls.json}
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN ADAPTER PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/checkpoints" "$OUT/run"
cd "$REPO_DIR"
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q peft accelerate soundfile librosa huggingface_hub silero-vad
python -c "from transformers import AutoModelForRNNT, ParakeetForCTC" 2>/dev/null || pip install -q -U "transformers>=5.13"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

say ">>> download $ADAPTER (latest upload) and the talker files the runtime loads"
python - <<'EOF'
import json, os
from huggingface_hub import snapshot_download
repo, a = os.environ["CKPT_REPO"], os.environ["ADAPTER"]
snapshot_download(repo, local_dir=".", allow_patterns=[f"checkpoints/{a}/*.pt", f"checkpoints/{a}/*.json", f"checkpoints/{a}/*.yaml",
                  f"checkpoints/{a}/lora/*", "checkpoints/talker_4b/*.pt", "checkpoints/talker_4b/*.json"])
assert os.path.exists(f"checkpoints/{a}/adapter.pt"), f"{a} not found in {repo}"
st = f"checkpoints/{a}/state.json"
step = json.load(open(st)).get("step") if os.path.exists(st) else "?"
print(f"{a}: step {step}, speech LoRA: {'yes' if os.path.isdir(f'checkpoints/{a}/lora') else 'no'}")
open("/tmp/v3_label", "w").write(f"{a} step {step}")
EOF

say ">>> Kokoro voice server"
apt-get -qq install -y espeak-ng > /dev/null 2>&1 || true
pip install -q uv
if ! /tmp/ktts/bin/python -c "import kokoro" 2>/dev/null; then
    [ -x /tmp/ktts/bin/python ] || uv venv -q -p 3.12 --seed /tmp/ktts
    uv pip install -p /tmp/ktts/bin/python -q torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install -p /tmp/ktts/bin/python -q "kokoro>=0.9.4" "misaki[en]" soundfile numpy
fi
DEV=$([ "$NGPU" -gt 1 ] && echo cuda:1 || echo cuda:0)
pkill -f "s2s/cli/(kokoro|qwen)_tts_server.py" 2>/dev/null && sleep 3 || true
nohup /tmp/ktts/bin/python s2s/cli/kokoro_tts_server.py --device "$DEV" --port 7791 --voice "${TTS_VOICE:-af_heart}" \
    > "$WORK/tts.log" 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:7791/health > /dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:7791/health || { tail -30 "$WORK/tts.log"; say "the Kokoro server did not start"; exit 1; }
echo

if [ -f "$PP_CALLS" ]; then
    CALLS=$PP_CALLS; say ">>> reusing the PersonaPlex test's calls ($CALLS)"
else
    say ">>> the hotel calls (Kokoro guests)"
    /tmp/ktts/bin/python s2s/eval/personaplex_calls.py --out-dir "$OUT/calls"
    CALLS=$OUT/calls/calls.json
fi

say ">>> $(cat /tmp/v3_label) on the calls (4B on GPU 0, listener + voice on GPU 1)"
SETS=(runtime.tts_url=http://127.0.0.1:7791/tts)
[ "$NGPU" -gt 1 ] && SETS+=(runtime.encoder_device=cuda:1)
[ "${GRAPHS:-0}" = 1 ] && SETS+=(runtime.cuda_graphs=true)
EXTRA=(); [ "${NO_TOOLS:-0}" = 1 ] && EXTRA+=(--no-tools)
python -m s2s.eval.v3_calls --config "$CFG" --calls "$CALLS" --out-dir "$OUT/run" --speech-llm-dir "checkpoints/$ADAPTER" \
    --talker-dir checkpoints/talker_4b --label "$(cat /tmp/v3_label)" "${EXTRA[@]}" --set "${SETS[@]}" 2>&1 \
    | grep -v -e "Warning" -e "warn(" | tee "$OUT/run.log"
pkill -f s2s/cli/kokoro_tts_server.py 2>/dev/null || true

say ">>> report"
python -m s2s.eval.personaplex_report --calls "$CALLS" --run-dir "$OUT/run" --out "$OUT/summary.md" \
    --title "Our agent ($(cat /tmp/v3_label), Qwen3-4B, Kokoro voice) on hotel calls"
[ -f /kaggle/working/personaplex/summary.md ] && say "PersonaPlex on the same calls: /kaggle/working/personaplex/summary.md"
say "DONE: $OUT/summary.md, listen to $OUT/run/*_mix.wav (left: guest, right: agent)"
