#!/usr/bin/env bash
# Live voice demo of the Qwen3-4B system on Kaggle (2x T4), in the browser UI (s2s/cli/static/live.html):
#   Parakeet listener -> speech adapter (speech_llm_4b_v2) -> Qwen3-4B-Instruct -> talker (talker_4b) -> Mimi
# The 4B, adapter, talker and Mimi run on GPU 0; the listener on GPU 1. Prints a public https:// link
# (Cloudflare quick tunnel; browsers need https for the microphone). The cell keeps running while the demo is up.
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints.
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_4b_demo.sh
# Options (put before `bash`):
#   MODEL=v3               the latest adapter v3 checkpoint (Nemotron listener -> adapter v3 + speech LoRA -> 4B), Kokoro voice
#   MODEL=v3h              the hybrid v3 (speech + the listener's transcript), once PART=hybrid has trained it
#   MODEL=0.6b             the earlier compact system instead: Parakeet -> speech_llm_pk4 -> fine-tuned Qwen3-0.6B
#                          (thinker_merged_v3) -> talker_v3 -> Mimi (faster, less expressive)
#   TALKER=best            the talker checkpoint with the lowest eval loss (step 1000) instead of the final one
#   PROMPT=/path/file.txt  your own system prompt (property facts, booking); default: the built-in hotel demo
#   TOOLS=order_product,create_issue   only these tools (default: all built-in hotel tools); TOOLS=none: no tools
#   GRAPHS=1               CUDA-graph reply step (off by default: on a T4 its fixed-size cache made a token
#                          81 ms vs 61 ms normally; the log shows "[graphs] ... one reply token X ms (was Y ms)")
#   EXTRA_SETS="runtime.emit_every_frames=1"   more config overrides (space-separated key=value)
#   TTS=kokoro             speak with Kokoro-82M (own Python environment, on GPU 1) instead of our talker:
#                          fast on a T4 and cannot babble; TTS_VOICE=af_heart|af_bella|am_michael|bf_emma|...
#                          TTS_SPEED=1.0
#   TTS=qwen               speak with Qwen3-TTS (0.6B CustomVoice, its own Python environment, on GPU 1)
#                          instead of our talker; TTS_SPEAKER=Ryan|Aiden|Vivian|Serena|... TTS_LANGUAGE=English
#                          TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (bigger, slower)
#                          Qwen3-TTS is slow on a T4: use it on L4 / L40S / A100-class GPUs
# Applications (e.g. a Gemini-Live-style bridge server) connect to wss://<link>/live: s2s/cli/live_api.py.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints
MODEL=${MODEL:-4b}
if [ "$MODEL" = v3h ]; then  # hybrid: speech vectors + the listener's own transcript (PART=hybrid of kaggle_v3.sh)
    ADAPTER=${ADAPTER:-speech_llm_v3h}; TALKER_DIR=talker_4b; THINKER=""; CFG=configs/qwen4b_v3h.yaml; TTS=${TTS:-kokoro}
elif [ "$MODEL" = v3 ]; then  # the latest adapter v3 upload (Nemotron listener, speech LoRA); Kokoro voice by default
    ADAPTER=${ADAPTER:-speech_llm_v3}; TALKER_DIR=talker_4b; THINKER=""; CFG=configs/qwen4b_v3.yaml; TTS=${TTS:-kokoro}
elif [ "$MODEL" = 0.6b ]; then
    ADAPTER=${ADAPTER:-speech_llm_pk4}; TALKER_DIR=talker_v3; THINKER=thinker_merged_v3; CFG=configs/small.yaml
else
    ADAPTER=${ADAPTER:-speech_llm_4b_v2}; TALKER_DIR=talker_4b; THINKER=""; CFG=configs/qwen4b.yaml
fi
TALKER=${TALKER:-last}
TOOLS=${TOOLS:-all}
PROMPT=${PROMPT:-}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_demo}
: "${HF_TOKEN:?HF_TOKEN is not set (Kaggle: Add-ons > Secrets)}"
export HF_TOKEN ADAPTER TALKER TALKER_DIR THINKER PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/checkpoints"
cd "$REPO_DIR"
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q peft accelerate soundfile librosa huggingface_hub fastapi uvicorn websockets silero-vad
python -c "from transformers import ParakeetForCTC" 2>/dev/null || pip install -q -U "transformers>=4.57"
if [ "$MODEL" = v3 ] || [ "$MODEL" = v3h ]; then  # the Nemotron listener needs a newer transformers
    python -c "from transformers import AutoModelForRNNT" 2>/dev/null || pip install -q -U "transformers>=5.13"
fi
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

say ">>> download the adapter and the talker"
python - <<'EOF'
import os, shutil
from huggingface_hub import snapshot_download
repo, adapter, talker, tdir, thinker = (os.environ[k] for k in ("CKPT_REPO", "ADAPTER", "TALKER", "TALKER_DIR", "THINKER"))
pats = [f"checkpoints/{adapter}/*.pt", f"checkpoints/{adapter}/*.yaml", f"checkpoints/{adapter}/*.json", f"checkpoints/{adapter}/lora/*",
        f"checkpoints/{tdir}/*.pt", f"checkpoints/{tdir}/*.json", f"checkpoints/{tdir}/*.yaml", f"checkpoints/{tdir}/last/*"]
if thinker:
    pats.append(f"checkpoints/{thinker}/*")
snapshot_download(repo, local_dir=".", allow_patterns=pats)
assert os.path.exists(f"checkpoints/{adapter}/adapter.pt"), f"{adapter} not found"
if talker == "last" and os.path.exists(f"checkpoints/{tdir}/last/talker.pt"):  # the final step, not the lowest loss
    shutil.copy(f"checkpoints/{tdir}/last/talker.pt", f"checkpoints/{tdir}/talker.pt")
print(f"adapter {adapter}, {tdir} ({talker})" + (f", thinker {thinker}" if thinker else ""))
EOF

SETS=()   # one --set with every override (a second --set would replace the first)
[ "$NGPU" -gt 1 ] && SETS+=(runtime.encoder_device=cuda:1)
[ -n "$THINKER" ] && SETS+=("thinker.model=checkpoints/$THINKER")
[ "${GRAPHS:-0}" = 1 ] && SETS+=(runtime.cuda_graphs=true)  # reply step from a CUDA graph (checked, falls back)
for kv in ${EXTRA_SETS:-}; do SETS+=("$kv"); done
TTS=${TTS:-talker}
TTS_DEV=$([ "$NGPU" -gt 1 ] && echo cuda:1 || echo cuda:0)
if [ "$TTS" = kokoro ] || [ "$TTS" = qwen ]; then  # a server left from an earlier (stopped) run holds the port
    pkill -f "s2s/cli/(kokoro|qwen)_tts_server.py" 2>/dev/null && sleep 3 || true
fi
if [ "$TTS" = kokoro ]; then
    say ">>> Kokoro-82M TTS in its own environment"
    apt-get -qq install -y espeak-ng > /dev/null 2>&1 || true
    pip install -q uv
    [ -x /tmp/ktts/bin/python ] || uv venv -q -p 3.12 --seed /tmp/ktts
    /tmp/ktts/bin/python -c "import kokoro" 2>/dev/null || {
        uv pip install -p /tmp/ktts/bin/python -q torch --index-url https://download.pytorch.org/whl/cu126
        uv pip install -p /tmp/ktts/bin/python -q "kokoro>=0.9.4" "misaki[en]" soundfile numpy; }
    TTS_PORT=7791 TTS_SERVER=kokoro_tts_server.py
    nohup /tmp/ktts/bin/python s2s/cli/kokoro_tts_server.py --device "$TTS_DEV" --port $TTS_PORT \
        --voice "${TTS_VOICE:-af_heart}" --speed "${TTS_SPEED:-1.0}" > "$WORK/tts.log" 2>&1 &
elif [ "$TTS" = qwen ]; then
    say ">>> Qwen3-TTS in its own environment (qwen-tts pins its own transformers)"
    apt-get -qq install -y sox > /dev/null 2>&1 || true
    pip install -q uv
    [ -x /tmp/qtts/bin/python ] || uv venv -q -p 3.12 --seed /tmp/qtts
    /tmp/qtts/bin/python -c "import qwen_tts" 2>/dev/null || uv pip install -p /tmp/qtts/bin/python -q qwen-tts
    TTS_PORT=7790 TTS_SERVER=qwen_tts_server.py
    nohup /tmp/qtts/bin/python s2s/cli/qwen_tts_server.py --device "$TTS_DEV" --port $TTS_PORT \
        --model "${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}" --speaker "${TTS_SPEAKER:-Ryan}" \
        --language "${TTS_LANGUAGE:-English}" > "$WORK/tts.log" 2>&1 &
fi
if [ "$TTS" = kokoro ] || [ "$TTS" = qwen ]; then
    say "    loading the $TTS server on $TTS_DEV (log: $WORK/tts.log) ..."
    for i in $(seq 1 180); do
        curl -sf http://127.0.0.1:$TTS_PORT/health > /dev/null 2>&1 && break
        if ! pgrep -f "$TTS_SERVER" > /dev/null; then tail -30 "$WORK/tts.log"; say "the TTS server stopped"; exit 1; fi
        sleep 5
    done
    curl -sf http://127.0.0.1:$TTS_PORT/health || { tail -30 "$WORK/tts.log"; say "the TTS server did not start"; exit 1; }
    echo; grep "\[tts\]" "$WORK/tts.log" | tail -3 || true
    SETS+=(runtime.tts_url=http://127.0.0.1:$TTS_PORT/tts)
fi
EXTRA=()
[ ${#SETS[@]} -gt 0 ] && EXTRA+=(--set "${SETS[@]}")
[ -n "$PROMPT" ] && EXTRA+=(--system-prompt-file "$PROMPT")
say ">>> starting the live demo (first start downloads the thinker; wait for the trycloudflare.com link)"
say "    open the https://....trycloudflare.com link, allow the microphone, press Start and just talk"
python -m s2s.cli.ws_live --config "$CFG" --speech-llm-dir "checkpoints/$ADAPTER" \
    --talker-dir "checkpoints/$TALKER_DIR" --tools "$TOOLS" --tunnel "${EXTRA[@]}"
