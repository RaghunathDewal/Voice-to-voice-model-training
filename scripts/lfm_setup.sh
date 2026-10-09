#!/usr/bin/env bash
# Fresh GPU droplet -> environment for s2s/eval/lfm_audio_probe.py (LFM2.5-Audio-1.5B test).
# Runs directly on the host (no Docker). Works on AMD (ROCm, e.g. MI300X) and NVIDIA (CUDA).
#
#   bash scripts/lfm_setup.sh            # once
#   source ~/lfm-venv/bin/activate       # in every new shell
#
# Override the PyTorch wheel index if needed, e.g. TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4
set -euo pipefail

if [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
  GPU=rocm; TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU=cuda; TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
else
  echo "no AMD or NVIDIA GPU found"; exit 1
fi
echo "== GPU: $GPU, PyTorch wheels from $TORCH_INDEX"

export DEBIAN_FRONTEND=noninteractive
# OS packages are optional (soundfile ships libsndfile); broken third-party apt sources on some images must
# not stop the setup, so failures here only print a note
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1 || echo "(apt update reported errors from other repositories; continuing)"
  apt-get install -y -qq git curl libsndfile1 >/dev/null 2>&1 || echo "(apt install skipped; continuing)"
fi

# liquid-audio needs Python >= 3.12; uv brings its own Python, independent of the OS one
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
# uv's own (managed) Python ships the C headers that torch.compile / Triton need to build GPU kernels;
# the OS Python on many images does not (error: "Python.h: No such file or directory")
uv python install 3.12
uv venv -q -p 3.12 --managed-python --seed --clear ~/lfm-venv
# shellcheck disable=SC1090
source ~/lfm-venv/bin/activate

# torch + torchaudio for this GPU first, then liquid-audio WITHOUT its deps (so pip never swaps in a
# CPU / wrong-GPU torch), then the remaining deps explicitly
echo "== installing PyTorch for $GPU (3-4 GB download, a few minutes)"
uv pip install torch torchaudio --index-url "$TORCH_INDEX"
uv pip install -q --no-deps liquid-audio
uv pip install -q "accelerate>=1.10.1" "datasets>=4.8.4" "einops>=0.8.1" "librosa>=0.11.0" \
  "sentencepiece>=0.2.1" "transformers>=4.55.4" safetensors soundfile jiwer numpy \
  fastapi uvicorn websockets silero-vad scipy pyyaml tqdm   # the live UI server (ws_live --backend lfm)

python - <<'EOF'
import torch, liquid_audio, transformers
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} (hip {torch.version.hip}, cuda {torch.version.cuda}) | GPU visible: {ok}"
      f" | {torch.cuda.get_device_name(0) if ok else '-'} | transformers {transformers.__version__}")
assert ok, "PyTorch cannot see the GPU: check the driver / TORCH_INDEX"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()
print("GPU matmul OK")
EOF

# download the model now (≈ 3 GB) so the test's load time is not a download time
python -c "from huggingface_hub import snapshot_download as s; print(s('LiquidAI/LFM2.5-Audio-1.5B'))"
echo "== ready: source ~/lfm-venv/bin/activate && python s2s/eval/lfm_audio_probe.py --out ~/lfm_results"
