#!/usr/bin/env bash
# Which talker_4b checkpoint speaks more clearly? (Kaggle 2x T4, about 40 min)
#   best      checkpoints/talker_4b/talker.pt       (lowest eval loss, early in training)
#   last      checkpoints/talker_4b/last/talker.pt  (the final training step)
# Both speak the same 100 held-out replies (talker_v3_valid) through Qwen3-4B's hidden states; Whisper
# transcribes them: words wrong (lower = clearer), runaway generations, and wavs to listen to.
#
# Kaggle: Accelerator "GPU T4 x2", Internet on, secret HF_TOKEN that can READ ThunderBlade7773/s2s-checkpoints
# and s2s-data-pk (with WRITE access the results are also uploaded to the data repo under results/talker_ab).
#   Cell 1:  !rm -rf /tmp/repo && git clone -b dev https://github.com/RaghunathDewal/Voice-to-voice-model-training /tmp/repo
#   Cell 2:  import os; from kaggle_secrets import UserSecretsClient
#            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Cell 3:  !cd /tmp/repo && bash scripts/kaggle_talker_ab.sh
# Results: /kaggle/working/talker_ab/summary.md, and <best|last>/*.wav to listen to.
set -euo pipefail

HF_USER=${HF_USER:-ThunderBlade7773}
export CKPT_REPO=$HF_USER/s2s-checkpoints DATA_REPO=$HF_USER/s2s-data-pk
N=${N:-100}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/s2s_talker_ab}
OUT=${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working/talker_ab || echo "$WORK/out")}
CFG="--config configs/qwen4b.yaml"
: "${HF_TOKEN:?HF_TOKEN is not set}"
export HF_TOKEN PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
mkdir -p "$WORK"/{data,checkpoints} "$OUT"
cd "$REPO_DIR"
ln -sfn "$WORK/data" data
ln -sfn "$WORK/checkpoints" checkpoints
say() { echo "[$(date +%H:%M:%S)] $*"; }

say ">>> setup"
pip install -q --no-deps -e .
pip install -q jiwer soundfile librosa huggingface_hub accelerate
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

say ">>> download talker_4b (both checkpoints) and the held-out replies"
python - <<'EOF'
import os, shutil
from huggingface_hub import snapshot_download
snapshot_download(os.environ["CKPT_REPO"], local_dir=".", allow_patterns=["checkpoints/talker_4b/*.pt", "checkpoints/talker_4b/*.json",
                  "checkpoints/talker_4b/*.yaml", "checkpoints/talker_4b/last/*"])
snapshot_download(os.environ["DATA_REPO"], repo_type="dataset", local_dir="data", allow_patterns=["manifests/talker_v3_valid.jsonl"])
import json  # only the reply texts are needed (the reference audio codes are not downloaded)
rows = [json.loads(l) for l in open("data/manifests/talker_v3_valid.jsonl")]
with open("data/manifests/talker_ab.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"id": r.get("id"), "text": r["text"]}) + "\n")
print(len(rows), "replies")
# two talker dirs with the same meta (thinker, layers) for s2s.eval.talker
for name, src in (("best", "checkpoints/talker_4b/talker.pt"), ("last", "checkpoints/talker_4b/last/talker.pt")):
    d = f"checkpoints/talker_4b_{name}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(src, f"{d}/talker.pt")
    shutil.copy("checkpoints/talker_4b/meta.json", f"{d}/meta.json")
for f in ("checkpoints/talker_4b/state.json", "checkpoints/talker_4b/last/state.json"):
    print(f, open(f).read().strip() if os.path.exists(f) else "(none)")
EOF

say ">>> both checkpoints speak the same $N replies (one per GPU)"
i=0
for name in best last; do
    CUDA_VISIBLE_DEVICES=$(( i % NGPU )) python -m s2s.eval.talker $CFG --talker-dir checkpoints/talker_4b_$name \
        --manifest data/manifests/talker_ab.jsonl --max "$N" --out-dir "$OUT/$name" > "$OUT/$name.log" 2>&1 &
    i=$(( i + 1 )); [ $(( i % NGPU )) = 0 ] && wait
done
wait
for name in best last; do echo "== $name"; grep -e "talker WER" -e "runaway" -e "mean duration" "$OUT/$name.log" || tail -5 "$OUT/$name.log"; done

python - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
L = ["# talker_4b: best-loss checkpoint vs last step", "", "| checkpoint | words wrong (Whisper) | runaway | mean length |", "|---|---|---|---|"]
for name in ("best", "last"):
    f = os.path.join(out, name, "result.json")
    if not os.path.exists(f):
        L.append(f"| {name} | failed: see {name}.log | | |"); continue
    r = json.load(open(f))
    L.append(f"| {name} | {r['talker_wer'] * 100:.1f}% | {r.get('hit_max_frames', '')} | {r['mean_seconds']:.2f} s |")
L += ["", "Listen to a few wavs in best/ and last/ too: a lower number is clearer, but not always more natural."]
open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
EOF
python - "$OUT" <<'EOF' || true
import os, sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=os.environ["DATA_REPO"], repo_type="dataset", folder_path=sys.argv[1], path_in_repo="results/talker_ab",
                      allow_patterns=["summary.md", "*/result.json"], commit_message="talker_4b: best vs last")
print("uploaded to", os.environ["DATA_REPO"], "results/talker_ab")
EOF
say "DONE: $OUT/summary.md"
