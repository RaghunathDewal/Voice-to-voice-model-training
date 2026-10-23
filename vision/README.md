# Qwen3-VL-4B room inventory fine-tune (Kaggle)

Counting and box-finding ("how many pillows", "locate every towel") for hotel rooms, trained with QLoRA in ms-swift
on Kaggle's free GPU T4 x2. One script, `run_kaggle.sh`, with the phases `setup`, `data`, `train` and `eval`.

| File | What it does |
|---|---|
| `run_kaggle.sh` | the four phases, every setting at the top |
| `prep_homeobjects.py` | HomeObjects-3K **train** split -> count + box rows over its 12 classes (it has no pillow / towel, so they are never asked there) |
| `prep_lvis.py` | Voxel51/LVIS -> 120 hotel classes in 5 areas (bedroom, bathroom, living, kitchen, building); zeros only where the official LVIS file says "verified absent" |
| `mix_data.py` | all rows -> `train.jsonl` + `val.jsonl` (an image's rows stay in one split) |
| `time_budget.py` | ms-swift plugin: fits the cosine learning rate into the session time, saves and stops at the deadline, uploads every checkpoint to Hugging Face |
| `eval_counts.py` | base vs fine-tuned on `eval_frozen.json`, same image size as training; reports present / absent accuracy and the "always 0" floor |

## Before the first run

1. Notebook settings: **Accelerator GPU T4 x2**, **Internet on**.
2. Add-ons -> Secrets -> **`HF_TOKEN`** (a token with WRITE access), switched on for this notebook.
   Checkpoints go to the private repo `ThunderBlade7773/qwen3vl-room-lora` (created on the first save).
3. Eval set: nothing to upload. The script downloads `Yesterday_Qwen3VL_FineTune_2026-10-21.zip` from
   `ThunderBlade7773/s2s-checkpoints` and uses its `eval_data/` (79 HomeObjects **val** images, never in training).
   To use another folder instead: `EVAL_DIR=/kaggle/input/...` in front of the eval command.

## Cells

**Cell 1: token and code**
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!rm -rf /tmp/repo && git clone -q -b vision-qwen3vl https://github.com/RaghunathDewal/Voice-to-voice-model-training.git /tmp/repo
!ls /tmp/repo/vision
```

**Cell 2: install (~5-10 min)**
```python
!bash /tmp/repo/vision/run_kaggle.sh setup
```

**Cell 3: data preparation (~30-40 min: HomeObjects, LVIS annotations, ~30k LVIS images, mix)**
```python
!bash /tmp/repo/vision/run_kaggle.sh data
```

**Cell 4: training (uses the rest of the session minus ~40 min kept for the eval)**
```python
!bash /tmp/repo/vision/run_kaggle.sh train
```

**Cell 5: eval, base vs fine-tuned (~20 min)**
```python
!bash /tmp/repo/vision/run_kaggle.sh eval
```

Then **Save Version -> Save & Run All (Commit)**. Results: `/kaggle/working/eval_results.json` (also uploaded to the HF
repo), checkpoints in `/kaggle/working/out/qwen3vl-room/` and on HF as `last-checkpoint/`.

## Settings you may change (put them in front of the command)

| Setting | Default | Meaning |
|---|---|---|
| `SESSION_HOURS` | 10 | what one session may use in total (Kaggle stops a commit at 12 h) |
| `TOTAL_HOURS` | = this session's training time | `18` = one run over two commits; the second continues and finishes the learning-rate decay |
| `IMAGE_MAX_TOKEN_NUM` | 768 | image size (1 token = 32x32 px); `512` is ~30% faster, less detail |
| `LVIS_IMAGES` | 0 (all, ~30k) | cap on LVIS images |
| `LVIS_GROUPS` | all 5 areas | e.g. `"bedroom bathroom"` |
| `GROUND_SHARE` | 0.2 | share of "locate every X" rows |
| `EPOCHS`, `LR`, `LORA_RANK` | 1, 1e-4, 16 | training |
| `SAVE_STEPS` | 50 | updates between checkpoints (~20-25 min) |
| `EXTRA` | empty | your own rows, `"/kaggle/input/x/rows.jsonl:1.0"` (same `messages` + `images` format) |

Example: `!SESSION_HOURS=11 IMAGE_MAX_TOKEN_NUM=512 bash /tmp/repo/vision/run_kaggle.sh train`

## If the session stops before the end

Run the same cells again (or commit again). `data` rebuilds the same rows (fixed seed), `train` downloads
`last-checkpoint` from Hugging Face and continues from that update with the same learning-rate plan.

## Expected numbers

- ~32k counting rows (one per image) + ~8k box rows = ~40k rows; ~16 rows per update on 2 GPUs.
- One 10 h session reaches roughly half of them; `TOTAL_HOURS=18` and two commits covers all.
- The eval prints, per class: exact-count accuracy, accuracy on images that HAVE the object, on images without it,
  off-by-one, and the "always 0" floor. A model has learned something only if it beats that floor.

## Licences

LVIS annotations CC BY 4.0, COCO images mixed Flickr licences; HomeObjects-3K AGPL-3.0 (Ultralytics). Check both
before any commercial use of the trained adapter.
