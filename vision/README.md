# Qwen3-VL-4B room inventory fine-tune (Kaggle)

Counting and box-finding ("how many pillows", "locate every towel") for hotel rooms, trained with QLoRA in ms-swift
on Kaggle's free GPU T4 x2. One script, `run_kaggle.sh`, with the phases `setup`, `data`, `train` and `eval`.

| File | What it does |
|---|---|
| `run_kaggle.sh` | the four phases, every setting at the top |
| `prep_homeobjects.py` | HomeObjects-3K **train** split -> count + box rows over its 12 classes (it has no pillow / towel, so they are never asked there) |
| `prep_lvis.py` | Voxel51/LVIS -> 120 hotel classes in 5 areas (bedroom, bathroom, living, kitchen, building); zeros only where the official LVIS file says "verified absent" |
| `mix_data.py` | all rows -> `train.jsonl` + `val.jsonl` (an image's rows stay in one split) |
| `smoke_check.py` | reads the smoke-test log: did updates finish, speed, memory |
| `common.py` | the production counting prompt and the rule for zeros (scene-matched, capped), shared by the preps and the eval |
| `time_budget.py` | ms-swift plugin: fits the cosine learning rate into the session time, saves and stops at the deadline, uploads every checkpoint to Hugging Face |
| `build_eval_set.py` | the mixed eval set (below), built in the `data` phase |
| `eval_counts.py` | base (GPU 0) and fine-tuned (GPU 1) at the same time on the eval set, same image size as training; reports present / absent accuracy and the "always 0" floor per source and per class |

## Runs

| Run | Data | Starts from | Saved to (HF) | Result |
|---|---|---|---|---|
| v1 | "Count the following objects" prompt, LVIS cushion merged into pillow, ~3 zero answers per present one for some classes | zero | `qwen3vl-room-lora` | 65 -> 79% exact overall; pillow over-counted (sofa cushions), towel said 0 when there |
| **v2** (default) | production prompt ("Expected items from the previous inspection: ... 0 if it is not there"), at most one missing item per question and only one that fits the scene (common.py), zeros per class <= 0.4 x present, cushion its own class | v1's LoRA (`--adapters`), lr 5e-5 | `qwen3vl-room-lora-v2` | eval: v2 vs v1 vs base |

`RUN=v3 INIT_FROM=ThunderBlade7773/qwen3vl-room-lora-v2` for a next round; `INIT_FROM=` (empty) to start from zero.

## Before the first run

1. Notebook settings: **Accelerator GPU T4 x2**, **Internet on**.
2. Add-ons -> Secrets -> **`HF_TOKEN`** (a token with WRITE access), switched on for this notebook.
   Checkpoints go to the private repo `ThunderBlade7773/qwen3vl-room-lora` (created on the first save).
3. Eval set: nothing to upload. The `data` phase builds it from images that are never trained on:

   | Source | Images | Classes asked |
   |---|---|---|
   | `frozen`: your 79 checked rooms, from `eval_data/` in `Yesterday_Qwen3VL_FineTune_2026-10-21.zip` (downloaded from `ThunderBlade7773/s2s-checkpoints`) | 79 | pillow, towel, lamp, chair |
   | `homeobjects_val`: HomeObjects **val** split (training uses train only) | 250 | its 12 classes, real zeros |
   | `lvis`: LVIS images held out by `mix_data.py` (2% of images, never in train or val) | 250 | the hotel classes, zeros only where LVIS verified them |

   `EVAL_PER_SOURCE=400` for more; `EVAL_DIR=/kaggle/input/...` to use another frozen folder.

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

**Cell 2b: smoke test (~10 min): 3 real training updates on 64 rows, BEFORE the long data step**
```python
!bash /tmp/repo/vision/run_kaggle.sh smoke
```
Must end with `SMOKE TEST OK: ... s per update, GPU memory ... GiB`. If it says FAILED, stop and fix first.

**Cell 3: data preparation (~30-40 min: HomeObjects, LVIS annotations, ~30k LVIS images, mix)**
```python
!bash /tmp/repo/vision/run_kaggle.sh data
```

**Cell 4: training (~7 h: the rest of the session minus ~2 h kept for the eval)**
```python
!bash /tmp/repo/vision/run_kaggle.sh train
```

**Cell 5: eval, v2 vs v1 vs base (~2 h for ~480 images: v2 then base on GPU 0, v1 on GPU 1)**
```python
!bash /tmp/repo/vision/run_kaggle.sh eval
```

Then **Save Version -> Save & Run All (Commit)**. Results: `/kaggle/working/eval_results.json` (also uploaded to the HF
repo), checkpoints in `/kaggle/working/out/qwen3vl-room/` and on HF as `last-checkpoint/`.

## Settings you may change (put them in front of the command)

| Setting | Default | Meaning |
|---|---|---|
| `RUN` | v2 | run name: output folder and HF repo `qwen3vl-room-lora-<RUN>` |
| `INIT_FROM` | `ThunderBlade7773/qwen3vl-room-lora` (v1) | LoRA to start from (its `last-checkpoint`); empty = from zero |
| `LR` | 5e-5 when continuing, 1e-4 from zero | learning rate |
| `SESSION_HOURS` | 10 | what one session may use in total (Kaggle stops a commit at 12 h) |
| `EVAL_BASE` | 1 | 0 = skip the base model in the eval (saves ~1 h, more training) |
| `EVAL_PER_SOURCE` | 200 | eval images per source (the 79 frozen rooms always all) |
| `TOTAL_HOURS` | = this session's training time | `14` = one run over two commits |
| `IMAGE_MAX_TOKEN_NUM` | 768 | image size (1 token = 32x32 px); `512` is ~30% faster |
| `DTYPE` | float16 on T4, bfloat16 on A100/L4/H100 | precision (bf16 is emulated and ~10x slower on a T4) |
| `QUANT` | 0 | 1 = 4-bit base (only if out of memory) |
| `LVIS_IMAGES`, `LVIS_GROUPS` | all | which LVIS images / areas |
| `GROUND_SHARE` | 0.2 | share of "locate every X" rows |
| `SAVE_STEPS` | 50 | updates between checkpoints |
| `EXTRA` | empty | your own rows, `"/kaggle/input/x/rows.jsonl:1.0"` (same `messages` + `images` format) |

## What the eval reports

Per model, for all sources, each source, and pillow / towel / lamp / chair:
`acc` exact count | `pres` exact when the item is there | `miss` the item is gone and the model says 0 (the alert is
caught) | `f.al` the item is there but the model says 0 (a false alarm, lower is better) | `off1` within +-1.
The model only counts; comparing with the "before" inventory and deciding what to alert on is done in code.

## If the session stops before the end

Run the same cells again (or commit again). `data` rebuilds the same rows (fixed seed), `train` downloads
`last-checkpoint` from Hugging Face and continues from that update with the same learning-rate plan.

## Expected numbers

- ~32k counting rows (one per image) + ~8k box rows = ~40k rows; ~16 rows per update on 2 GPUs.
- One 10 h session reaches roughly half of them; `TOTAL_HOURS=18` and two commits covers all.
- The eval prints base and fine-tuned side by side, for all sources together, each source on its own, and
  pillow / towel / lamp / chair: exact-count accuracy, accuracy on images that HAVE the object, on images without
  it, off-by-one, and the "always 0" floor. A model has learned something only if it beats that floor. Every
  class's numbers are in `eval_results.json`.

## Licences

LVIS annotations CC BY 4.0, COCO images mixed Flickr licences; HomeObjects-3K AGPL-3.0 (Ultralytics). Check both
before any commercial use of the trained adapter.
