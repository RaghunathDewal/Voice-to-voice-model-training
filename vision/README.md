# Qwen3-VL-4B room inventory (Kaggle)

Hotel room check after a stay: the **before** inventory (`pillow 4, towel 2, lamp 2, kettle 1 ...`) and the **after**
photo(s) go in; **missing / changed / new** items come out. Qwen3-VL-4B with a LoRA trained in ms-swift on Kaggle's free
GPU T4 x2. One script, `run_kaggle.sh`: `setup`, `smoke`, `data`, `train`, `eval`.

## How it works in production (`infer_room.py`)

| Call | Question to the model | Model | Catches |
|---|---|---|---|
| 1 | "Expected items from the previous inspection: pillow, towel, lamp, kettle. Count how many of each are in this image now (0 if it is not there)." | **fine-tuned LoRA** | missing items (0) and changed counts |
| 2 | "... List every OTHER object you can see that is not one of those, with how many." | **base model** (LoRA switched off) | new / left-behind items |

The model only counts; the comparison with the before inventory is code you control (tolerance per item, several
photos per room: max count over the photos). A count is per photo: an item out of frame counts 0, so take the after
photos like the before ones.
```
python infer_room.py --before before.json --images after_1.jpg after_2.jpg --adapter ThunderBlade7773/qwen3vl-room-lora-v3
```

## Runs

| Run | Data | Starts from | HF repo | Frozen rooms (exact) | Missing caught (main 4) |
|---|---|---|---|---|---|
| base | - | - | - | 73.1 | 76.3 |
| v1 | old prompt, zeros from any class (towel ~75% zeros), cushion merged into pillow | zero | `qwen3vl-room-lora` | **75.3** | **85.9** |
| v2 | production prompt, ~13% zeros, only "scene-matched" ones | v1 | `qwen3vl-room-lora-v2` | 50.3 | 53.8: claimed pillows in 31 of 40 rooms without |
| **v3** | production prompt, **~1/3 zeros for every class**, any item can be missing in any indoor photo, data gate | v2 | `qwen3vl-room-lora-v3` | this run | this run |

(v1 / v2 numbers: the v2 run's eval, same frozen questions as v3. v3's eval runs v1 and v2 again on its eval set.)

## Rules learned (do / do not)

1. **Only real zeros.** A class that is not labelled in a photo is unknown, not 0 (v0 answered 0 for everything).
   Zeros come from LVIS "verified absent" or HomeObjects (all 12 classes labelled on every image).
2. **Every class answered 0 in about 1/3 of the questions about it.** Too many zeros -> "towel = 0" (v1); too few ->
   "on the list = there" (v2). `common.allocate_zeros` gives each class its share; a class that cannot get enough
   real zeros (LVIS rarely verifies "no pillow") is asked less often where it is present, down to 25%.
3. **A missing item can be any item in any indoor photo** (a towel or a kettle asked in a living room: hotel
   checklists do that). Never an item whose near-twin is visible (pillow / cushion, lamp / lampshade ...).
4. **Same prompt in training, eval and production** (`common.py`), same image size (`IMAGE_MAX_TOKEN_NUM`), same
   inference code (`vl_infer.py`).
5. **Compare models only on the same eval set**, in the same run. The 79 frozen rooms (real rooms, checked inventory)
   decide; HomeObjects val / LVIS held-out look like the training data and flatter the fine-tune.
6. **"New items" are not trained** on partial labels (it would teach leaving unlabelled things out). Fine-tuning
   narrows open listing (v2 found 33% of other objects vs base 48%), so call 2 uses the base model.
7. **fp16 on T4, never bf16** (emulated, 12x slower). No 4-bit on T4 (slower, and the 4B fits in fp16).
8. **Gate before GPU hours:** `check_data.py` stops the run if a main class has <15% or >50% zeros, a question is not
   the production prompt, or an eval image is in training. `smoke` runs 3 real updates before the data phase.

## Files

| File | What it does |
|---|---|
| `run_kaggle.sh` | the phases, every setting at the top |
| `common.py` | prompts, the zero policy (`allocate_zeros`), near-twins, label matching for new items |
| `prep_lvis.py` | Voxel51/LVIS -> 120 hotel classes in 5 areas; boxes -> counts; LVIS verified absences -> zeros |
| `prep_homeobjects.py` | HomeObjects-3K train (training) and val (eval) -> counting + box rows over its 12 classes |
| `mix_data.py` | all rows -> `train.jsonl` / `val.jsonl` / held-out `test.jsonl` (an image's rows stay in one split) |
| `build_eval_set.py` | eval set: 79 frozen rooms + HomeObjects val + held-out LVIS |
| `check_data.py` | the data gate (rule 8) |
| `smoke_check.py` | reads the smoke test: updates finished, speed, memory |
| `time_budget.py` | ms-swift plugin: fits the cosine schedule into the session, stops at the deadline, uploads every checkpoint |
| `vl_infer.py` | the one inference path (eval and production): Qwen3-VL and Qwen3.5, batched, LoRA on/off, counting / box-finding / new items |
| `eval_counts.py` | runs a model on the eval set (optionally also counting by boxes); report: summary + acc / present / missing caught / false alarms / new items |
| `demo_app.py` | browser demo (Gradio + Cloudflare link): single vs hybrid, boxes on the photo, new items |
| `infer_room.py` | production: before inventory + after photos -> alerts + new items |

## Before the first run

1. Notebook settings: **Accelerator GPU T4 x2**, **Internet on**.
2. Add-ons -> Secrets -> **`HF_TOKEN`** (WRITE access), switched on for this notebook.
3. Nothing to upload: the eval set is built in the `data` phase (79 frozen rooms from
   `Yesterday_Qwen3VL_FineTune_2026-10-21.zip` in `ThunderBlade7773/s2s-checkpoints`, HomeObjects val, held-out LVIS).

## Cells (Save Version -> Save & Run All)

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!rm -rf /tmp/repo && git clone -q -b vision-qwen3vl https://github.com/RaghunathDewal/Voice-to-voice-model-training.git /tmp/repo
!ls /tmp/repo/vision
```
```python
!bash /tmp/repo/vision/run_kaggle.sh setup      # ~2 min
```
```python
!bash /tmp/repo/vision/run_kaggle.sh smoke      # ~4 min: 3 real updates from the starting LoRA, must say SMOKE TEST OK
```
```python
!bash /tmp/repo/vision/run_kaggle.sh data       # ~35 min, ends with DATA CHECK OK
```
```python
!bash /tmp/repo/vision/run_kaggle.sh train      # the rest of the session minus the eval reserve (~7.5 h)
```
```python
!bash /tmp/repo/vision/run_kaggle.sh eval       # ~1.6-2 h: v3 and v1 on the two GPUs, then v2
```
If the session dies: run the same cells again; `train` continues from `last-checkpoint` on HF.

## Demo in the browser (latest model, v3)

Same cell 1 (token + code), then:
```python
!bash /tmp/repo/vision/run_kaggle.sh setup
```
```python
!bash /tmp/repo/vision/run_kaggle.sh demo
```
Wait for `OPEN THIS: https://....trycloudflare.com` (model load ~1-2 min) and open it on any device.
1. **Step 1, BEFORE:** upload the before photo, type the items to count (`pillow, towel, lamp, chair`), click
   **1. Count BEFORE**. No after photo needed: you get the counts (single and hybrid), the boxes, and the inventory
   is filled in (`pillow 3, towel 0, ...`, edit it if a count is wrong), plus the other objects on the photo.
2. **Step 2, AFTER:** upload the after photo, click **2. Check AFTER**: ok / MISSING / fewer / more per item, and the
   objects that are new since the before photo.

You can skip step 1 by typing the inventory yourself (`pillow 4, towel 2`). **single**, **hybrid** or **both**
applies to both steps. Your 79 frozen rooms are clickable examples. The link lives while the cell runs.
`DEMO_LORA=ThunderBlade7773/qwen3vl-room-lora-v2` for another run.

## Bench: compare models and ways of counting, no training (~2-2.5 h)

Same cells 1-2 (token + code, `setup`), then:
```python
!bash /tmp/repo/vision/run_kaggle.sh bench
```
It builds only the eval set (~6 min: HomeObjects val, the 79 frozen rooms, the 200 held-out LVIS images) and runs,
spread over the two GPUs:

| Job | Model | How it counts |
|---|---|---|
| `v3` | Qwen3-VL-4B + v3 LoRA | the counting answer; plus `v3+boxes` ("locate every pillow", count the boxes) and `v3+hybrid` (0 if the counting answer says 0, else the boxes) for pillow / towel / lamp / chair |
| `qwen3.5-4b` | Qwen3.5-4B, no training (thinking off) | same three ways |
| `qwen3vl-4b-base` | Qwen3-VL-4B, no training | the counting answer (the reference) |

The log ends with a SUMMARY table (one line per model and way of counting); `bench_results.json` goes to
`qwen3vl-room-lora-v3` on HF. Other candidates: `BENCH="name|model id|LoRA repo or empty|main or none;..."`, e.g.
`qwen3.5-9b|Qwen/Qwen3.5-9B||main` (fp16 does not fit one T4: add `QUANT=1` for 4-bit). Qwen3.5 runs its new
attention layers without the fast kernels by default (slower, safe); `BENCH_FLA=1` tries them.

## Settings (put them in front of the command, e.g. `!EVAL_BASE=1 bash ... eval`)

| Setting | Default | Meaning |
|---|---|---|
| `RUN` | v3 | run name: output folder and HF repo `qwen3vl-room-lora-<RUN>` |
| `INIT_FROM` | `qwen3vl-room-lora-v2` | LoRA to start from (its `last-checkpoint`); empty = from zero |
| `COMPARE` | v1 and v2 repos | earlier runs evaluated next to this one |
| `EVAL_BASE` | 0 | 1 = also the base model (+~0.8 h) |
| `EVAL_BATCH`, `EVAL_JOB_H` | 4, 0.8 | eval batch; hours per model, used to reserve time for the eval |
| `EPOCHS` | 2 | at most; the time budget fits what the session allows |
| `LR` | 5e-5 continuing, 1e-4 from zero | learning rate |
| `SESSION_HOURS` | 10 | what one session may use in total (Kaggle stops a commit at 12 h) |
| `TOTAL_HOURS` | this session | `16` = one run over two commits |
| `IMAGE_MAX_TOKEN_NUM` | 768 | image size (1 token = 32x32 px) |
| `DTYPE` / `QUANT` | fp16 on T4 / 0 | precision / 4-bit base |
| `LVIS_IMAGES`, `LVIS_GROUPS`, `GROUND_SHARE` | all, all, 0.2 | data |
| `EXTRA` | empty | your own rows `"/kaggle/input/x/rows.jsonl:1.0"` (same `messages` + `images` format) |

## The eval report

Per model: `acc` exact count | `pres` exact when the item is there | `miss` the item is gone and the model says 0
(alert caught) | `f.al` the item is there but the model says 0 (false alarm, lower is better) | `off1` within +-1;
for all sources, each source, and pillow / towel / lamp / chair. Then **new items** on the 79 frozen rooms (full
inventory): share of the real other objects found, share of what is listed that is real (after the cleaning
production does: count > 0, not an expected item, not a wall / floor / ceiling).

## Next (beyond public data)

The biggest gain left is **your own hotel photos**: a few thousand, labelled with a full inventory by a teacher model
(as the frozen set was), added with `EXTRA`. That also allows training call 2 (new items) properly, and pairs of
before / after photos would allow the visual-difference task.

## Licences

LVIS annotations CC BY 4.0, COCO images mixed Flickr licences; HomeObjects-3K AGPL-3.0 (Ultralytics). Check both
before commercial use of the trained adapter.
