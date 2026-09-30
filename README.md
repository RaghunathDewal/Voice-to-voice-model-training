# Voice-to-voice model training (English, turn-based)

A speech-to-speech LLM for a hotel voice agent, trainable on a free Kaggle or Colab GPU:

```
user audio ─► Mimi encoder ─► speech adapter ─► Qwen3 thinker ─► talker ─► Mimi decoder ─► reply audio
                (frozen)        (trained)       (LoRA, tools)    (trained)     (frozen)
```

There is **no ASR text → LLM → TTS pipeline at runtime**. The thinker receives speech embeddings,
and the talker generates Mimi codec tokens straight from the thinker's token embeddings and hidden
states. Design details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

> **Status (read this first).** The code is complete and tested, but **no model in this repo has been
> trained yet**. You will train it by following the steps below. What has been verified:
> * 23 unit/integration tests pass on CPU with tiny random models (`pytest`). They cover training
>   ↔ streaming equivalence of the talker, KV-cache reuse across turns, the tool-call loop,
>   feature pairing, chat-template layout and causal Mimi batching.
> * `scripts/smoke_test.sh` runs **every** script end-to-end (data prep → 4 training stages →
>   5 experiments → CLI chat) on tiny models.
> * The real `kyutai/mimi` weights load, and the latents the adapter consumes are exactly what
>   Mimi's own quantiser uses. Qwen3-0.6B and 1.7B were run through the text tool eval (below).
>
> Whether the trained system is *good* is what experiments 1–5 measure. Don't assume it is until
> they pass.

---

## 0. Hardware and environment

| | Kaggle | Colab (free) |
|---|---|---|
| GPU | T4 ×2 or P100, 16 GB | T4, 16 GB |
| Precision used | fp16 (T4/P100 have no bf16) | fp16 |
| Session limit | ~12 h per session, weekly GPU quota | varies, can disconnect |
| Keep outputs in | `/kaggle/working` (saved with the notebook version, space-limited) | Google Drive |

**Both GPUs are used** (Kaggle T4 ×2):
* data prep (`extract_mimi`, `distill`) starts one worker per visible GPU automatically. Use
  `--gpus 1` to use a single GPU.
* training stages 2–4 run with PyTorch DDP through `torchrun --nproc_per_node=$NGPU`. With 2 GPUs,
  `grad_accum` is halved, so the effective batch (and the results) stay the same and each stage
  finishes in about half the time. Checkpoints, evaluation and samples are written by GPU 0 only.
* the CTC probe, the evals and the runtime use one GPU. They are short.

Set `NGPU` once per session: `NGPU=2` on Kaggle T4 ×2, `NGPU=1` on Colab. In a notebook, use a
Python cell: `NGPU = 2`.

Training stages save checkpoints regularly. If a session dies, restart the stage with
`--set train_...init_from=<last checkpoint dir>`. Feature extraction and distillation skip work
that is already done, so you can simply re-run them.

**Start with `configs/small.yaml`** (Qwen3-0.6B, smaller adapter/talker) to get the whole pipeline
working, then switch to `configs/default.yaml` (Qwen3-1.7B).

## 1. Setup (every new session)

```bash
git clone -b claude/new-session-02rye0 https://github.com/RaghunathDewal/Voice-to-voice-model-training.git
cd Voice-to-voice-model-training
pip install -e ".[dev]"
# needed only for synthetic speech data (step 5/7):
pip install "kokoro>=0.9" && apt-get -qq install -y espeak-ng
```

Or open [`notebooks/kaggle_colab.ipynb`](notebooks/kaggle_colab.ipynb), which runs the same commands.
On Kaggle, turn on *Internet* and a GPU accelerator in the notebook settings.

## 2. Smoke test (≈2–5 min): do this first

```bash
bash scripts/smoke_test.sh outputs/smoke     # tiny random models, synthetic audio
pytest -q                                     # 23 tests
```

It must end with `SMOKE TEST PASSED`. The printed WERs and replies are garbage (the models are random);
the test only proves the code, library versions and GPU work together.

## 3. Full pipeline

All commands accept `--config <yaml>` and `--set key.sub=value` overrides. Paths below assume the
default `paths.data_dir=data`. Use `CFG="--config configs/small.yaml"` for the first run.

### Step 1: input data + **Experiment 1 (go / no-go)**

```bash
python -m s2s.prep.librispeech $CFG --subsets dev-clean test-clean train-clean-100
for s in train dev test; do
  python -m s2s.prep.extract_mimi $CFG --mode latents \
    --in data/manifests/librispeech_${s}_raw.jsonl --out data/manifests/librispeech_${s}.jsonl
done
python -m s2s.train.probe_ctc $CFG           # CTC probe on frozen Mimi latents -> valid WER
```

*Question answered:* can speech be recognised from Mimi latents at all? **Set your pass threshold
before looking at the number.** If WER is poor, the LLM will mishear too, and the fix is the input
front-end (e.g. a streaming FastConformer encoder), not more training downstream.

Disk: train-clean-100 is a 6.3 GB download. Latents take ~2 bytes × 512 × 12.5 frames/s (≈4.6 GB
for 100 h). You can delete `data/librispeech/LibriSpeech/train-clean-100` after extraction.

### Step 2: **Experiment 2**: can the thinker call the tools from *text*?

```bash
python -m s2s.prep.hotel_data $CFG --train 4000 --eval 200
python -m s2s.eval.text_tools $CFG --manifest data/manifests/hotel_eval_text.jsonl
```

Measured on CPU with the default system prompt, **before any training**, on the same 20 template
examples (small sample, so treat these as rough):

| thinker | correct | typical misses |
|---|---|---|
| Qwen3-0.6B | 6/20 (30 %) | mostly doesn't call tools; replies "I can't…" or invents facts |
| Qwen3-1.7B | 16/20 (80 %) | wrong argument values (`"water"` vs `"bottle of water"`, category `other`), one refusal, one wrong fact from context |

So 0.6B is not usable for tools without training, and 1.7B is close but still wrong on argument
values. Stage 3 (LoRA on spoken tool requests) is required, not optional. Re-run this eval with `--lora-dir checkpoints/speech_llm_tools/lora` after
stage 3.

### Step 3: behaviour-alignment targets

```bash
python -m s2s.prep.distill $CFG --in data/manifests/librispeech_train.jsonl \
    --out data/manifests/librispeech_train.jsonl --max-utts 20000
```

The thinker answers each transcript as *text*. Stage 2 then teaches the adapter to make *speech*
produce the same answer, so the model responds to speech instead of just transcribing it.

### Step 4: Stage 2: speech alignment (adapter + CTC/EOT heads + LoRA)

```bash
torchrun --standalone --nproc_per_node=$NGPU -m s2s.train.speech_llm $CFG
```

The script logs `lm`, `ctc` and `eot` losses. Every `eval_every` steps it prints CTC WER, EOT accuracy
and greedy generations for transcribe/respond prompts. Checkpoint: `checkpoints/speech_llm_align/`.

### Step 5: Stage 3: spoken tool requests

```bash
python -m s2s.prep.synth_kokoro $CFG --in data/manifests/hotel_train_text.jsonl \
    --out data/manifests/hotel_train_audio.jsonl --voices af_heart af_bella af_sarah am_adam am_michael
python -m s2s.prep.synth_kokoro $CFG --in data/manifests/hotel_eval_text.jsonl \
    --out data/manifests/hotel_eval_audio.jsonl --voices af_nicole am_puck --seed 1
for s in train eval; do
  python -m s2s.prep.extract_mimi $CFG --mode latents \
    --in data/manifests/hotel_${s}_audio.jsonl --out data/manifests/hotel_${s}.jsonl
done
torchrun --standalone --nproc_per_node=$NGPU -m s2s.train.speech_llm $CFG --set \
  train_speech_llm.init_from=checkpoints/speech_llm_align \
  train_speech_llm.output_dir=checkpoints/speech_llm_tools \
  "train_speech_llm.train_manifests=[{path: data/manifests/hotel_train.jsonl, weight: 0.5}, {path: data/manifests/librispeech_train.jsonl, weight: 0.5}]" \
  train_speech_llm.valid_manifest=data/manifests/hotel_eval.jsonl \
  train_speech_llm.max_steps=4000 train_speech_llm.warmup_steps=200
```

**In a notebook**, don't write this command with `$CFG`/`$NGPU`: the `{path: …}` braces make Jupyter
skip all `$` substitution, so `--nproc_per_node` ends up empty. Build it as a Python string and
run it with `!{cmd}`:
```python
cmd = (f"torchrun --standalone --nproc_per_node={NGPU} -m s2s.train.speech_llm {CFG} --set "
       "train_speech_llm.init_from=checkpoints/speech_llm_align train_speech_llm.output_dir=checkpoints/speech_llm_tools "
       "'train_speech_llm.train_manifests=[{path: data/manifests/hotel_train.jsonl, weight: 0.5}, "
       "{path: data/manifests/librispeech_train.jsonl, weight: 0.5}]' "
       "train_speech_llm.valid_manifest=data/manifests/hotel_eval.jsonl train_speech_llm.max_steps=4000 "
       "train_speech_llm.warmup_steps=200")
!{cmd}
```

The eval voices differ from the training voices, so the eval checks speaker generalisation.
Voice names are from Kokoro-82M's voice list (checked). The first Kokoro run also downloads a
small spaCy English model.

### Step 6: **Experiment 3**: speech input vs text input

```bash
python -m s2s.eval.speech_llm $CFG --speech-llm-dir checkpoints/speech_llm_tools --manifest data/manifests/hotel_eval.jsonl
python -m s2s.eval.speech_llm $CFG --speech-llm-dir checkpoints/speech_llm_tools --manifest data/manifests/librispeech_test.jsonl --max 300
```

Reports tool accuracy from speech vs from the text of the same request, plus the thinker's
transcription WER and the CTC WER. The gap between speech and text accuracy is the cost of the
speech path. **Record 50–100 real spoken requests** and evaluate on those too, since template
data overestimates accuracy.

### Step 7: freeze the thinker, then Stage 4: talker

```bash
python -m s2s.prep.merge_lora $CFG --speech-llm-dir checkpoints/speech_llm_tools --out checkpoints/thinker_merged

# talker data, option A: LJSpeech (one real voice, ~24 h, public domain)
python -m s2s.prep.ljspeech $CFG
# option B (recommended in addition): the thinker's own replies spoken in ONE Kokoro voice
python -m s2s.prep.synth_kokoro $CFG --in data/manifests/librispeech_train.jsonl --text-field response \
    --out data/manifests/talker_kokoro_raw.jsonl --voices af_heart --max-utts 20000
```

Use a **single voice** for the talker; don't mix LJSpeech and Kokoro unless you want a blended
voice. For option B, first make a small validation split, e.g.
`head -100 data/manifests/talker_kokoro_raw.jsonl > data/manifests/talker_valid_raw.jsonl` and
`tail -n +101 data/manifests/talker_kokoro_raw.jsonl > data/manifests/talker_train_raw.jsonl`.

```bash
for s in train valid; do
  python -m s2s.prep.extract_mimi $CFG --mode codes \
    --in data/manifests/talker_${s}_raw.jsonl --out data/manifests/talker_${s}.jsonl
done
torchrun --standalone --nproc_per_node=$NGPU -m s2s.train.talker $CFG
```

Every eval writes `checkpoints/talker/samples/step_N/*.wav`. **Listen to them.** Loss and
per-codebook accuracy are printed too.

### Step 8: **Experiment 4**: talker intelligibility

```bash
python -m s2s.eval.talker $CFG --talker-dir checkpoints/talker --manifest data/manifests/talker_valid.jsonl --max 50
```

Whisper transcribes the generated speech (WER). The same text's reference audio passed through Mimi
gives the best achievable WER for comparison. The eval also counts runaway generations that hit
`talker.max_frames`.

### Step 9: talk to it + **Experiment 5**: latency

```bash
python -m s2s.cli.chat $CFG --wav my_question.wav my_followup.wav            # one file per turn
python -m s2s.cli.chat $CFG --wav my_question.wav --stream                   # 80 ms chunks + endpointing
pip install gradio && python -m s2s.cli.gradio_app $CFG --share              # microphone demo in the browser (record, then submit)
pip install gradio && python -m s2s.cli.gradio_live $CFG --share             # hands-free: just talk, end of turn is detected
python -m s2s.eval.latency $CFG --wav data/tts/hotel_eval_audio/*.wav --max 20
```

The latency report splits **endpoint wait** (silence needed before the turn ends, set by
`runtime.endpoint.*` and the EOT head) from **compute** (end of turn → first reply audio). It also
reports input processing per 80 ms frame, which must stay below 80 ms. Network and playback
buffering are not included.

## 4. Configuration reference

| Key | Meaning |
|---|---|
| `thinker.model` | `Qwen/Qwen3-1.7B` (default) or `Qwen/Qwen3-0.6B` (small) |
| `thinker.dtype` | `auto` → bf16 on Ampere+, fp16 on T4/P100. Use `fp32` if you see NaN losses |
| `thinker.system_prompt` | used in distillation, training and runtime. **Keep it identical** across stages |
| `adapter.*` | adapter size, CTC upsampling |
| `talker.first_text_chunk / text_chunk / audio_chunk` | interleaving schedule (8 / 4 / 10). Changing it needs retraining |
| `talker.acoustic_delay` | codebooks 2..8 lag codebook 1 by this many frames |
| `train_*.batch_size / grad_accum` | reduce batch size (and raise grad_accum) on out-of-memory |
| `runtime.endpoint.*` | min/max silence and EOT threshold for end-of-turn |
| `runtime.decode_context_frames` | left context for chunked Mimi decoding |

## 5. Repository layout

```
configs/            default.yaml (1.7B), small.yaml (0.6B)
s2s/models/         codec.py (Mimi), adapter.py, thinker.py (Qwen3 + prompts), talker.py, speech_llm.py (glue/losses)
s2s/data/           datasets.py (manifests, collators), hotel.py (tools, mock backend, data generator)
s2s/prep/           librispeech, ljspeech, extract_mimi, distill, hotel_data, synth_kokoro, merge_lora, smoke
s2s/train/          probe_ctc (exp 1), speech_llm (stages 2+3), talker (stage 4)
s2s/eval/           text_tools (exp 2), speech_llm (exp 3), talker (exp 4), latency (exp 5)
s2s/runtime/        agent.py (sessions, KV cache, tool loop, streaming talker), endpoint.py
s2s/cli/            chat.py (wav files), gradio_app.py (record + submit), gradio_live.py (hands-free)
scripts/            smoke_test.sh
tests/              pytest suite (CPU)
```

## 6. Troubleshooting

* **`loss nan` with fp16**: lower the learning rate or use `--set thinker.dtype=fp32` (fits for 0.6B).
* **CUDA out of memory**: `--set train_speech_llm.batch_size=4 train_speech_llm.grad_accum=4`
  (same for `train_talker`). Gradient checkpointing is already on for the thinker, and the LM loss
  only computes logits at target positions. GPU memory use has **not** been measured on a T4 yet,
  so the default batch sizes are a starting point.
* **`ImportError: Found an incompatible version of torchao`** (Kaggle ships torchao 0.10 next to
  a newer PEFT): handled automatically since this project doesn't use torchao. If you still see it
  from your own code, `pip uninstall -y torchao`.
* **OpenSLR download slow or failing**: `--mirror https://us.openslr.org/resources/12`.
* **Talker samples silent or endless**: check `acc per codebook` in the logs. Codebook 1 accuracy
  must rise first. Endless outputs hit `talker.max_frames` and are counted by experiment 4.
* **The talker sounds wrong after changing the thinker**: the talker depends on the exact thinker
  (`checkpoints/talker/meta.json` records which). Retrain it.
* **Kokoro import error**: `pip install "kokoro>=0.9" misaki[en]` and `apt-get install espeak-ng`.

## 7. Known limitations (dev build)

See the end of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): input re-encoding is O(n²) per turn,
decoding uses windowed context, tool JSON is validated after generation rather than
grammar-constrained, training is single-GPU, and there is no barge-in.
