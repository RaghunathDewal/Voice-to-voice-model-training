# Architecture (V0.1, English, strict turn-taking)

```text
USER AUDIO 24 kHz
   │
   ├──► energy VAD ───────────────────────────────┐
   ▼                                              │
Mimi encoder (FROZEN, causal)                     │
continuous pre-quantisation latent, 12.5 Hz, 512-d│
   ▼                                              │
Speech adapter (TRAINED)                          │
LayerNorm → Linear → causal transformer (4L)      │
   ├──► CTC head (chars, ×4 upsampled)  → user transcript (logs/eval only)
   ├──► end-of-turn head (per frame)  ───────────►┤ Endpointer
   ▼                                              │  (min/max silence + EOT prob)
speech embeddings → incremental prefill           │
   ▼                                              ▼
┌────────────────────────────────────────────────────────┐
│ Qwen3 thinker (+ LoRA, merged after stage 3)           │
│ thinking OFF; system + reservation + tools prefilled   │
│ once per session and reused through the KV cache       │
└───────────────┬────────────────────────────────────────┘
                │ token stream
     ┌──────────┴──────────────┐
     ▼                         ▼
<tool_call> JSON          spoken tokens
     │                         │ token embedding + hidden state of the position that
     ▼                         │ produced it (learned mix of layers at 50/75/100 % depth)
schema check → backend         ▼
     │                   Fusion → Temporal talker (12L, d=1024)
     └─ <tool_response> ─►     interleaved [8 text][10 audio][4 text][10 audio]…
        back to thinker        ▼
                         Depth transformer (6L, d=512): codebooks 1..8 per frame
                         acoustic delay 1 (codebooks 2..8 lag codebook 1)
                               ▼
                         Mimi decoder (FROZEN), decoded every 2 frames
                               ▼
                         24 kHz AUDIO OUT
```

## Component decisions

| Component | Choice | Where |
|---|---|---|
| Input features | Mimi latent *before* quantisation (verified: the codes Mimi produces are exactly the quantisation of these latents) | `s2s/models/codec.py` |
| Adapter | causal transformer; RMS of outputs initialised to match Qwen text embeddings; learned speech start/end embeddings | `s2s/models/adapter.py` |
| Aux CTC head | characters, 12.5 Hz frames upsampled ×4 to 50 Hz (character CTC needs more outputs than 12.5 Hz provides) | `adapter.py`, `text.py` |
| End-of-turn head | per-frame logit on the adapter's causal states. Labels: frames of the appended trailing silence = 1, frames more than 4 frames before the end of speech = 0 | `datasets.py` |
| Prompt | rendered with Qwen3's own chat template, split around a speech placeholder; tool calls use Qwen3's `<tool_call>` format | `s2s/models/thinker.py` |
| Stage 2 targets | transcription (30 %) + **behaviour alignment**: the thinker's own text reply to the transcript | `prep/distill.py` |
| Stage 3 targets | spoken hotel requests → exact tool call (or a reply from the reservation context) | `data/hotel.py` |
| Talker input pairing | token *i* is paired with the hidden state at the position that predicted it (tested against step-by-step decoding) | `models/speech_llm.py::talker_features` |
| Talker alignment | fixed interleaving schedule, identical in training and streaming (tested: teacher-forced streaming reproduces training logits) | `models/talker.py` |
| Codebooks | depth transformer; EOA token on codebook 1 ends the utterance; PAD never a target | `models/talker.py` |
| Output decoding | HF Mimi has no decoder conv cache, so each chunk is decoded with 25 frames of left context | `codec.py::StreamingDecoder` |

## Training stages and what is frozen

| Stage | Trains | Frozen | Data |
|---|---|---|---|
| 1 probe (go/no-go) | small CTC model | Mimi | LibriSpeech |
| 2 alignment | adapter, CTC + EOT heads, LoRA | Mimi, Qwen base | LibriSpeech + distilled replies |
| 3 tools | adapter, heads, LoRA (continued) | Mimi, Qwen base | spoken hotel requests (Kokoro TTS) + stage 2 data |
| merge | – | – | LoRA merged into Qwen → `thinker_merged` |
| 4 talker | fusion, temporal, depth | Mimi, merged thinker | single-voice speech + its text (LJSpeech and/or Kokoro) |

Do not change the thinker after stage 4: the talker is trained on its hidden states.

## Development-grade simplifications (not production yet)

* **Input streaming** re-encodes the growing utterance with Mimi every 80 ms (exact because Mimi is causal, but O(n²)). Production: keep Mimi's streaming state (e.g. the `moshi` package).
* **Output decoding** uses a 25-frame left-context window instead of a true streaming decoder state.
* **Tool JSON** is validated against the schema after generation, not grammar-constrained during decoding.
* **No barge-in, no echo cancellation** (strict turn-taking by design).
* **Hotel data is template-generated**: eval sets from the same templates overestimate real accuracy. Record real guest utterances for evaluation.
* **Single GPU** training scripts (no DDP).
