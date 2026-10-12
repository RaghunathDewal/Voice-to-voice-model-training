# Full-duplex experiment

A small test of whether a 0.6B model can hold a hotel phone call in **full duplex**: it listens and speaks
at the same time, every 80 ms, and decides on every frame whether to talk, stay quiet, say "mhm", or stop
because the guest cut in. It is separate from the main thinker–talker system and does not change it.

## Model

```
guest audio ─► Nemotron streaming encoder (frozen) ─► 1 vector / 80 ms ─┐
                                                                         ├─► Qwen3-0.6B (LoRA) ─► text token (inner monologue, PAD = nothing)
agent's own previous Mimi codes + previous text token ──────────────────┘                  └─► depth transformer ─► 8 Mimi codes (agent audio)
```

* One LLM position per 80 ms frame. Its input is the sum of the guest's listener vector, the agent's own
  codes from the previous frame and its previous text token.
* Outputs per frame: a text token (the agent's plan of what it says; PAD when silent) and the 8 Mimi codes
  of the agent's audio. Silence is predicted like any other sound, so turn-taking is learned from data.
* Moshi-style acoustic delay: codebook 1 of frame t is predicted with codebooks 2–8 of frame t−1.
* Tool calls: when the text token is `<tool_call>`, the model writes the call as ordinary tokens. The
  runtime executes it and feeds `<tool_response>…</tool_response>` back, then the frame loop continues.
  Tool calls stay exact text.

## Data (generated, commercially clean)

`s2s/data/duplex_calls.py` and `s2s/prep/duplex_data.py` build two-channel calls from the hotel_v3
generator, which supplies the property prompt, the tool subset, requests, tool calls and replies:

* the agent greets first; then 1–3 guest requests, each answered (with a tool call where needed); then a goodbye
* the guest's voice is one of 23 Kokoro voices, with noise or a phone line on half of the calls; the agent
  has one fixed voice
* timing: the agent answers 0.25–0.6 s after the guest stops; a guest pause after a preface ("Hi, this
  is room 12 …") must not be answered (the agent may say "Mhm."); guest "mhm"s during agent speech must not
  stop it; a real guest interruption stops the agent within 0.25–0.45 s

## Run (Kaggle, 2×T4)

```
PART=data  bash scripts/kaggle_duplex.sh   # ~2-3 h: 10k calls, features -> HF
PART=train bash scripts/kaggle_duplex.sh   # ~8 h; rerun to continue from the last save
PART=eval  bash scripts/kaggle_duplex.sh   # ~30 min
```

## Evaluation (`s2s/eval/duplex_eval.py`)

The model runs frame by frame on held-out calls whose guest channel is fixed. Measured:
* answered within 2 s, and the median response gap
* answering during a guest's mid-turn pause (should be low)
* talking over the guest (should be low)
* stopping when the guest really speaks, and continuing through a guest "mhm"
* tool recall and precision
* intelligibility (Whisper on the agent audio vs its own text)
* real-time factor

With `--wav me.wav` you can be the guest yourself; the output is a stereo file (left = guest, right = agent).

## Limits

* Small-scale experiment: one domain, one agent voice, synthetic guests. Real voices are expected to work
  because the listener is pretrained on real speech, but that is what the test should show.
* The guest channel is assumed echo-free (the agent's own voice removed by echo cancellation).
* While a tool call is written and its result read, the frame clock pauses; the guest audio of that time
  is processed right after.
