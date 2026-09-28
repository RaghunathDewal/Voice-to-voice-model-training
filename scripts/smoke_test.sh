#!/usr/bin/env bash
# End-to-end smoke test with tiny random models and synthetic audio.
# Runs every stage for a few steps (~2-5 min on CPU, faster on GPU).
# Outputs are meaningless; this only proves the code + environment work.
set -euo pipefail
R=${1:-outputs/smoke}
C="--config $R/smoke.yaml"
M=$R/data/manifests

python -m s2s.prep.smoke --root "$R"

echo "== extract Mimi features"
for s in librispeech_train librispeech_dev hotel_train hotel_eval; do
  src=$M/${s}_raw.jsonl; [ -f "$src" ] || src=$M/${s}_audio.jsonl
  python -m s2s.prep.extract_mimi $C --mode latents --in "$src" --out "$M/$s.jsonl"
done
for s in talker_train talker_valid; do
  python -m s2s.prep.extract_mimi $C --mode codes --in "$M/${s}_raw.jsonl" --out "$M/$s.jsonl"
done

echo "== experiment 1: CTC probe"
python -m s2s.train.probe_ctc $C
echo "== distill replies"
python -m s2s.prep.distill $C --in "$M/librispeech_train.jsonl" --out "$M/librispeech_train.jsonl" --max-utts 8 --max-new-tokens 8
echo "== experiment 2: text tool accuracy"
python -m s2s.eval.text_tools $C --manifest "$M/hotel_eval_text.jsonl" --max 2
echo "== stage 2: speech alignment"
python -m s2s.train.speech_llm $C
echo "== stage 3: tools"
python -m s2s.train.speech_llm $C --set \
  train_speech_llm.init_from=$R/checkpoints/speech_llm_align \
  train_speech_llm.output_dir=$R/checkpoints/speech_llm_tools \
  "train_speech_llm.train_manifests=[{path: $M/hotel_train.jsonl, weight: 0.5}, {path: $M/librispeech_train.jsonl, weight: 0.5}]" \
  train_speech_llm.valid_manifest=$M/hotel_eval.jsonl
echo "== experiment 3: speech vs text"
python -m s2s.eval.speech_llm $C --speech-llm-dir $R/checkpoints/speech_llm_tools --manifest "$M/hotel_eval.jsonl" --max 2
python -m s2s.eval.speech_llm $C --speech-llm-dir $R/checkpoints/speech_llm_tools --manifest "$M/librispeech_dev.jsonl" --max 2
echo "== merge LoRA"
python -m s2s.prep.merge_lora $C --speech-llm-dir $R/checkpoints/speech_llm_tools --out $R/checkpoints/thinker_merged
echo "== stage 4: talker"
python -m s2s.train.talker $C
echo "== runtime: chat (whole file and streamed)"
WAV=$(ls $R/data/wavs/hotel_0000.wav)
python -m s2s.cli.chat $C --wav "$WAV" "$WAV" --out-dir $R/chat
python -m s2s.cli.chat $C --wav "$WAV" --out-dir $R/chat_stream --stream
echo "== experiment 5: latency"
python -m s2s.eval.latency $C --wav $R/data/wavs/hotel_000*.wav --max 2 --out $R/latency.json
if [ "${SKIP_WHISPER:-0}" != "1" ]; then
  echo "== experiment 4: talker intelligibility (downloads whisper-tiny)"
  python -m s2s.eval.talker $C --talker-dir $R/checkpoints/talker --manifest "$M/talker_valid.jsonl" --max 2 \
    --asr openai/whisper-tiny --out-dir $R/talker_eval
fi
echo "SMOKE TEST PASSED"
