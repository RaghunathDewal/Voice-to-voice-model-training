"""Turn-based voice agent runtime.

    agent = VoiceAgent(cfg)
    session = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=HotelBackend(res))
    for event in session.respond(wav_24k):         # whole utterance at once
        ...
    # or stream microphone audio:
    turn = session.stream_input()
    for chunk in mic_chunks:                        # any chunk size
        if turn.feed(chunk): break                  # end of turn detected
    for event in turn.finish(): ...

Events are dicts with a "type": user_transcript, token, assistant_text,
tool_call, tool_result, audio (24 kHz float32 numpy), timings, done.

Per session state: thinker KV cache (system prompt, context and tools are
prefilled once and reused for every turn), talker cache, Mimi decoder window.
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch

from s2s.audio import resample
from s2s.models.adapter import SpeechAdapter
from s2s.models.codec import MimiCodec, StreamingDecoder
from s2s.models.talker import Talker, TalkerStream, sample_logits
from s2s.models.thinker import Thinker
from s2s.runtime.endpoint import Endpointer
from s2s.text import ctc_greedy_decode
from s2s.utils import Timer, cuda_sync, load_json, resolve_device, resolve_dtype

_MARKDOWN = set("*#_`~>|[]{}<")


def speakable(piece: str) -> bool:
    """False for tokens that have no sound: emoji (or emoji byte fragments), markdown markers.
    They stay in the text reply but are not sent to the talker (it was trained on plain text)."""
    import unicodedata

    if any(ch.isalnum() for ch in piece):
        return True
    for ch in piece.strip():
        if ch in _MARKDOWN or ch == "\ufffd" or unicodedata.category(ch) in ("So", "Sk", "Cs", "Cf", "Mn", "Co"):
            continue
        return True  # ordinary punctuation shapes the prosody
    return False


class VoiceAgent:
    def __init__(self, cfg, speech_llm_dir: str | None = None, talker_dir: str | None = None,
                 device: str | None = None):
        rt = cfg.runtime
        self.cfg = cfg
        self.device = resolve_device(device or cfg.device)
        self.dtype = resolve_dtype(cfg.thinker.dtype, self.device)
        speech_llm_dir = speech_llm_dir or rt.speech_llm_dir
        talker_dir = talker_dir or rt.talker_dir

        meta_path = os.path.join(talker_dir, "meta.json")
        self.talker_meta = load_json(meta_path) if os.path.exists(meta_path) else {}
        merged = self.talker_meta.get("thinker")
        lora = os.path.join(speech_llm_dir, "lora")
        if merged and os.path.isdir(merged):
            # the talker was trained on this exact (merged) thinker: use it as is
            self.thinker = Thinker(merged, self.device, self.dtype, attn_implementation=cfg.thinker.attn_implementation)
            thinker_desc = merged
        else:
            self.thinker = Thinker(cfg.thinker.model, self.device, self.dtype,
                                   lora_dir=lora if os.path.isdir(lora) else None, merge_lora=True,
                                   attn_implementation=cfg.thinker.attn_implementation)
            thinker_desc = f"{cfg.thinker.model} + {lora if os.path.isdir(lora) else 'no LoRA'}"
            if merged:
                print(f"WARNING: talker was trained on thinker '{merged}' which is not available; using {thinker_desc}")
        self.thinker.model.eval()
        self.adapter = SpeechAdapter.load(os.path.join(speech_llm_dir, "adapter.pt")).to(self.device).eval()
        self.talker = Talker.load(os.path.join(talker_dir, "talker.pt")).to(self.device).eval()
        self.layer_idx = self.talker_meta.get("layer_idx") or self.thinker.hidden_layer_indices(list(cfg.talker.hidden_layers))
        self.codec = MimiCodec(cfg.codec.model, self.device, self.talker.K)
        # speech INPUT encoder: whatever the adapter was trained on (Mimi output codec is separate)
        self.encoder_spec = self.adapter.hparams.get("encoder", "mimi")
        if self.encoder_spec == "mimi":
            self.input_encoder = None
        else:
            from s2s.models.encoders import build_encoder

            self.input_encoder = build_encoder(self.encoder_spec, self.device)
        # only Mimi is causal: its frames never change, so they can be prefilled while the user speaks
        self.incremental_input = self.input_encoder is None
        print(f"[agent] input encoder: {self.encoder_spec}")
        print(f"[agent] thinker: {thinker_desc}; talker layers {self.layer_idx}; device {self.device} {self.dtype}")

    @torch.no_grad()
    def encode_input(self, wavs: list[np.ndarray]) -> list[torch.Tensor]:
        """24 kHz float32 waveforms -> list of [T, D] input features at 12.5 Hz."""
        if self.input_encoder is None:
            return self.codec.encode_latents(wavs)
        sr = self.input_encoder.sample_rate
        return self.input_encoder.encode([resample(w, self.codec.sample_rate, sr) for w in wavs])

    @torch.no_grad()
    def eot_probability(self, wav: np.ndarray) -> float:
        """End-of-turn probability at the last frame of a (partial) 24 kHz utterance."""
        out = self.adapter(self.encode_input([wav])[0][None].to(self.device))
        return float(torch.sigmoid(out["eot_logits"][0, -1]))

    def new_session(self, system_prompt: str | None = None, context: str | None = None,
                    tools: list | None = None, backend=None) -> "VoiceSession":
        system = Thinker.system_content(system_prompt or self.cfg.thinker.system_prompt, context)
        return VoiceSession(self, system, tools, backend)


class VoiceSession:
    def __init__(self, agent: VoiceAgent, system: str, tools: list | None, backend):
        self.a = agent
        self.rt = agent.cfg.runtime
        self.tools = tools
        self.backend = backend
        self.history: list[dict] = []
        self.past = None
        self.n_tokens = 0
        self.first_turn = True
        prompts = agent.thinker.prompts
        self.prefix_ids, self.first_close_ids = prompts.prompt_parts(system, tools, None)
        self.next_open_ids, self.next_close_ids = prompts.user_turn_parts()
        self.generator = torch.Generator(device=agent.device) if agent.device.type == "cuda" else None
        with torch.no_grad():
            self._feed_ids(self.prefix_ids)  # system prompt + context + tools, prefilled once

    # ------------------------------------------------------------- thinker io
    @torch.no_grad()
    def _feed_embeds(self, embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """embeds [n, H] -> (logits at last position [V], selected hidden states [n_sel, H])."""
        th = self.a.thinker
        out = th.model(inputs_embeds=embeds[None].to(th.dtype), past_key_values=self.past, use_cache=True,
                       output_hidden_states=True, logits_to_keep=1)
        self.past = out.past_key_values
        self.n_tokens += embeds.shape[0]
        hidden = torch.stack([out.hidden_states[j][0, -1] for j in self.a.layer_idx], dim=0)
        return out.logits[0, -1], hidden

    def _feed_ids(self, ids: list[int]):
        return self._feed_embeds(self.a.thinker.embed(torch.tensor(ids, device=self.a.device)))

    # --------------------------------------------------------------- turns
    def stream_input(self) -> "StreamingTurn":
        return StreamingTurn(self)

    def respond(self, wav: np.ndarray, timer: Timer | None = None) -> Iterator[dict]:
        """Whole user utterance (24 kHz float32 numpy) -> reply events."""
        timer = timer or Timer()
        turn = StreamingTurn(self, timer=timer)
        timer.mark("endpoint")
        turn.add_latents(self.a.encode_input([wav])[0], wav_frames=None)
        timer.mark("input_encoded")
        yield from turn.finish()

    def _open_turn(self) -> None:
        start, _ = self.a.adapter.boundary_embeddings()
        ids = [] if self.first_turn else self.next_open_ids
        embeds = [self.a.thinker.embed(torch.tensor(ids, device=self.a.device))] if ids else []
        embeds.append(start[None].to(self.a.thinker.dtype))
        self._feed_embeds(torch.cat(embeds, dim=0))

    def _close_turn(self):
        _, end = self.a.adapter.boundary_embeddings()
        ids = self.first_close_ids if self.first_turn else self.next_close_ids
        self.first_turn = False
        embeds = torch.cat([end[None].to(self.a.thinker.dtype),
                            self.a.thinker.embed(torch.tensor(ids, device=self.a.device))], dim=0)
        return self._feed_embeds(embeds)

    # ----------------------------------------------------------- generation
    @torch.no_grad()
    def _generate(self, logits: torch.Tensor, hidden: torch.Tensor, timer: Timer) -> Iterator[dict]:
        a, rt = self.a, self.rt
        th, talker = a.thinker, a.talker
        dev = a.device
        for round_idx in range(int(rt.max_tool_rounds) + 1):
            stream: TalkerStream | None = None
            decoder = StreamingDecoder(a.codec, int(rt.decode_context_frames))
            pending: list[torch.Tensor] = []
            spoken: list[int] = []
            tool_buf: list[int] | None = None
            calls: list[dict] = []
            finished_by_eos = False

            def drain(final: bool = False) -> Iterator[dict]:
                if stream is None:
                    return
                while stream.can_step():
                    new = stream.step()
                    if new:
                        timer.mark("first_audio_frame")
                        pending.extend(new)
                    if len(pending) >= int(rt.emit_every_frames):
                        yield self._emit(decoder, pending, timer)
                if final and pending:
                    yield self._emit(decoder, pending, timer)

            for _ in range(int(rt.max_new_tokens)):
                tok = sample_logits(logits, float(rt.temperature), 0, self.generator)
                timer.mark("first_token")
                if tok in th.eos_ids:
                    finished_by_eos = True
                    break
                if tok == th.prompts.tool_call_start_id:
                    tool_buf = []
                elif tok == th.prompts.tool_call_end_id:
                    call = Thinker.parse_tool_call(th.tokenizer.decode(tool_buf or []))
                    calls.append(call if call else {"name": None, "raw": th.tokenizer.decode(tool_buf or [])})
                    timer.mark("tool_call")
                    tool_buf = None
                elif tool_buf is not None:
                    tool_buf.append(tok)
                else:
                    piece = th.tokenizer.decode([tok])
                    if stream is None and not piece.strip():
                        pass  # leading whitespace is never spoken
                    elif not speakable(piece):
                        spoken.append(tok)  # emoji / markdown: kept in the text, not spoken
                        yield {"type": "token", "text": piece}
                    else:
                        if stream is None:
                            stream = TalkerStream(talker, float(rt.talker_temperature), int(rt.talker_top_k),
                                                  generator=self.generator)
                        spoken.append(tok)
                        timer.mark("first_spoken_token")
                        yield {"type": "token", "text": piece}
                        item = talker.fuse(th.embed(torch.tensor([tok], device=dev)).float(), hidden[None].float())
                        stream.push_text(item)
                logits, hidden = self._feed_ids([tok])
                yield from drain()
            if finished_by_eos:
                self._feed_ids([th.prompts.im_end_id])  # keep the cache aligned with the chat format
            if stream is not None:
                stream.end_text()
                yield from drain(final=True)
            text = th.tokenizer.decode(spoken, skip_special_tokens=True).strip()
            self.history.append({"role": "assistant", "text": text, "tool_calls": calls})
            yield {"type": "assistant_text", "text": text}
            if not calls:
                break
            results = []
            for call in calls:
                yield {"type": "tool_call", "call": call}
                result = self._execute(call)
                results.append(result)
                yield {"type": "tool_result", "call": call, "result": result}
            timer.mark("tool_result")
            if round_idx == int(rt.max_tool_rounds):
                break
            logits, hidden = self._feed_ids(th.prompts.tool_response_ids(results))

    def _execute(self, call: dict) -> dict:
        if call.get("name") is None:
            return {"error": "could not parse tool call", "raw": call.get("raw")}
        if self.backend is None:
            return {"error": "no backend configured"}
        from s2s.data.hotel import validate_call

        err = validate_call(call) if self.tools else None
        if err:
            return {"error": err}
        return self.backend.execute(call)

    def _emit(self, decoder: StreamingDecoder, pending: list[torch.Tensor], timer: Timer) -> dict:
        wav = decoder.push(list(pending))
        pending.clear()
        cuda_sync(self.a.device)
        timer.mark("first_audio_out")
        return {"type": "audio", "audio": wav, "sample_rate": self.a.codec.sample_rate}


class StreamingTurn:
    """One user turn. Audio can be fed incrementally; speech embeddings are
    prefilled into the thinker as they arrive, so little work remains at the
    end of the turn."""

    def __init__(self, session: VoiceSession, timer: Timer | None = None):
        self.s = session
        a = session.a
        self.timer = timer or Timer()
        self.codec = a.codec
        self.audio = np.zeros(0, dtype=np.float32)
        self.latents: torch.Tensor | None = None
        self.fed_frames = 0
        self.eot_probs: list[float] = []
        self.endpointer = Endpointer.from_config(a.cfg.runtime.endpoint, 1000.0 / self.codec.frame_rate)
        self.ended = False
        with torch.no_grad():
            session._open_turn()

    @torch.no_grad()
    def feed(self, chunk: np.ndarray) -> bool:
        """Feed 24 kHz audio. Returns True once the end of the turn is detected."""
        if self.ended:
            return True
        hop = self.codec.hop
        prev_frames = len(self.audio) // hop
        self.audio = np.concatenate([self.audio, np.asarray(chunk, dtype=np.float32)])
        n_frames = len(self.audio) // hop
        if n_frames == prev_frames:
            return False
        if not self.s.a.incremental_input:
            return self._feed_buffered(prev_frames, n_frames)
        # Mimi is causal: re-encoding the buffer leaves earlier frames unchanged.
        # (Simple and correct; a production build would keep Mimi's streaming state.)
        lat = self.codec.encode_latents([self.audio[: n_frames * hop]])[0]
        self.add_latents(lat, wav_frames=(prev_frames, n_frames))
        return self.ended

    def _feed_buffered(self, prev_frames: int, n_frames: int) -> bool:
        """Whole-utterance encoders (e.g. Parakeet): endpoint on the buffer, encode once in finish().
        The end-of-turn head is only run once enough silence has passed (it needs a full encode)."""
        hop, ep = self.codec.hop, self.endpointer
        for f in range(prev_frames, n_frames):
            frame = self.audio[f * hop:(f + 1) * hop]
            eot = 0.0
            if (not ep.is_speech(frame) and ep.speech_ms >= ep.min_speech_ms
                    and ep.silence_ms + ep.frame_ms >= ep.min_silence_ms):
                eot = self.s.a.eot_probability(self.audio[: (f + 1) * hop])
                self.eot_probs.append(eot)
            if ep.update(frame, eot):
                self.ended = True
                self.timer.mark("endpoint")
                break
        return self.ended

    @torch.no_grad()
    def add_latents(self, latents: torch.Tensor, wav_frames: tuple[int, int] | None) -> None:
        a = self.s.a
        self.latents = latents
        out = a.adapter(latents[None].to(a.device))
        new = out["embeds"][0, self.fed_frames:]
        probs = torch.sigmoid(out["eot_logits"][0]).tolist()
        self.eot_probs = probs
        if new.shape[0]:
            self.s._feed_embeds(new)  # incremental prefill
            self.fed_frames = latents.shape[0]
        if wav_frames is not None:
            hop = self.codec.hop
            for f in range(*wav_frames):
                frame = self.audio[f * hop:(f + 1) * hop]
                if self.endpointer.update(frame, probs[f] if f < len(probs) else 0.0):
                    self.ended = True
                    self.timer.mark("endpoint")
                    break

    @torch.no_grad()
    def finish(self) -> Iterator[dict]:
        a = self.s.a
        if self.latents is None and len(self.audio):  # whole-utterance encoder: encode the turn now
            self.add_latents(a.encode_input([self.audio])[0], wav_frames=None)
            self.timer.mark("input_encoded")
        if self.latents is not None:
            ctc = a.adapter(self.latents[None].to(a.device))["ctc_logits"][0].argmax(-1).tolist()
            transcript = ctc_greedy_decode(ctc)
        else:
            transcript = ""
        self.s.history.append({"role": "user", "text": transcript})
        yield {"type": "user_transcript", "text": transcript,
               "eot_prob_last": self.eot_probs[-1] if self.eot_probs else None,
               "endpoint_reason": self.endpointer.reason}
        logits, hidden = self.s._close_turn()
        cuda_sync(a.device)
        self.timer.mark("prefill_done")
        yield from self.s._generate(logits, hidden, self.timer)
        cuda_sync(a.device)
        self.timer.mark("done")
        yield {"type": "timings", "ms": dict(self.timer.marks)}
        yield {"type": "done"}
