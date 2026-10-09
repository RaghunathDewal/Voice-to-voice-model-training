"""LFM2.5-Audio (base or fine-tuned) behind the same agent interface as VoiceAgent, so the live UI
(s2s/cli/ws_live.py, /ws and /device) and the application API (/live) can serve it unchanged:

    python -m s2s.cli.ws_live --backend lfm --lfm-model-dir ~/lfm_ft/model \
        --system-prompt-file demo/thinker/example_prompt.txt --tools-file demo/thinker/tools_template.json \
        --mock-results demo/thinker/mock_results_template.json --tunnel

One end-to-end model: the guest's audio goes in, interleaved text + audio comes out and is streamed frame by
frame (Mimi streaming decoder, 80 ms per frame). A turn that starts with a tool call is text only; the call is
executed by the session's backend and the result goes back in as a "tool" turn (lfm_audio_probe.generate_hybrid,
the same layout as the fine-tuning data). Turn ends are detected by the live listener's VAD; LFM has no
end-of-turn head, so the end is decided by silence alone (runtime.endpoint.min_silence_ms).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from s2s.eval.lfm_audio_probe import (EOS_AUDIO, HF_REPO, OUT_SR, generate_hybrid, parse_tool_calls, setup_for,
                                      spoken)


class _Codec:
    """What LiveListener / the servers read from agent.codec."""
    sample_rate = OUT_SR  # 24 kHz in and out
    frame_rate = 12.5
    hop = int(OUT_SR / 12.5)  # 1920 samples = 80 ms


class LFMAgent:
    def __init__(self, cfg, model_dir: str | None = None, device: str = "cuda", audio_temperature: float = 1.0,
                 audio_top_k: int = 4, max_new_tokens: int = 768, emit_frames: int = 2,
                 end_silence_ms: float | None = None):
        from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality

        self.cfg = cfg
        if end_silence_ms is not None:  # no end-of-turn head: silence alone ends the turn
            cfg.runtime.endpoint.min_silence_ms = float(end_silence_ms)
        self.ChatState, self.Mod = ChatState, LFMModality
        src = Path(os.path.expanduser(model_dir)) if model_dir else HF_REPO
        self.name = str(model_dir or HF_REPO)
        self.proc = LFM2AudioProcessor.from_pretrained(src, device=device).eval()
        self.model = LFM2AudioModel.from_pretrained(src, device=device, dtype=torch.bfloat16).eval()
        self.mimi = self.proc.mimi.eval()
        self.codec = _Codec()
        self.device = device
        self.gen_kw = dict(max_new_tokens=max_new_tokens, audio_temperature=audio_temperature, audio_top_k=audio_top_k)
        self.emit_frames = emit_frames
        with torch.no_grad(), self.mimi.streaming(1):  # warm-up (kernels, compiled Mimi graph)
            for _ in range(3):
                self.mimi.decode(torch.randint(2048, (1, 8, 1), device=device))
        print(f"[lfm] {self.name} ready on {device}", flush=True)

    def eot_probability(self, wav: np.ndarray) -> float:
        return 1.0  # no end-of-turn head: the listener ends the turn after min_silence_ms of silence

    def new_session(self, system_prompt: str | None = None, context: str | None = None, tools: list | None = None,
                    backend=None) -> "LFMSession":
        prompt = system_prompt or ("You are the voice concierge of a holiday park, speaking with a guest out loud.\n"
                                   + (context or ""))
        return LFMSession(self, prompt, tools or [], backend)


class LFMSession:
    def __init__(self, agent: LFMAgent, prompt: str, tools: list, backend):
        self.a = agent
        self.backend = backend
        fn_tools = [t.get("function", t) for t in tools]
        system, self.prefix = setup_for(prompt, fn_tools, "system")
        c = agent.ChatState(agent.proc, dtype=torch.bfloat16)
        c.new_turn("system")
        c.add_text(system)
        c.end_turn()
        self.chat = c

    # ------------------------------------------------------------------ turns
    def respond(self, wav: np.ndarray, timer=None) -> Iterator[dict]:
        """A whole guest utterance (24 kHz float32) -> reply events (same events as VoiceSession.respond)."""
        t0 = time.perf_counter()
        self.chat.new_turn("user")
        self.chat.add_audio(torch.from_numpy(np.asarray(wav, dtype=np.float32))[None], OUT_SR)
        self.chat.end_turn()
        self.chat.new_turn("assistant")
        yield {"type": "user_transcript", "text": f"(speech, {len(wav) / OUT_SR:.1f} s)", "asr": None}
        yield from self._reply(t0)

    def respond_text(self, text: str, timer=None) -> Iterator[dict]:
        t0 = time.perf_counter()
        self.chat.new_turn("user")
        self.chat.add_text(text)
        self.chat.end_turn()
        self.chat.new_turn("assistant")
        yield from self._reply(t0)

    @torch.no_grad()
    def _reply(self, t0: float) -> Iterator[dict]:
        a = self.a
        marks: dict[str, float] = {}
        for _ in range(3):  # tool rounds
            text_toks, frames, mod, pending = [], [], [], []
            with a.mimi.streaming(1):
                for t in generate_hybrid(a.model, **self.chat, **a.gen_kw):
                    if t.numel() == 1:
                        text_toks.append(t)
                        mod.append(a.Mod.TEXT)
                        marks.setdefault("first_text", 1000 * (time.perf_counter() - t0))
                        continue
                    frames.append(t)
                    mod.append(a.Mod.AUDIO_OUT)
                    if bool((t == EOS_AUDIO).any()):
                        continue
                    pending.append(a.mimi.decode(t[None, :, None])[0, 0].float().cpu().numpy())
                    if len(pending) >= a.emit_frames:
                        marks.setdefault("first_audio", 1000 * (time.perf_counter() - t0))
                        yield {"type": "audio", "audio": np.concatenate(pending), "sample_rate": OUT_SR}
                        pending = []
                if pending:
                    marks.setdefault("first_audio", 1000 * (time.perf_counter() - t0))
                    yield {"type": "audio", "audio": np.concatenate(pending), "sample_rate": OUT_SR}
            dev = self.chat.text.device
            self.chat.append(
                text=torch.stack(text_toks, 1) if text_toks else torch.empty((1, 0), dtype=torch.long, device=dev),
                audio_out=torch.stack(frames, 1) if frames else torch.empty((8, 0), dtype=torch.long, device=dev),
                modality_flag=torch.tensor([int(m) for m in mod], dtype=torch.long))
            self.chat.end_turn()
            raw = a.proc.text.decode(torch.cat(text_toks)) if text_toks else ""
            calls = parse_tool_calls(raw)
            if not calls:
                yield {"type": "assistant_text", "text": spoken(raw)}
                break
            results = []
            for call in calls:
                yield {"type": "tool_call", "call": call}
                result = self.backend.execute(call) if self.backend is not None else {"error": "no tool backend"}
                results.append(result)
                yield {"type": "tool_result", "call": call, "result": result}
            payload = json.dumps(results[0] if len(results) == 1 else results)
            self.chat.new_turn("tool")
            self.chat.add_text(f"<|tool_response_start|>{payload}<|tool_response_end|>")
            self.chat.end_turn()
            self.chat.new_turn("assistant")
        marks["done"] = 1000 * (time.perf_counter() - t0)
        yield {"type": "timings", "ms": marks}
        yield {"type": "done"}


class CannedBackend:
    """Tool results from a JSON file ({"tool_name": result, ...}); other tools echo their arguments."""

    def __init__(self, path: str):
        self.canned = json.loads(Path(path).read_text(encoding="utf-8"))

    def execute(self, call: dict) -> dict:
        return self.canned.get(call.get("name"), {"success": True, **(call.get("arguments") or {})})
