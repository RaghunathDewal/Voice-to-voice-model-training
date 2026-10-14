"""/live for MANY calls at once on one GPU (adapter v3 / v4): the whole pipeline batched.

    python -m s2s.cli.batch_live --config configs/qwen4b_v3.yaml --speech-llm-dir checkpoints/speech_llm_v4 \\
        --tts-url http://127.0.0.1:7791/tts,http://127.0.0.1:7792/tts --max-calls 64
    python -m s2s.eval.load_test --url ws://127.0.0.1:7860/live --concurrency 1 8 16 32 64   # the guests

Same protocol as s2s/cli/live_api.py (setup / binary PCM / text / tool_response -> setup_complete /
input_transcription / tool_call / binary PCM / output_transcription / turn_complete), half duplex.

    listener thread  every ~40 ms: each call's new microphone audio -> voice activity + end-of-turn rule
                     (runtime.endpoint, the adapter's end-of-turn head on the call's listener vectors) ->
                     ALL calls' 160 ms chunks through Nemotron in one pass (s2s/models/batched_listener.py).
                     At the end of a turn its vectors go to the call's coroutine.
    scheduler thread the 4B for every call (s2s/runtime/batch_engine.py): new turns prefilled in budgeted
                     pieces, one decode pass for all answering calls.
    per-call coroutine  reply tokens -> tool calls to the application, speakable text in phrases to a pool of
                     Kokoro servers (each phrase to the least busy one) -> audio to the guest.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import threading
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # module level: the route annotations must resolve

from s2s.audio import resample

HOP = 1280  # 80 ms at 16 kHz: one listener vector


class Call:
    """One connected guest. Audio and endpoint state are touched by the listener thread only (under `lock`)."""

    def __init__(self, pipe: "Pipeline", loop: asyncio.AbstractEventLoop):
        from s2s.runtime.endpoint import Endpointer

        self.pipe, self.loop = pipe, loop
        self.lock = threading.Lock()
        self.inbox: collections.deque = collections.deque()       # 16 kHz float32 chunks from the socket
        self.stream = pipe.listener.new_stream()
        self.ep = Endpointer.from_config(pipe.cfg.runtime.endpoint, 80.0, 16000)
        self.listening = False
        self.buf = np.zeros(0, np.float32)                         # audio not yet cut into 80 ms frames
        self.samples = 0                                           # 80 ms frames of audio seen so far
        self.frames: list[torch.Tensor] = []                       # listener vectors, frames[i] = vector f0 + i
        self.f0 = 0
        self.preroll = 4
        self.turn_start: int | None = None                         # frame index where the turn's speech starts
        self.turn_end: int | None = None                           # set at the end of turn, until dispatched
        self.t_endpoint = 0.0
        self.turns: asyncio.Queue = asyncio.Queue()
        self.conv = None
        self.first_turn = True
        self.closed = False

    # ---------------------------------------------------------- listener thread
    def take_audio(self) -> None:
        while self.inbox:
            x = self.inbox.popleft()
            self.stream.push(x)  # the listener hears everything (its frame k = audio frame k)
            self.buf = np.concatenate([self.buf, x])
            while len(self.buf) >= HOP:
                frame, self.buf = self.buf[:HOP], self.buf[HOP:]
                if self.listening:
                    self._frame(frame, self.samples)
                self.samples += 1

    def _frame(self, frame: np.ndarray, k: int) -> None:
        ep = self.ep
        if self.turn_end is not None:
            return
        if self.turn_start is None:
            if ep.is_speech(frame):
                self.turn_start = max(0, k - self.preroll)
                ep.reset()
            else:
                return
        silent = not ep.is_speech(frame)
        eot = 0.0
        if silent and ep.speech_ms >= ep.min_speech_ms and ep.silence_ms + ep.frame_ms >= ep.eot_check_ms:
            eot = self.pipe.eot(self.vectors(self.turn_start, k + 1))
        if ep.update(frame, eot):
            self.turn_end, self.t_endpoint = k + 1, time.perf_counter()
            return
        if ep.speech_ms < ep.min_speech_ms and ep.silence_ms >= ep.max_silence_ms:
            self.turn_start = None  # a click or a cough
            ep.reset()
        elif k + 1 - self.turn_start >= 30 * 12.5:
            self.turn_end, self.t_endpoint = k + 1, time.perf_counter()

    def add_frames(self, frames: list[torch.Tensor]) -> None:
        self.frames.extend(frames)
        keep_from = self.turn_start if self.turn_start is not None else self.f0 + len(self.frames) - 8
        drop = max(0, keep_from - self.f0)
        if drop:
            self.frames, self.f0 = self.frames[drop:], self.f0 + drop

    def vectors(self, start: int, end: int) -> torch.Tensor | None:
        a, b = max(0, start - self.f0), min(len(self.frames), end - self.f0)
        return torch.stack(self.frames[a:b]) if b > a else None

    def dispatch(self) -> None:
        """At the end of a turn, once its vectors are encoded (+1 frame of look-ahead): hand them to the call."""
        if self.turn_end is None or self.f0 + len(self.frames) < self.turn_end + 1:
            return
        lat = self.vectors(self.turn_start, self.turn_end + 1)
        t_end = self.t_endpoint
        self.turn_start = self.turn_end = None
        self.listening = False  # half duplex: until the reply has played
        self.loop.call_soon_threadsafe(self.turns.put_nowait, ("audio", lat, t_end))


class Pipeline:
    def __init__(self, args):
        from s2s.config import load_config
        from s2s.models.adapter import SpeechAdapter
        from s2s.models.batched_listener import BatchedListener
        from s2s.models.encoders import build_encoder
        from s2s.models.thinker import Thinker
        from s2s.runtime.batch_engine import BatchedThinker, BatchScheduler
        from s2s.runtime.external_tts import ExternalTTS
        from s2s.utils import resolve_device, resolve_dtype

        cfg = self.cfg = load_config(args.config, args.set)
        self.args = args
        device = self.device = resolve_device(args.device or cfg.device)
        dtype = resolve_dtype(cfg.thinker.dtype, device)
        lora = os.path.join(args.speech_llm_dir, "lora")
        self.th = Thinker(cfg.thinker.model, device, dtype, lora_dir=lora if os.path.isdir(lora) else None,
                          merge_lora=True)
        self.th.model.eval()
        self.adapter = SpeechAdapter.load(os.path.join(args.speech_llm_dir, "adapter.pt")).to(device).eval()
        spec = self.adapter.hparams.get("encoder", "")
        if not str(spec).startswith("nemotron"):
            raise SystemExit(f"batch_live needs a Nemotron-listener adapter (v3/v4), this one uses {spec!r}")
        enc_dev = resolve_device(args.encoder_device) if args.encoder_device else device
        self.nem = build_encoder(spec, enc_dev)
        self.listener = BatchedListener(self.nem, lookahead=self.nem.lookahead)
        self.engine = BatchedThinker(self.th.model, slots=args.max_calls, max_tokens=args.max_tokens,
                                     device=device, dtype=dtype)
        emb = self.th.model.get_input_embeddings()
        start, end = self.adapter.boundary_embeddings()
        self.sched = BatchScheduler(self.engine, lambda ids: emb(torch.tensor(ids, device=device)), self.adapter,
                                    (start.detach(), end.detach()), eos_ids=self.th.eos_ids,
                                    im_end_id=self.th.im_end_id, prefill_budget=args.prefill_budget)
        self.sched.start()
        self.tts = ExternalTTS(args.tts_url)
        self.calls: list[Call] = []
        self.calls_lock = threading.Lock()
        self.stats = {"listener_ticks": 0, "listener_s": 0.0, "vad_eot_s": 0.0, "turns": 0}
        self.stop = False
        threading.Thread(target=self._listen_loop, daemon=True).start()
        print(f"[batch_live] {args.speech_llm_dir} on {device} (listener {enc_dev}), up to {args.max_calls} calls, "
              f"KV cache {self.engine.cache_gb:.1f} GB, voice: {len(self.tts.urls)} Kokoro server(s)", flush=True)

    @torch.no_grad()
    def eot(self, lat: torch.Tensor | None) -> float:
        if lat is None:
            return 0.0
        return float(torch.sigmoid(self.adapter(lat[None].to(self.device))["eot_logits"][0, -1]))

    @torch.no_grad()
    def transcript(self, lat: torch.Tensor) -> str:
        from s2s.text import ctc_greedy_decode

        ids = self.adapter(lat[None].to(self.device))["ctc_logits"][0].argmax(-1).tolist()
        return ctc_greedy_decode(ids)

    def _listen_loop(self) -> None:
        while not self.stop:
            t0 = time.perf_counter()
            with self.calls_lock:
                calls = [c for c in self.calls if not c.closed]
            for c in calls:
                with c.lock:
                    c.take_audio()
            t1 = time.perf_counter()
            res = self.listener.step([c.stream for c in calls])
            for c in calls:
                with c.lock:
                    c.add_frames(res.get(id(c.stream), []))
                    c.dispatch()
            self.stats["listener_ticks"] += 1
            self.stats["vad_eot_s"] += t1 - t0
            self.stats["listener_s"] += time.perf_counter() - t1
            time.sleep(max(0.0, 0.04 - (time.perf_counter() - t0)))


def build_app(pipe: Pipeline):
    from s2s.cli.live_api import to_openai_tools
    from s2s.models.thinker import Thinker
    from s2s.runtime.agent import speakable
    from s2s.runtime.batch_engine import Job

    app = FastAPI()
    th, prompts, args = pipe.th, pipe.th.prompts, pipe.args
    next_open, next_close = prompts.user_turn_parts()

    @app.get("/health")
    async def health():
        return {"calls": len(pipe.calls), "max_calls": args.max_calls, "scheduler": dict(pipe.sched.stats),
                "listener": dict(pipe.listener.stats), "pipeline": dict(pipe.stats)}

    @app.websocket("/live")
    async def live(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        tool_waits: dict[str, asyncio.Future] = {}

        async def send(obj) -> None:
            async with send_lock:
                if isinstance(obj, bytes):
                    await ws.send_bytes(obj)
                else:
                    await ws.send_text(json.dumps(obj, default=str))

        if len(pipe.calls) >= args.max_calls:
            await send({"type": "error", "message": f"server full ({args.max_calls} calls)"})
            await ws.close()
            return
        call = Call(pipe, loop)
        state = {"in_sr": 16000, "out_sr": 24000, "close_ids": None}

        async def generate(pieces: list, timings: dict) -> None:
            """One guest turn: reply tokens -> tool calls / phrases -> audio; tool rounds until a reply."""
            out_sr = state["out_sr"]
            play_end, t_ref = loop.time(), timings.pop("_t_end")
            for round_idx in range(int(pipe.cfg.runtime.max_tool_rounds) + 1):
                q: asyncio.Queue = asyncio.Queue()
                pipe.sched.submit(Job(call.conv, pieces, max_new=int(pipe.cfg.runtime.max_new_tokens),
                                      temperature=float(pipe.cfg.runtime.temperature),
                                      on_token=lambda t, now: loop.call_soon_threadsafe(q.put_nowait, (t, now)),
                                      on_done=lambda st: loop.call_soon_threadsafe(q.put_nowait, (None, st))))
                phrases = pipe.tts.stream(quick_start=round_idx == 0)
                spoken, tool_buf, calls = [], None, []

                async def flush(final: bool = False):
                    nonlocal play_end
                    if final:
                        phrases._submit()
                    while phrases.futures and (final or phrases.futures[0].done()):
                        fut = phrases.futures[0]
                        if not fut.done():
                            await asyncio.sleep(0.005)
                            continue
                        phrases.futures.popleft()
                        wav, rate = fut.result()
                        if rate != out_sr:
                            wav = resample(wav, rate, out_sr)
                        timings.setdefault("first_audio", round((time.perf_counter() - t_ref) * 1000))
                        await send((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
                        play_end = max(play_end, loop.time() + 0.15) + len(wav) / out_sr

                while True:
                    tok, extra = await q.get()
                    if tok is None:
                        if "error" in extra:
                            await send({"type": "error", "message": extra["error"]})
                        break
                    timings.setdefault("first_token", round((extra - t_ref) * 1000))
                    if tok == prompts.tool_call_start_id:
                        tool_buf = []
                    elif tok == prompts.tool_call_end_id:
                        call_ = Thinker.parse_tool_call(th.tokenizer.decode(tool_buf or []))
                        calls.append(call_ or {"name": None})
                        tool_buf = None
                    elif tool_buf is not None:
                        tool_buf.append(tok)
                    else:
                        piece = th.tokenizer.decode([tok])
                        if not spoken and not piece.strip():
                            continue
                        spoken.append(tok)
                        if speakable(piece) or not piece.strip():
                            phrases.push(piece)
                    await flush()
                await flush(final=True)
                phrases.pool.shutdown(wait=False)
                text = th.tokenizer.decode(spoken, skip_special_tokens=True).strip()
                if text:
                    await send({"type": "output_transcription", "text": text})
                if not calls or round_idx == int(pipe.cfg.runtime.max_tool_rounds):
                    break
                results = []
                for c in calls:
                    if not c.get("name"):
                        results.append({"error": "could not parse the tool call"})
                        continue
                    cid = uuid.uuid4().hex[:12]
                    fut = tool_waits[cid] = loop.create_future()
                    await send({"type": "tool_call", "id": cid, "name": c["name"], "args": c.get("arguments") or {}})
                    try:
                        res = await asyncio.wait_for(fut, 20.0)
                    except asyncio.TimeoutError:
                        res = {"error": f"tool {c['name']} timed out"}
                    tool_waits.pop(cid, None)
                    results.append(res if isinstance(res, dict) else {"result": res})
                pieces = [("ids", prompts.tool_response_ids(results))]
            await asyncio.sleep(max(0.0, play_end + 0.2 - loop.time()))  # let the reply play out
            await send({"type": "turn_complete", "timings_ms": timings})

        async def turns_worker():
            while True:
                kind, payload, t_end = await call.turns.get()
                timings: dict = {"_t_end": t_end}
                try:
                    if kind == "audio":
                        text = pipe.transcript(payload) if payload is not None else ""
                        if payload is None or not text.strip():  # noise, no words: no reply
                            continue
                        await send({"type": "input_transcription", "text": text})
                        speech = [("speech", payload)]
                    else:
                        speech = [("ids", prompts.ids(payload))]
                    pieces = ([] if call.first_turn else [("ids", next_open)]) + speech + \
                        [("ids", state["close_ids"] if call.first_turn else next_close)]
                    call.first_turn = False
                    pipe.stats["turns"] += 1
                    await generate(pieces, timings)
                except (WebSocketDisconnect, RuntimeError):
                    return
                finally:
                    with call.lock:
                        call.ep.reset()
                        call.ep.reset_vad()
                        call.listening = True

        worker = None
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if call.conv is None:
                        continue
                    x = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    call.inbox.append(x if state["in_sr"] == 16000 else resample(x, state["in_sr"], 16000))
                    continue
                try:
                    data = json.loads(msg.get("text") or "{}")
                except ValueError:
                    continue
                kind = data.get("type") if isinstance(data, dict) else None
                if kind == "setup" and call.conv is None:
                    tools = to_openai_tools(data.get("tools")) or None
                    state["in_sr"] = int(data.get("input_sample_rate", 16000))
                    state["out_sr"] = int(data.get("output_sample_rate", 24000))
                    system = Thinker.system_content(str(data.get("system_prompt") or "") or pipe.cfg.thinker.system_prompt)
                    prefix, state["close_ids"] = prompts.prompt_parts(system, tools, None)
                    call.conv = pipe.sched.open()
                    done = asyncio.Event()
                    pipe.sched.submit(Job(call.conv, [("ids", prefix)], generate=False,
                                          on_done=lambda st: loop.call_soon_threadsafe(done.set)))
                    await done.wait()
                    with pipe.calls_lock:
                        pipe.calls.append(call)
                    call.listening = True
                    worker = asyncio.create_task(turns_worker())
                    await send({"type": "setup_complete"})
                elif kind == "text" and call.conv is not None and str(data.get("text", "")).strip():
                    call.listening = False
                    call.turns.put_nowait(("text", str(data["text"]), time.perf_counter()))
                elif kind == "tool_response":
                    fut = tool_waits.get(data.get("id")) or (next(iter(tool_waits.values())) if len(tool_waits) == 1 else None)
                    if fut is not None and not fut.done():
                        fut.set_result(data.get("response"))
        except WebSocketDisconnect:
            pass
        finally:
            call.closed = True
            if worker:
                worker.cancel()
            with pipe.calls_lock:
                if call in pipe.calls:
                    pipe.calls.remove(call)
            if call.conv is not None:
                pipe.sched.close(call.conv)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/qwen4b_v3.yaml")
    p.add_argument("--speech-llm-dir", default="checkpoints/speech_llm_v4")
    p.add_argument("--tts-url", default="http://127.0.0.1:7791/tts", help="Kokoro server(s), comma-separated")
    p.add_argument("--max-calls", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=4096, help="KV cache positions per call")
    p.add_argument("--prefill-budget", type=int, default=1024)
    p.add_argument("--device", default=None)
    p.add_argument("--encoder-device", default=None, help="listener GPU (default: the 4B's)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--set", nargs="*", default=[], help="config overrides, key=value")
    args = p.parse_args()
    import uvicorn

    pipe = Pipeline(args)
    uvicorn.run(build_app(pipe), host=args.host, port=args.port, log_level="warning", ws_max_size=2**24)


if __name__ == "__main__":
    main()
