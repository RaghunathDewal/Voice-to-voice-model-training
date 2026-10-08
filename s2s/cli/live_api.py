"""/live: the model as an upstream voice API for an application backend (in place of e.g. Gemini Live).

The application owns the guest connection, the system prompt and the tools; this socket only does
speech in -> speech out. Tool calls are sent to the application, which executes them (e.g. against
its own booking backend) and answers with the result; the model then speaks the outcome.

client -> server
  {"type": "setup", "system_prompt": "...", "tools": [...], "input_sample_rate": 16000,
   "output_sample_rate": 24000}                      first message. Tools: OpenAI-style
                                                     {"type": "function", "function": {...}} or
                                                     Gemini-style {"name", "description", "parameters"}
                                                     (Type.OBJECT / "OBJECT" type names are accepted)
  binary int16 mono PCM at input_sample_rate         the guest's microphone; may stream all the time
                                                     (ignored while the model thinks / speaks)
  {"type": "text", "text": "..."}                    a typed user turn, e.g. "Greet the guest"
  {"type": "tool_response", "id": "...", "response": {...}}   one per tool_call

server -> client
  {"type": "setup_complete"}
  {"type": "input_transcription", "text": "...", "confidence": 0.97}   what the guest said
  {"type": "tool_call", "id": "...", "name": "...", "args": {...}}       execute and answer
  binary int16 mono PCM at output_sample_rate                           the reply, streamed
  {"type": "output_transcription", "text": "..."}                        what the model said
  {"type": "turn_complete", "timings_ms": {...}}     sent once the reply has PLAYED (audio is made
                                                     faster than real time), so a client may flush
                                                     its playback buffer on it; listening resumes
  {"type": "error", "message": "..."}

Half duplex: the guest's audio is ignored from the end of their turn until turn_complete.
With --live-token, clients must pass ?token=<token> or an "Authorization: Bearer <token>" header.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import uuid

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from s2s.audio import resample
from s2s.runtime.live import LiveListener

TOOL_TIMEOUT_S = 20.0


def to_openai_tools(tools: list | None) -> list:
    """Accept OpenAI-style or Gemini-style function declarations; return OpenAI-style (lower-case JSON schema)."""

    def lower(schema):
        if isinstance(schema, dict):
            out = {k: lower(v) for k, v in schema.items()}
            if isinstance(out.get("type"), str):
                out["type"] = out["type"].lower()
            return out
        if isinstance(schema, list):
            return [lower(x) for x in schema]
        return schema

    out = []
    for t in tools or []:
        if "functionDeclarations" in t:  # a Gemini `tools` entry: {functionDeclarations: [...]}
            out += to_openai_tools(t["functionDeclarations"])
            continue
        fn = t["function"] if t.get("type") == "function" and "function" in t else t
        params = lower(fn.get("parameters") or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        out.append({"type": "function", "function": {"name": fn["name"], "description": fn.get("description", ""),
                                                      "parameters": params}})
    return out


class RemoteTools:
    """Agent backend whose tools are executed by the client. Called from the generation thread: it sends
    the call, frees the GPU for other conversations while the client works, and waits for the answer."""

    def __init__(self, send, loop: asyncio.AbstractEventLoop, gpu_lock: threading.Lock,
                 timeout: float = TOOL_TIMEOUT_S):
        self.send, self.loop, self.gpu_lock, self.timeout = send, loop, gpu_lock, timeout
        self.pending: dict[str, concurrent.futures.Future] = {}

    def execute(self, call: dict) -> dict:
        call_id = uuid.uuid4().hex[:12]
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.pending[call_id] = fut
        asyncio.run_coroutine_threadsafe(self.send({"type": "tool_call", "id": call_id, "name": call["name"],
                                                    "args": call.get("arguments") or {}}), self.loop)
        self.gpu_lock.release()  # held by the generation thread; other sessions may use the GPU meanwhile
        try:
            result = fut.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            result = {"error": f"tool {call['name']} timed out after {self.timeout:.0f} s"}
        finally:
            self.gpu_lock.acquire()
            self.pending.pop(call_id, None)
        return result if isinstance(result, dict) else {"result": result}

    def answer(self, call_id: str | None, response) -> None:
        fut = self.pending.get(call_id) if call_id else None
        if fut is None and len(self.pending) == 1:  # a client that does not echo ids
            fut = next(iter(self.pending.values()))
        if fut is not None and not fut.done():
            fut.set_result(response)

    def cancel_all(self) -> None:
        for fut in list(self.pending.values()):
            if not fut.done():
                fut.set_result({"error": "connection closed"})


def add_live_api(app: FastAPI, agent, gpu_lock: threading.Lock, token: str | None = None) -> None:
    sr = agent.codec.sample_rate if agent is not None else 24000

    @app.websocket("/live")
    async def live(ws: WebSocket):
        if token:
            given = ws.query_params.get("token") or ws.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if given != token:  # the client sees HTTP 403
                print(f"[live] refused a connection: {'wrong' if given else 'missing'} token", flush=True)
                await ws.close(code=1008, reason="invalid token")
                return
        await ws.accept()
        print("[live] application connected", flush=True)
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        conv: dict = {"state": "setup", "session": None}

        async def send(obj) -> None:
            async with send_lock:
                if isinstance(obj, bytes):
                    await ws.send_bytes(obj)
                else:
                    await ws.send_text(json.dumps(obj, default=str))

        async def run_turn(kind: str, payload) -> None:
            """kind: "audio" (wav at the model rate) or "text"."""
            q: asyncio.Queue = asyncio.Queue()
            session = conv["session"]

            def worker():
                it = session.respond(payload) if kind == "audio" else session.respond_text(payload)
                try:
                    while True:
                        with gpu_lock:  # per step, so conversations interleave and tools can release it
                            ev = next(it, None)
                        if ev is None:
                            break
                        loop.call_soon_threadsafe(q.put_nowait, ev)
                except Exception as e:  # noqa: BLE001 - report, keep the connection alive
                    loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "text": repr(e)})
                loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()
            out_sr = conv["out_sr"]
            play_end, timings = loop.time(), {}
            try:
                while (ev := await q.get()) is not None:
                    t = ev["type"]
                    if t == "audio":
                        rate = ev.get("sample_rate", sr)
                        wav = ev["audio"] if rate == out_sr else resample(ev["audio"], rate, out_sr)
                        await send((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
                        play_end = max(play_end, loop.time() + 0.15) + len(wav) / out_sr
                    elif t == "user_transcript" and kind == "audio":
                        text = ev.get("asr") or ev["text"]
                        await send({"type": "input_transcription", "text": text,
                                    "confidence": ev.get("asr_confidence")})
                    elif t == "assistant_text" and ev["text"]:
                        await send({"type": "output_transcription", "text": ev["text"]})
                    elif t == "timings":
                        timings = {k: round(v) for k, v in ev["ms"].items()}
                    elif t == "error":
                        await send({"type": "error", "message": ev["text"]})
                await asyncio.sleep(max(0.0, play_end + 0.2 - loop.time()))  # let the reply play out
                await send({"type": "turn_complete", "timings_ms": timings})
            except (WebSocketDisconnect, RuntimeError):
                return
            conv["listener"].reset()
            conv["state"] = "listening"

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if conv["state"] != "listening":
                        continue  # half duplex: ignored while the model thinks / speaks (and before setup)
                    chunk = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    if conv["in_sr"] != sr:
                        chunk = resample(chunk, conv["in_sr"], sr)

                    def feed(c=chunk):
                        with gpu_lock:
                            return conv["listener"].feed(c)

                    utterance = await asyncio.to_thread(feed)
                    if utterance is not None:
                        conv["state"] = "thinking"
                        asyncio.create_task(run_turn("audio", utterance))
                    continue
                try:
                    data = json.loads(msg.get("text") or "{}")
                except ValueError:
                    continue
                kind = data.get("type") if isinstance(data, dict) else None
                if kind == "setup":
                    tools = to_openai_tools(data.get("tools"))
                    conv["in_sr"] = int(data.get("input_sample_rate", 16000))
                    conv["out_sr"] = int(data.get("output_sample_rate", 24000))
                    conv["tools"] = RemoteTools(send, loop, gpu_lock)
                    prompt = str(data.get("system_prompt") or "")

                    def build():
                        with gpu_lock:
                            return agent.new_session(system_prompt=prompt or None, tools=tools or None,
                                                     backend=conv["tools"])

                    conv["session"] = await asyncio.to_thread(build)
                    conv["listener"] = LiveListener.for_agent(agent)
                    conv["state"] = "listening"
                    print(f"[live] session: prompt {len(prompt)} chars, tools {[t['function']['name'] for t in tools]}",
                          flush=True)
                    await send({"type": "setup_complete"})
                elif kind == "text" and conv["state"] == "listening" and str(data.get("text", "")).strip():
                    conv["state"] = "thinking"
                    asyncio.create_task(run_turn("text", str(data["text"])))
                elif kind == "tool_response" and conv.get("tools"):
                    conv["tools"].answer(data.get("id"), data.get("response"))
                elif conv["session"] is None:
                    await send({"type": "error", "message": "send a setup message first"})
        except WebSocketDisconnect:
            pass
        finally:
            if conv.get("tools"):
                conv["tools"].cancel_all()
