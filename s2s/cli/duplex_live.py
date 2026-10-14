"""Talk to the full-duplex model in the browser: you and the agent can speak at the same time.

    python -m s2s.cli.duplex_live --ckpt checkpoints/duplex --tunnel
    # Kaggle: bash scripts/kaggle_duplex_demo.sh

The page streams the microphone continuously; every 80 ms the server feeds the newest listener vector to
the model, which decides by itself whether to talk, stay quiet, say "mhm" or stop because you cut in, and
streams its audio back frame by frame. Unlike the half-duplex demo (ws_live) the microphone is never
muted, so use headphones (or rely on the browser's echo cancellation): the model was trained on a guest
channel without its own voice in it.

Each call gets a random hotel (property facts in the system prompt + a random tool subset) from the
training generator, shown on the page; tool calls run against a mock backend.

Protocol on /ws: client -> server {"type": "hello", "sr": <mic rate>, "temperature": t, "top_k": k},
binary int16 mic PCM, {"type": "reset"}; server -> client {"type": "session", ...}, {"type": "text", "t"},
{"type": "tool", "call", "result"}, {"type": "stats", "lag_ms", "rtf"}, binary int16 24 kHz agent audio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # module level: the route annotations must resolve
from fastapi.responses import HTMLResponse

from s2s.audio import resample, save_audio
from s2s.cli.ws_live import pcm16, start_tunnel
from s2s.data.hotel_v3 import GenericBackend, _Ctx
from s2s.models.streaming_listener import StreamingListener
from s2s.runtime.duplex import DuplexSession, load_duplex
from s2s.utils import resolve_device

STATIC = os.path.join(os.path.dirname(__file__), "static")
HOP = 1280  # 80 ms at 16 kHz


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/duplex")
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder-device", default=None, help="e.g. cuda:1 (default: cuda:1 if there are 2 GPUs)")
    p.add_argument("--encoder", default="nemotron:nvidia/nemotron-speech-streaming-en-0.6b@1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--tunnel", action="store_true")
    p.add_argument("--graphs", type=int, default=1, help="1 = the whole 80 ms frame from one CUDA graph "
                   "(falls back to the normal path if it cannot be recorded)")
    p.add_argument("--graph-max-tokens", type=int, default=4096, help="prompt + frames per call (~4.5 min)")
    p.add_argument("--speech-gate", type=int, default=1, help="1 = the model hears digital silence outside speech "
                   "(Silero VAD): room noise no longer starts phantom turns")
    p.add_argument("--guest-text", type=int, default=1, help="1 = show what the guest said (Nemotron's own "
                   "transcript of the same listener vectors; display only, the model never reads it)")
    p.add_argument("--save-dir", default=None, help="write each call as a stereo wav (left = you, right = agent)")
    args = p.parse_args()

    from s2s.models.codec import MimiCodec, StreamingDecoder
    from s2s.models.encoders import build_encoder

    device = resolve_device(args.device)
    enc_dev = args.encoder_device or ("cuda:1" if torch.cuda.device_count() > 1 else str(device))
    model, tok, cfg, _ = load_duplex(args.ckpt, device)
    codec = MimiCodec(cfg.codec.model, device, num_codebooks=int(cfg.codec.num_codebooks))
    encoder = build_encoder(args.encoder, resolve_device(enc_dev))
    pad_id = model.pad_id
    gate = None
    if args.speech_gate:
        from s2s.runtime.speech_gate import SpeechGate

        gate = SpeechGate()
        print(f"speech gate on ({gate.name} VAD)", flush=True)
    transcriber = None
    if args.guest_text and args.encoder.startswith("nemotron:"):
        from s2s.models.nemotron_text import NemotronTranscriber

        transcriber = NemotronTranscriber.load(args.encoder.split(":", 1)[1].split("@")[0], resolve_device(enc_dev))
    print(f"duplex model on {device}, listener on {enc_dev}", flush=True)
    if args.graphs:  # record the frame graph now (the page's default sampling), not at the first call
        DuplexSession(model, tok, "You are a hotel concierge.", None, None, 0.6, 30, graphs=True,
                      graph_max_tokens=args.graph_max_tokens)
    page = open(os.path.join(STATIC, "duplex.html"), encoding="utf-8").read()
    gpu = threading.Lock()   # one call at a time on the GPU
    busy = {"on": False}
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(page)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        if busy["on"]:
            await sock.send_text(json.dumps({"type": "busy"}))
            await sock.close()
            return
        busy["on"] = True
        try:
            await run_call(sock)
        except WebSocketDisconnect:
            pass
        finally:
            busy["on"] = False

    async def run_call(sock):
        hello = json.loads(await sock.receive_text())
        sr = int(hello.get("sr", 48000))
        temp, top_k = float(hello.get("temperature", 0.6)), int(hello.get("top_k", 30))
        while True:  # one iteration per call; "reset" starts a new one
            seed = random.randrange(10**6)
            ctx = _Ctx(random.Random(seed))
            tools = list(ctx.tools.values()) or None
            with gpu:
                sess = DuplexSession(model, tok, ctx.system, tools, GenericBackend(tools), temp, top_k, seed=seed,
                                     graphs=bool(args.graphs), graph_max_tokens=args.graph_max_tokens)
            dec = StreamingDecoder(codec)
            listener = StreamingListener(encoder)
            if gate is not None:
                gate.reset()
            await sock.send_text(json.dumps({
                "type": "session", "property": ctx.prop, "system": ctx.system,
                "tools": [t["function"]["name"] for t in tools or []], "temperature": temp, "top_k": top_k}))
            mic_rec, agent_rec, t0 = [], [], time.time()
            try:
                await call_loop(sock, sr, sess, dec, listener, mic_rec, agent_rec, t0)
            finally:
                save_call(mic_rec, agent_rec)

    async def call_loop(sock, sr, sess, dec, listener, mic_rec, agent_rec, t0):
        """Two tasks: the receiver feeds the microphone to the listener (GPU 1) and queues its vectors; the
        stepper runs the model on every queued frame (GPU 0) and decodes their audio in one Mimi call. The
        listener and the model therefore work in parallel."""
        frames: asyncio.Queue = asyncio.Queue()
        enc_lock = threading.Lock()
        timing = {"listener": 0.0, "chunks": 0, "decode": 0.0, "decoded": 0}
        encoding = {"busy": False}
        guest = {"frames": [], "ended": 0, "last": 0.0, "sent": ""}

        def transcribe():
            with enc_lock:
                return transcriber([torch.stack(guest["frames"])])[0] if guest["frames"] else ""

        async def guest_text():
            """The guest's words so far (every ~1 s while they speak) and the final line when the gate closes."""
            if transcriber is None:
                return
            if gate is not None and not gate.active and gate.ended == guest["ended"]:
                guest["frames"] = guest["frames"][-25:]  # nobody speaking: keep only the last 2 s
            ended = gate is not None and gate.ended > guest["ended"]
            speaking = gate is None or gate.active
            if not ended and not (speaking and time.time() - guest["last"] > 1.0):
                return
            guest["last"] = time.time()
            text = await asyncio.to_thread(transcribe)
            if ended:
                guest["ended"], guest["frames"] = gate.ended, []
            elif gate is None and len(guest["frames"]) > 150:  # no gate: start a new line every 12 s
                guest["frames"], ended = [], True
            if text and (text != guest["sent"] or ended):
                await sock.send_text(json.dumps({"type": "guest", "t": text, "final": ended}))
            guest["sent"] = "" if ended else text

        def encode():
            with enc_lock:
                t = time.perf_counter()
                out = listener.new_frames()
                timing["listener"] += time.perf_counter() - t
                timing["chunks"] += len(out)   # per 80 ms frame
                return out

        async def run_encode():
            try:
                for f in await asyncio.to_thread(encode):
                    frames.put_nowait(f)
                    guest["frames"].append(f)
                await guest_text()
            finally:
                encoding["busy"] = False

        async def receiver():
            while True:
                msg = await sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if msg.get("bytes"):
                    x16 = resample(np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0, sr, 16000)
                    listener.push(gate.push(x16) if gate is not None else x16)
                    if args.save_dir:
                        mic_rec.append(x16)  # the recording keeps the real microphone
                    if not encoding["busy"] and listener.ready() - listener.done >= 2:
                        encoding["busy"] = True
                        asyncio.create_task(run_encode())
                elif msg.get("text") and json.loads(msg["text"]).get("type") == "reset":
                    return

        def work(batch):
            with gpu:
                outs = [sess.step(f) for f in batch]
                done = [o["frame"].clamp(min=0) for o in outs if o["frame"] is not None]
                t = time.perf_counter()
                wav = dec.push(done) if done else None   # one decoder call for every finished frame
                timing["decode"] += time.perf_counter() - t
                timing["decoded"] += len(done)
                return outs, wav

        async def stepper():
            last_stats = time.time()
            while True:
                batch = [await frames.get()]
                while not frames.empty():
                    batch.append(frames.get_nowait())
                outs, wav = await asyncio.to_thread(work, batch)
                if wav is not None and len(wav):
                    await sock.send_bytes(pcm16(wav))
                    if args.save_dir:
                        agent_rec.append(wav)
                for o in outs:
                    if o["tool"]:
                        await sock.send_text(json.dumps({"type": "tool", "call": o["tool"]["call"],
                                                         "result": o["tool"]["result"]}))
                    elif o["text"] != pad_id:
                        piece = tok.decode([o["text"]], skip_special_tokens=True)
                        if piece:
                            await sock.send_text(json.dumps({"type": "text", "t": piece}))
                if time.time() - last_stats > 1.0:
                    last_stats = time.time()
                    st, n = sess.timing, max(1, sess.timing["frames"])
                    per = {"llm_ms": 1000 * st["llm"] / n, "depth_ms": 1000 * st["depth"] / n,
                           "decode_ms": 1000 * timing["decode"] / max(1, timing["decoded"]),
                           "listener_ms": 1000 * timing["listener"] / max(1, timing["chunks"])}
                    rtf = (per["llm_ms"] + per["depth_ms"] + per["decode_ms"]) / 80.0
                    await sock.send_text(json.dumps({
                        "type": "stats", "lag_ms": int(max(0, listener.ready() - sess.t) * 80), "rtf": round(rtf, 2),
                        "seconds": round(time.time() - t0), **{k: round(v, 1) for k, v in per.items()}}))

        recv, step = asyncio.create_task(receiver()), asyncio.create_task(stepper())
        try:
            done, _ = await asyncio.wait({recv, step}, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                d.result()   # re-raise a disconnect or an error
        finally:
            for task in (recv, step):
                task.cancel()

    def save_call(mic_rec, agent_rec):
        if not (args.save_dir and mic_rec):
            return
        os.makedirs(args.save_dir, exist_ok=True)
        g = resample(np.concatenate(mic_rec), 16000, 24000)
        a = np.concatenate(agent_rec) if agent_rec else np.zeros(0, np.float32)
        n = max(len(g), len(a))
        path = os.path.join(args.save_dir, f"call_{int(time.time())}.wav")
        save_audio(path, np.stack([np.pad(g, (0, n - len(g))), np.pad(a, (0, n - len(a)))], 1), 24000)
        print(f"saved {path}", flush=True)

    if args.tunnel:
        start_tunnel(args.port)
    import uvicorn

    print(f"open http://localhost:{args.port}" + ("  (public link below)" if args.tunnel else ""), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
