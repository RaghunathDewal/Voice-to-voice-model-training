"""Hands-free voice demo over a WebSocket: audio is streamed both ways while it is produced.

    python -m s2s.cli.ws_live --config configs/small.yaml \
        --speech-llm-dir checkpoints/speech_llm_pk3 --talker-dir checkpoints/talker_v2 --tunnel

The browser sends microphone PCM continuously; the server detects the end of the turn
(energy VAD + the adapter's end-of-turn head), runs the agent and sends every audio
chunk the moment it is decoded. The page plays chunks back to back with a ~120 ms
jitter buffer, so the reply starts as soon as its first audio exists instead of after
the whole reply has been generated. Turn-taking is strict: the microphone is ignored
while the agent thinks or speaks, and listening resumes once the browser has played
the reply.

--tunnel starts a Cloudflare quick tunnel (no account needed) and prints a public
https:// URL, which browsers require for microphone access (Kaggle / Colab / remote GPU).

Protocol on /ws: client -> server: {"type": "hello", "sr": <mic rate>}, binary int16 mic PCM,
{"type": "played"}, {"type": "reset"}, {"type": "latency", "first_audio_ms": ...};
server -> client: {"type": "state", "state": listening|thinking|speaking}, {"type": "log", "text"},
binary int16 24 kHz reply audio, {"type": "reply_done"}.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # module level: route annotations must resolve
from fastapi.responses import HTMLResponse, Response

from s2s.audio import resample
from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.runtime.agent import VoiceAgent
from s2s.runtime.live import LiveListener

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hotel voice agent</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
 button{font-size:16px;padding:10px 18px;margin-right:8px;border-radius:8px;border:1px solid #888;cursor:pointer}
 #status{font-size:22px;margin:18px 0}
 #log{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;height:420px;overflow:auto;white-space:pre-wrap;font-size:13px}
</style></head><body>
<h2>Hotel voice agent</h2>
<button id="start">Start talking</button><button id="reset" disabled>New session</button>
<div id="status">Press "Start talking" and allow the microphone.</div>
<div id="log"></div>
<script>
let ws, ctx, state = 'idle', sendBuf = [], playHead = 0, playing = 0, replyDone = false, firstAudio = true, turnEndAt = 0;
const logEl = document.getElementById('log');
const log = t => { logEl.textContent += t + "\n"; logEl.scrollTop = logEl.scrollHeight; };
const setStatus = t => document.getElementById('status').textContent = t;
const LABEL = {listening: '🎙️ Listening — just speak', thinking: '🤔 Thinking…', speaking: '🗣️ Speaking…'};

document.getElementById('start').onclick = async () => {
  document.getElementById('start').disabled = true;
  ctx = new AudioContext();
  const stream = await navigator.mediaDevices.getUserMedia(
    {audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1}});
  const code = "class C extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}};registerProcessor('cap',C);";
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code], {type: 'application/javascript'})));
  const node = new AudioWorkletNode(ctx, 'cap');
  ctx.createMediaStreamSource(stream).connect(node);
  const mute = ctx.createGain(); mute.gain.value = 0; node.connect(mute); mute.connect(ctx.destination);
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { ws.send(JSON.stringify({type: 'hello', sr: ctx.sampleRate})); document.getElementById('reset').disabled = false; };
  ws.onclose = () => setStatus('Disconnected — reload the page.');
  ws.onmessage = ev => {
    if (typeof ev.data !== 'string') { playChunk(new Int16Array(ev.data)); return; }
    const m = JSON.parse(ev.data);
    if (m.type === 'state') {
      state = m.state; setStatus(LABEL[state] || state);
      if (state === 'thinking') { turnEndAt = performance.now(); firstAudio = true; replyDone = false; }
    } else if (m.type === 'log') log(m.text);
    else if (m.type === 'reply_done') { replyDone = true; maybeDone(); }
  };
  node.port.onmessage = e => {
    if (state !== 'listening' || ws.readyState !== 1) { sendBuf = []; return; }
    sendBuf.push(e.data);
    const n = sendBuf.reduce((a, b) => a + b.length, 0);
    if (n < ctx.sampleRate * 0.08) return;                       // send 80 ms at a time
    const pcm = new Int16Array(n); let o = 0;
    for (const b of sendBuf) for (let i = 0; i < b.length; i++) pcm[o++] = Math.max(-1, Math.min(1, b[i])) * 32767;
    sendBuf = []; ws.send(pcm.buffer);
  };
};
document.getElementById('reset').onclick = () => { logEl.textContent = ''; ws.send(JSON.stringify({type: 'reset'})); };

function playChunk(pcm) {
  const f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
  const buf = ctx.createBuffer(1, f.length, 24000); buf.copyToChannel(f, 0);
  const s = ctx.createBufferSource(); s.buffer = buf; s.connect(ctx.destination);
  const now = ctx.currentTime;
  if (playHead < now + 0.02) playHead = now + 0.12;              // (re)start with a small jitter buffer
  s.start(playHead);
  if (firstAudio) {
    firstAudio = false;
    ws.send(JSON.stringify({type: 'latency', first_audio_ms: Math.round(performance.now() - turnEndAt + (playHead - now) * 1000)}));
  }
  playHead += buf.duration; playing++;
  s.onended = () => { playing--; maybeDone(); };
}
function maybeDone() { if (replyDone && playing === 0) { replyDone = false; ws.send(JSON.stringify({type: 'played'})); } }
</script></body></html>
"""


def start_tunnel(port: int) -> subprocess.Popen:
    """Cloudflare quick tunnel -> public https URL (printed). Downloads cloudflared if needed."""
    exe = shutil.which("cloudflared") or "/tmp/cloudflared"
    if not os.path.exists(exe):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        print("downloading cloudflared ...", flush=True)
        urllib.request.urlretrieve(url, exe)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def watch():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                print(f"\nOPEN THIS: {m.group(0)}\n", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    return proc


def pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def build_app(agent: VoiceAgent) -> FastAPI:
    sr = agent.codec.sample_rate
    gpu_lock = threading.Lock()  # one model call at a time across connections
    app = FastAPI()

    @app.get("/")
    def index():
        return HTMLResponse(PAGE)

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        conv: dict = {"mic_sr": 48000, "state": "listening", "turn": 0}

        async def say(text: str) -> None:
            print(text, flush=True)
            await ws.send_text(json.dumps({"type": "log", "text": text}))

        async def set_state(s: str) -> None:
            conv["state"] = s
            await ws.send_text(json.dumps({"type": "state", "state": s}))

        async def new_session() -> None:
            res = make_reservation()

            def build():
                with gpu_lock:  # taken in the worker thread, never on the event loop
                    return agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS,
                                             backend=HotelBackend(res))

            conv["session"] = await asyncio.to_thread(build)
            conv["listener"] = LiveListener.for_agent(agent)
            await say(f"Reservation: {res}")
            await set_state("listening")

        async def run_turn(wav: np.ndarray) -> None:
            try:
                await stream_reply(wav)
            except (WebSocketDisconnect, RuntimeError):
                pass  # the page was closed mid-reply

        async def stream_reply(wav: np.ndarray) -> None:
            q: asyncio.Queue = asyncio.Queue()

            def worker():
                try:
                    with gpu_lock:
                        for ev in conv["session"].respond(wav):
                            loop.call_soon_threadsafe(q.put_nowait, ev)
                except Exception as e:  # noqa: BLE001 - report to the page, keep the server alive
                    loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "text": repr(e)})
                loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()
            spoke = False
            while (ev := await q.get()) is not None:
                t = ev["type"]
                if t == "audio":
                    if not spoke:
                        spoke = True
                        await set_state("speaking")
                    wav_out = ev["audio"] if ev.get("sample_rate", sr) == 24000 else resample(ev["audio"], ev["sample_rate"], 24000)
                    await ws.send_bytes(pcm16(wav_out))
                elif t == "user_transcript":
                    await say(f"USER (ctc): {ev['text']}")
                elif t == "assistant_text":
                    await say(f"ASSISTANT: {ev['text']}")
                elif t == "tool_call":
                    await say(f"TOOL CALL: {ev['call']}")
                elif t == "tool_result":
                    await say(f"TOOL RESULT: {ev['result']}")
                elif t == "timings":
                    await say("timings ms: " + ", ".join(f"{k}={v:.0f}" for k, v in ev["ms"].items()))
                elif t == "error":
                    await say(f"ERROR: {ev['text']}")
            await ws.send_text(json.dumps({"type": "reply_done"}))
            if not spoke:
                await set_state("speaking")  # the page answers "played" right away and listening resumes

        try:
            await new_session()
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if conv["state"] != "listening":
                        continue  # strict turn-taking: the mic is ignored while the agent thinks / speaks
                    chunk = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    chunk = resample(chunk, conv["mic_sr"], sr) if conv["mic_sr"] != sr else chunk

                    def feed(c=chunk):
                        with gpu_lock:
                            return conv["listener"].feed(c)

                    utterance = await asyncio.to_thread(feed)
                    if utterance is not None:
                        conv["turn"] += 1
                        await set_state("thinking")
                        await say(f"--- turn {conv['turn']} ({len(utterance) / sr:.1f}s, end: {conv['listener'].reason})")
                        asyncio.create_task(run_turn(utterance))
                    continue
                data = json.loads(msg.get("text") or "{}")
                if data.get("type") == "hello":
                    conv["mic_sr"] = int(data.get("sr", 48000))
                elif data.get("type") == "played":
                    conv["listener"].reset()
                    await set_state("listening")
                elif data.get("type") == "latency":
                    await say(f"first audio heard by the browser {data.get('first_audio_ms')} ms after the end of your turn")
                elif data.get("type") == "reset":
                    await new_session()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001 - print the reason instead of silently dropping the connection
            import traceback

            traceback.print_exc()
            raise

    return app


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--speech-llm-dir", default=None)
    p.add_argument("--talker-dir", default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--tunnel", action="store_true", help="public https URL via a Cloudflare quick tunnel")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = config_from_args(args)

    import uvicorn

    app = build_app(VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir))
    if args.tunnel:
        start_tunnel(args.port)
    print(f"serving on http://localhost:{args.port}  (use --tunnel for a public https link)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
