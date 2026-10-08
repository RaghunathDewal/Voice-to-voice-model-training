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
{"type": "session", "tools": [...], "reservation": {...} | null}, {"type": "vad", "speech": bool}
(the listener heard speech start / a noise burst was dropped), {"type": "user", "text"},
{"type": "assistant", "text"}, {"type": "tool", "call", "result"}, {"type": "error", "text"},
binary int16 24 kHz reply audio, {"type": "reply_done"} (always sent, even when the turn fails).

Raw PCM on /device (ESP32 boards, phone-call bridges, simple test consoles): binary int16 mono
mic audio in (16 kHz, or ?input=<rate>), binary int16 reply audio out (24 kHz, or ?output=16000),
the text "__TURN_COMPLETE__" once the reply has finished playing, and "you: ..." / "agent: ..."
transcript lines (?text=0 turns them off). The mic may stream all the time: it is ignored while
the agent thinks and speaks (half duplex), so no echo cancellation is needed on the device.

The page is s2s/cli/static/live.html; /?demo=<dir> serves the showcase page (showcase.html) that
replays a pre-generated conversation for the demo video.

/live is the application API (system prompt + tools from the application, tool calls executed by
it, Gemini-Live-like events): see s2s/cli/live_api.py and integrations/nestjs/.
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
from s2s.data.hotel_v3 import GenericBackend, known_tools, select_tools  # noqa: F401  (re-exported)
from s2s.runtime.agent import VoiceAgent
from s2s.runtime.live import LiveListener

STATIC = os.path.join(os.path.dirname(__file__), "static")
PAGE = open(os.path.join(STATIC, "live.html"), encoding="utf-8").read()
SHOWCASE = open(os.path.join(STATIC, "showcase.html"), encoding="utf-8").read()


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


def load_prompt(path: str) -> str:
    """A system prompt file. `{{` / `}}` (template escaping) become single braces."""
    with open(path, encoding="utf-8") as f:
        return f.read().replace("{{", "{").replace("}}", "}").strip()


def save_turn(save_dir: str, wav: np.ndarray, sr: int, info: dict) -> str:
    """Write one received user turn (what the model actually heard) as WAV + JSON, for debugging."""
    import soundfile as sf

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, f"turn_{len([f for f in os.listdir(save_dir) if f.endswith('.wav')]):04d}")
    sf.write(stem + ".wav", wav, sr)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    return stem


def build_app(agent: VoiceAgent | None, system_prompt: str | None = None, tools: list | None = HOTEL_TOOLS,
              demo_dir: str | None = None, save_turns: str | None = None, live_token: str | None = None) -> FastAPI:
    """system_prompt: replaces the default prompt AND the generated demo reservation (put the guest data in it)."""
    sr = agent.codec.sample_rate if agent is not None else 24000
    gpu_lock = threading.Lock()  # one model call at a time across connections
    app = FastAPI()

    @app.get("/")
    def index(demo: str | None = None):
        return HTMLResponse(SHOWCASE if demo else PAGE)

    if demo_dir:  # /?demo=demo replays a pre-generated conversation (conversation.json + audio/)
        from fastapi.staticfiles import StaticFiles

        app.mount("/demo", StaticFiles(directory=demo_dir), name="demo")

    from s2s.cli.live_api import add_live_api

    add_live_api(app, agent, gpu_lock, live_token)  # /live: the model as an upstream API for an application

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await converse(ws, device=False)

    @app.websocket("/device")
    async def device_endpoint(ws: WebSocket, input: int = 16000, output: int = 24000, text: int = 1):
        """Raw PCM for devices and simple clients (ESP32, the Gemini-bridge test console): binary int16 mono
        mic audio in at `input` Hz, binary int16 reply audio out at `output` Hz, the text "__TURN_COMPLETE__"
        once the reply has finished PLAYING (the client may flush its buffer then), and, with text=1, the
        transcripts as plain "you: ..." / "agent: ..." lines. No hello / played messages needed."""
        await converse(ws, device=True, in_sr=int(input), out_sr=int(output), text_lines=bool(text))

    async def converse(ws: WebSocket, device: bool, in_sr: int = 48000, out_sr: int = 24000,
                       text_lines: bool = True) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        conv: dict = {"mic_sr": in_sr, "state": "listening", "turn": 0}

        async def say(text: str) -> None:
            print(text, flush=True)
            if not device:
                await ws.send_text(json.dumps({"type": "log", "text": text}))

        async def send(kind: str, **data) -> None:
            if not device:
                await ws.send_text(json.dumps({"type": kind, **data}, default=str))
            elif text_lines and kind in ("user", "assistant") and data.get("text"):
                await ws.send_text(f"{'you' if kind == 'user' else 'agent'}: {data['text']}")

        async def set_state(s: str) -> None:
            conv["state"] = s
            if not device:
                await ws.send_text(json.dumps({"type": "state", "state": s}))

        async def reply_finished(play_end: float) -> None:
            """Browser: it answers "played" after playing. Device: wait until the reply has played out
            (audio is generated faster than real time), then tell it and listen again."""
            if not device:
                await ws.send_text(json.dumps({"type": "reply_done"}))
                return
            await asyncio.sleep(max(0.0, play_end - loop.time()))
            await ws.send_text("__TURN_COMPLETE__")
            conv["listener"].reset()
            await set_state("listening")

        async def new_session() -> None:
            res = make_reservation()

            def build():
                with gpu_lock:  # taken in the worker thread, never on the event loop
                    if system_prompt:
                        return agent.new_session(system_prompt=system_prompt, tools=tools, backend=GenericBackend(tools))
                    return agent.new_session(context=reservation_context(res), tools=tools, backend=HotelBackend(res))

            conv["session"] = await asyncio.to_thread(build)
            conv["listener"] = LiveListener.for_agent(agent)
            await send("session", tools=[t["function"]["name"] for t in tools or []],
                       reservation=None if system_prompt else res)
            if system_prompt:
                await say(f"Custom system prompt ({len(system_prompt)} chars); tools: "
                          f"{[t['function']['name'] for t in tools or []]}")
            else:
                await say(f"Reservation: {res}")
            await set_state("listening")

        async def run_turn(wav: np.ndarray) -> None:
            info = {}
            try:
                info = await stream_reply(wav)
            except (WebSocketDisconnect, RuntimeError):
                return  # the page was closed mid-reply
            except Exception as e:  # noqa: BLE001 - never leave the page stuck in "thinking"
                import traceback

                traceback.print_exc()
                try:
                    await send("error", text=repr(e))
                    await set_state("speaking")  # the page answers "played" and listening resumes
                    await reply_finished(loop.time())
                except (WebSocketDisconnect, RuntimeError):
                    return
            if save_turns:
                try:
                    stem = save_turn(save_turns, wav, sr, info)
                    print(f"saved {stem}.wav", flush=True)
                except OSError as e:
                    print(f"could not save the turn: {e}", flush=True)

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
            spoke, last_call = False, None
            play_end = loop.time()  # when the client will have played everything sent so far
            info = {"seconds": round(len(wav) / sr, 2), "end": conv["listener"].reason,
                    "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)), 1)}
            while (ev := await q.get()) is not None:
                t = ev["type"]
                if t == "audio":
                    if not spoke:
                        spoke = True
                        await set_state("speaking")
                    rate = ev.get("sample_rate", sr)
                    wav_out = ev["audio"] if rate == out_sr else resample(ev["audio"], rate, out_sr)
                    await ws.send_bytes(pcm16(wav_out))
                    play_end = max(play_end, loop.time() + 0.15) + len(wav_out) / out_sr  # + client jitter buffer
                elif t == "user_transcript":
                    info["ctc"] = ev["text"]
                    await say(f"USER (ctc): {ev['text']}")
                    if ev.get("asr") is not None:
                        info["asr"], info["asr_confidence"] = ev["asr"], round(float(ev["asr_confidence"]), 3)
                        info["unclear"] = bool(ev.get("unclear"))
                        await say(f"USER (parakeet, confidence {ev['asr_confidence']:.2f}"
                                  f"{', unclear: asking again' if ev.get('unclear') else ''}): {ev['asr']}")
                    # the page shows the best transcript we have: Parakeet's own ASR head, else the adapter's CTC
                    await send("user", text=ev.get("asr") or ev["text"], ctc=ev["text"])
                elif t == "assistant_text":
                    info["reply"] = ev["text"]
                    await say(f"ASSISTANT: {ev['text']}")
                    await send("assistant", text=ev["text"])
                elif t == "tool_call":
                    last_call = ev["call"]
                    await say(f"TOOL CALL: {ev['call']}")
                elif t == "tool_result":
                    await say(f"TOOL RESULT: {ev['result']}")
                    await send("tool", call=last_call, result=ev["result"])
                elif t == "timings":
                    await say("timings ms: " + ", ".join(f"{k}={v:.0f}" for k, v in ev["ms"].items()))
                elif t == "error":
                    await say(f"ERROR: {ev['text']}")
                    await send("error", text=ev["text"])
            if not spoke:  # the page answers "played" right away and listening resumes
                await set_state("speaking")
            await reply_finished(play_end + 0.2)
            return info

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

                    was_in_turn = conv["listener"].in_turn
                    utterance = await asyncio.to_thread(feed)
                    if utterance is None and conv["listener"].in_turn != was_in_turn:
                        await send("vad", speech=conv["listener"].in_turn)  # "I hear you" / noise dropped
                    if utterance is not None:
                        conv["turn"] += 1
                        await set_state("thinking")
                        await say(f"--- turn {conv['turn']} ({len(utterance) / sr:.1f}s, end: {conv['listener'].reason})")
                        asyncio.create_task(run_turn(utterance))
                    continue
                try:
                    data = json.loads(msg.get("text") or "{}")
                except ValueError:
                    data = {}  # devices may send plain-text pings / notes: ignored
                if not isinstance(data, dict):
                    data = {}
                if data.get("type") == "hello":
                    conv["mic_sr"] = int(data.get("sr", 48000))
                    await say(f"mic sample rate reported by the browser: {conv['mic_sr']} Hz")
                elif data.get("type") == "played":
                    if conv["state"] == "speaking":  # late or duplicate "played" messages are ignored
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
    p.add_argument("--system-prompt-file", default=None,
                   help="your own system prompt (with the guest/reservation data inside); replaces the demo one")
    p.add_argument("--demo-dir", default=None, help="serve a pre-generated conversation at /?demo=demo")
    p.add_argument("--tools", default="all", help="all | none | comma-separated built-in names, e.g. "
                                                  "order_product,create_issue (the v1 hotel tools and the hotel_v3 pool)")
    p.add_argument("--tools-file", default=None, help="JSON list of your own tool schemas (overrides --tools)")
    p.add_argument("--live-token", default=os.environ.get("S2S_LIVE_TOKEN"),
                   help="require this token on /live (?token= or Authorization: Bearer); default $S2S_LIVE_TOKEN")
    p.add_argument("--voice", choices=["talker", "kokoro"], default=None,
                   help="reply voice (default: runtime.voice from the config); kokoro needs: pip install kokoro misaki[en]")
    p.add_argument("--save-turns", default=None,
                   help="debug: save every user turn as heard by the server (WAV + JSON: length, end reason, level)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = config_from_args(args)
    if args.voice:
        cfg.runtime.voice = args.voice

    import uvicorn

    prompt = load_prompt(args.system_prompt_file) if args.system_prompt_file else None
    app = build_app(VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir), prompt, select_tools(args.tools, args.tools_file), args.demo_dir,
                    args.save_turns, args.live_token)
    if args.tunnel:
        start_tunnel(args.port)
    print(f"serving on http://localhost:{args.port}  (use --tunnel for a public https link)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
