"""Hands-free browser demo: just talk. No record / submit buttons.

    pip install "gradio>=5"
    python -m s2s.cli.gradio_live --share

Press the microphone once to start. The end-of-turn detector (energy VAD + the
adapter's end-of-turn head) decides when you have finished speaking, the reply
is streamed back while it is generated, and listening resumes when it has been
played. Turn-taking is strict: while the agent speaks, the microphone is
ignored, so it cannot interrupt or hear itself. Headphones still help.

Per browser tab: one session (KV cache, reservation, tool state) until you
press "New session".
"""

import io
import threading
import time
import wave

import gradio as gr  # module level: gr.Request annotations must resolve for Gradio to inject the request
import numpy as np

from s2s.audio import resample, to_mono
from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.runtime.agent import VoiceAgent
from s2s.runtime.live import LiveListener

PLAYBACK_MARGIN_S = 0.6  # extra mute after the reply should have finished playing in the browser
FIRST_CHUNK_S = 0.16     # send the first audio as soon as it exists ...
CHUNK_S = 0.48           # ... then in larger pieces (fewer, smoother stream segments)


def wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def mic_to_24k(audio, sr_out: int) -> np.ndarray:
    in_sr, data = audio
    data = np.asarray(data)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return resample(to_mono(data.astype(np.float32)), int(in_sr), sr_out)


class Conversation:
    def __init__(self, agent: VoiceAgent):
        self.agent = agent
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        res = make_reservation()
        self.session = self.agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS,
                                              backend=HotelBackend(res))
        self.listener = LiveListener.for_agent(self.agent)
        self.log = [f"Reservation: {res}"]
        self.pending: np.ndarray | None = None
        self.busy = False
        self.mute_until = 0.0
        self.turns = 0


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--share", action="store_true")
    p.add_argument("--speech-llm-dir", default=None)
    p.add_argument("--talker-dir", default=None)
    p.add_argument("--stream-every", type=float, default=0.24, help="seconds of microphone audio per update")
    args = p.parse_args()
    cfg = config_from_args(args)
    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    sr = agent.codec.sample_rate
    gpu_lock = threading.Lock()  # one model call at a time across browser tabs
    conversations: dict[str, Conversation] = {}

    def conv_for(request: gr.Request) -> Conversation:
        key = request.session_hash or "default"
        if key not in conversations:
            with gpu_lock:
                conversations[key] = Conversation(agent)
        return conversations[key]

    def status_text(c: Conversation) -> str:
        if c.busy:
            return "### 🗣️ Agent is answering…"
        if time.time() < c.mute_until:
            return "### 🔈 Playing the reply…"
        if c.listener.in_turn:
            return "### 👂 Hearing you…"
        return "### 🎙️ Listening — just speak"

    def on_chunk(audio, turn_id, request: gr.Request):
        c = conv_for(request)
        if audio is None:
            return turn_id, status_text(c)
        with c.lock:
            if c.busy or time.time() < c.mute_until:
                c.listener.reset()  # strict turn-taking: ignore the mic while the agent talks
                return turn_id, status_text(c)
            with gpu_lock:
                utterance = c.listener.feed(mic_to_24k(audio, sr))
            if utterance is None:
                return turn_id, status_text(c)
            c.pending = utterance
            c.busy = True
            c.turns += 1
            c.log.append(f"--- turn {c.turns} ({len(utterance) / sr:.1f}s, end: {c.listener.reason})")
            return c.turns, status_text(c)

    def on_reply(turn_id, request: gr.Request):
        c = conv_for(request)
        with c.lock:
            wav, c.pending = c.pending, None
        if wav is None:
            return
        yield None, "\n".join(c.log), status_text(c)
        buf: list[np.ndarray] = []
        sent_s, first_sent_at = 0.0, None
        try:
            with gpu_lock:
                for ev in c.session.respond(wav):
                    t = ev["type"]
                    if t == "user_transcript":
                        c.log.append(f"USER (ctc): {ev['text']}")
                    elif t == "assistant_text":
                        c.log.append(f"ASSISTANT: {ev['text']}")
                    elif t == "tool_call":
                        c.log.append(f"TOOL CALL: {ev['call']}")
                    elif t == "tool_result":
                        c.log.append(f"TOOL RESULT: {ev['result']}")
                    elif t == "timings":
                        c.log.append("timings ms: " + ", ".join(f"{k}={v:.0f}" for k, v in ev["ms"].items()))
                    elif t == "audio":
                        buf.append(ev["audio"])
                        have = sum(len(x) for x in buf) / sr
                        if have >= (FIRST_CHUNK_S if first_sent_at is None else CHUNK_S):
                            first_sent_at = first_sent_at or time.time()
                            sent_s += have
                            yield wav_bytes(np.concatenate(buf), sr), "\n".join(c.log), status_text(c)
                            buf = []
                        continue
                    yield gr.skip(), "\n".join(c.log), status_text(c)
            if buf:
                first_sent_at = first_sent_at or time.time()
                sent_s += sum(len(x) for x in buf) / sr
                yield wav_bytes(np.concatenate(buf), sr), "\n".join(c.log), status_text(c)
        finally:
            playback_end = (first_sent_at or time.time()) + sent_s
            c.mute_until = max(playback_end, time.time()) + PLAYBACK_MARGIN_S
            c.busy = False
        yield gr.skip(), "\n".join(c.log), status_text(c)

    def on_new_session(request: gr.Request):
        c = conv_for(request)
        with c.lock, gpu_lock:
            c.reset()
        return None, "\n".join(c.log), status_text(c)

    def show_log(request: gr.Request):
        return "\n".join(conv_for(request).log)

    with gr.Blocks(title="Live hotel voice agent") as demo:
        gr.Markdown("## Live hotel voice agent\nClick the microphone **once**, allow access, then just talk. "
                    "Pause when you are done; the agent answers out loud. Use headphones if you can.")
        status = gr.Markdown("### 🎙️ Press the microphone to start")
        mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
        reply = gr.Audio(label="Agent", streaming=True, autoplay=True, interactive=False)
        turn_id = gr.Number(value=0, visible=False)
        log = gr.Textbox(label="Log", lines=16)
        new = gr.Button("New session")
        mic.stream(on_chunk, [mic, turn_id], [turn_id, status], stream_every=args.stream_every,
                   time_limit=None, show_progress="hidden")
        turn_id.change(on_reply, [turn_id], [reply, log, status], show_progress="hidden")
        new.click(on_new_session, None, [reply, log, status])
        demo.load(show_log, None, log)
    demo.queue().launch(share=args.share)


if __name__ == "__main__":
    main()
