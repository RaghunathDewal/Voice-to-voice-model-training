"""Browser demo (works inside Colab/Kaggle with a public share link).

    pip install gradio
    python -m s2s.cli.gradio_app --share

Record a turn with the microphone, press Submit, hear the reply. The session
(KV cache, reservation, tool state) persists until you press "New session".
"""

from __future__ import annotations

import numpy as np

from s2s.audio import resample, to_mono
from s2s.cli_common import base_parser, config_from_args
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.runtime.agent import VoiceAgent


def main() -> None:
    import gradio as gr

    p = base_parser(__doc__)
    p.add_argument("--share", action="store_true")
    p.add_argument("--speech-llm-dir", default=None)
    p.add_argument("--talker-dir", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    agent = VoiceAgent(cfg, args.speech_llm_dir, args.talker_dir)
    sr = agent.codec.sample_rate

    def new_session():
        res = make_reservation()
        return {"session": agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS,
                                             backend=HotelBackend(res)), "log": [f"Reservation: {res}"]}

    def on_turn(audio, state):
        if state is None:
            state = new_session()
        if audio is None:
            return None, "\n".join(state["log"]), state
        in_sr, data = audio
        data = np.asarray(data)
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        data = to_mono(data)
        wav = resample(data.astype(np.float32), int(in_sr), sr)
        chunks = []
        for ev in state["session"].respond(wav):
            t = ev["type"]
            if t == "user_transcript":
                state["log"].append(f"USER (ctc): {ev['text']}")
            elif t == "assistant_text":
                state["log"].append(f"ASSISTANT: {ev['text']}")
            elif t == "tool_call":
                state["log"].append(f"TOOL CALL: {ev['call']}")
            elif t == "tool_result":
                state["log"].append(f"TOOL RESULT: {ev['result']}")
            elif t == "audio":
                chunks.append(ev["audio"])
            elif t == "timings":
                state["log"].append("timings ms: " + ", ".join(f"{k}={v:.0f}" for k, v in ev["ms"].items()))
        reply = (sr, np.concatenate(chunks)) if chunks else None
        return reply, "\n".join(state["log"]), state

    with gr.Blocks(title="Speech-to-speech hotel agent") as demo:
        state = gr.State(None)
        mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your turn")
        btn = gr.Button("Submit turn")
        reset = gr.Button("New session")
        out_audio = gr.Audio(label="Reply", autoplay=True)
        log = gr.Textbox(label="Log", lines=16)
        btn.click(on_turn, [mic, state], [out_audio, log, state])
        reset.click(lambda: (None, "", None), None, [out_audio, log, state])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
