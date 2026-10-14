"""Integration tests on tiny random Qwen3 + Mimi models (CPU, ~1 min)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from s2s.config import load_config
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.models.adapter import SpeechAdapter
from s2s.models.codec import MimiCodec, StreamingDecoder
from s2s.models.speech_llm import assemble_inputs, talker_features, target_lm_loss
from s2s.models.talker import Talker
from s2s.models.thinker import Thinker
from s2s.utils import save_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_codec_batching_and_streaming_decode(tiny_models, cpu):
    codec = MimiCodec(tiny_models["mimi"], cpu, 8)
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(24000).astype(np.float32) * 0.1, rng.standard_normal(9000).astype(np.float32) * 0.1
    batched = codec.encode_latents([a, b])
    single = codec.encode_latents([b])[0]
    assert batched[1].shape == single.shape == (codec.num_frames(9000), codec.latent_dim)
    assert torch.allclose(batched[1], single, atol=1e-4)  # causal: padding does not leak
    codes = codec.encode_codes([a])[0]
    full = codec.decode(codes)
    dec = StreamingDecoder(codec)  # default 4 frames of conv context: exact (transformer KV cache)
    frames = [codes[:, i] for i in range(codes.shape[1])]
    chunks = [dec.push(frames[i:i + 3]) for i in range(0, len(frames), 3)]
    assert np.allclose(np.concatenate(chunks), full, atol=1e-4)


def test_prompt_layout(tiny_models):
    from s2s.models.thinker import PromptBuilder

    pb = PromptBuilder.from_pretrained(tiny_models["qwen"])
    pre, suf = pb.prompt_parts("SYS", HOTEL_TOOLS, None)
    assert pb.tokenizer.decode(pre).endswith("<|im_start|>user\n")
    assert pb.tokenizer.decode(suf) == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    call = {"name": "order_product", "arguments": {"product": "towel", "quantity": 2}}
    assert pb.target_text("SYS", HOTEL_TOOLS, tool_calls=[call]) == (
        '<tool_call>\n{"name": "order_product", "arguments": {"product": "towel", "quantity": 2}}\n</tool_call><|im_end|>')
    assert pb.tokenizer.decode(pb.tool_response_ids([{"ok": True}])).startswith("\n<|im_start|>user\n<tool_response>")


def test_text_sft_conversation_matches_runtime_layout(tiny_models):
    """Thinker text fine-tuning: tool call + tool result + spoken reply, laid out exactly as the runtime feeds it."""
    from s2s.data.hotel_v2 import generate_examples_v2
    from s2s.models.thinker import PromptBuilder
    from s2s.prep.thinker_sft_data import hotel_conversation

    pb = PromptBuilder.from_pretrained(tiny_models["qwen"])
    row = next(r for r in generate_examples_v2(80, seed=0) if r.get("tool_calls") and r.get("history"))
    conv = hotel_conversation(row, "SYS")
    ids, labels = pb.conversation_ids(conv["messages"], conv["tools"])
    msgs = conv["messages"]
    system, history, user, tool_msg = msgs[0]["content"], msgs[1:-4], msgs[-4]["content"], msgs[-2]["content"]
    runtime = (pb.text_prompt_ids(system, user, conv["tools"], history)
               + pb.target_ids(system, conv["tools"], tool_calls=row["tool_calls"], history=history)
               + pb.tool_response_ids([json.loads(tool_msg)]))
    assert ids[: len(runtime)] == runtime
    trained = pb.tokenizer.decode([i for i, y in zip(ids, labels) if y != -100])
    assert trained.startswith("<tool_call>") and trained.endswith(row["reply_after_tool"] + "<|im_end|>")
    assert "<tool_response>" not in trained and "SYS" not in trained


def test_text_sft_v3_uses_the_rows_own_prompt_and_tools(tiny_models):
    from s2s.data.hotel_v3 import generate_examples_v3
    from s2s.models.thinker import PromptBuilder
    from s2s.prep.thinker_sft_data import hotel_conversation

    pb = PromptBuilder.from_pretrained(tiny_models["qwen"])
    rows = generate_examples_v3(200, seed=0)
    row = next(r for r in rows if r.get("tool_calls"))
    conv = hotel_conversation(row, "DEFAULT PROMPT")
    assert conv["messages"][0]["content"] == row["system"] and conv["tools"] == row["tools"]
    ids, labels = pb.conversation_ids(conv["messages"], conv["tools"])
    trained = pb.tokenizer.decode([i for i, y in zip(ids, labels) if y != -100])
    assert row["tool_calls"][0]["name"] in trained and trained.endswith(row["reply_after_tool"] + "<|im_end|>")
    plain = next(r for r in rows if not r["tools"])  # no tools offered: plain reply, no tool section
    conv = hotel_conversation(plain, "DEFAULT PROMPT")
    ids, labels = pb.conversation_ids(conv["messages"], conv["tools"])
    assert "<tools>" not in pb.tokenizer.decode(ids)


def test_thinker_chat_runs_a_turn(tiny_models, cpu):
    from s2s.data.hotel_v3 import GenericBackend, select_tools
    from s2s.eval.thinker_chat import respond

    th = Thinker(tiny_models["qwen"], cpu, torch.float32)
    tools = select_tools("order_product,create_issue")
    messages = [{"role": "system", "content": "You are the voice assistant for Test Inn."},
                {"role": "user", "content": "Can I get two towels?"}]
    lines = respond(th, messages, tools, GenericBackend(tools))
    assert lines and messages[-1]["role"] == "assistant"       # random tiny model: any reply, no crash


def test_labels_align_with_targets(tiny_models, cpu):
    th = Thinker(tiny_models["qwen"], cpu, torch.float32)
    ad = SpeechAdapter(64, th.hidden_size, d_model=32, n_layers=1, n_heads=4)
    speech = ad(torch.randn(2, 9, 64))["embeds"]
    pre, suf = th.prompts.prompt_parts("SYS", None, None)
    tgt = [th.prompts.target_ids("SYS", None, content="Hi there."), th.prompts.target_ids("SYS", None, content="Ok.")]
    emb, mask, labels = assemble_inputs(th, ad, speech, torch.tensor([9, 5]), [pre, pre], [suf, suf], tgt)
    for i, n in enumerate([9, 5]):
        length = len(pre) + 1 + n + 1 + len(suf) + len(tgt[i])
        assert int(mask[i].sum()) == length
        assert labels[i][labels[i] != -100].tolist() == tgt[i]
        assert labels[i, length - len(tgt[i]) - 1] == -100
    # memory-light loss (logits only at target positions) == Hugging Face's full-sequence loss
    reference = th.model(inputs_embeds=emb, attention_mask=mask, labels=labels).loss
    assert torch.allclose(target_lm_loss(th, emb, mask, labels), reference, atol=1e-5)
    # same with a LoRA-wrapped thinker
    lora = Thinker(tiny_models["qwen"], cpu, torch.float32,
                   new_lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]})
    ref2 = lora.model(inputs_embeds=emb, attention_mask=mask, labels=labels).loss
    assert torch.allclose(target_lm_loss(lora, emb, mask, labels), ref2, atol=1e-5)


def test_talker_features_pairing(tiny_models, cpu):
    """Feature for response token i must be the hidden state that predicted it."""
    th = Thinker(tiny_models["qwen"], cpu, torch.float32)
    prefix = th.prompts.text_prompt_ids("SYS", "hello")
    resp = th.prompts.ids("Your towels are on the way.")
    layers = th.hidden_layer_indices([0.5, 1.0])
    toks, hids = talker_features(th, prefix, [resp, resp[:3]], layers)
    for i in (0, 2, len(resp) - 1):
        out = th.model(input_ids=torch.tensor([prefix + resp[:i]]), output_hidden_states=True)
        expect = torch.stack([out.hidden_states[j][0, -1] for j in layers])
        assert torch.allclose(hids[0][i], expect, atol=1e-4)
    assert torch.allclose(hids[1], hids[0][:3], atol=1e-4)  # padding does not change features
    assert torch.allclose(toks[0], th.embed(torch.tensor(resp)))


@pytest.fixture()
def agent(tiny_models, tmp_path, cpu):
    th = Thinker(tiny_models["qwen"], cpu, torch.float32)
    speech_dir, talker_dir = tmp_path / "speech", tmp_path / "talker"
    os.makedirs(speech_dir)
    os.makedirs(talker_dir)
    ad = SpeechAdapter(64, th.hidden_size, d_model=32, n_layers=1, n_heads=4)
    ad.init_scale(th.text_embedding_rms())
    ad.save(str(speech_dir / "adapter.pt"))
    layers = th.hidden_layer_indices([0.5, 1.0])
    Talker(th.hidden_size, len(layers), num_codebooks=8, card=64, d_model=32, n_layers=1, n_heads=4,
           depth_d_model=16, depth_layers=1, depth_heads=2, max_frames=30).save(str(talker_dir / "talker.pt"))
    save_json(str(talker_dir / "meta.json"), {"thinker": tiny_models["qwen"], "layer_idx": layers})
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"), [
        f"codec.model={tiny_models['mimi']}", f"thinker.model={tiny_models['qwen']}", "device=cpu",
        "runtime.endpoint.vad=energy",  # synthetic tones stand in for speech here; Silero rightly ignores them
        "runtime.spec_lookup=false"])  # tests script the thinker's choices; the prompt-lookup test turns it on
    from s2s.runtime.agent import VoiceAgent

    return VoiceAgent(cfg, str(speech_dir), str(talker_dir))


def test_runtime_tool_loop_and_audio(agent, monkeypatch):
    import s2s.runtime.agent as agent_mod

    p = agent.thinker.prompts
    call = '\n{"name": "order_product", "arguments": {"product": "towel", "quantity": 2}}\n'
    script = (p.ids("Sure, one moment.") + [p.tool_call_start_id] + p.ids(call) + [p.tool_call_end_id, p.im_end_id]
              + p.ids("Two towels are on the way.") + [p.im_end_id])
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)

    res = make_reservation()
    backend = HotelBackend(res)
    session = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=backend)
    wav = np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.05
    events = list(session.respond(wav))
    kinds = [e["type"] for e in events]
    texts = [e["text"] for e in events if e["type"] == "assistant_text"]
    assert texts == ["Sure, one moment.", "Two towels are on the way."]
    assert backend.orders and backend.orders[0]["quantity"] == 2
    results = [e["result"] for e in events if e["type"] == "tool_result"]
    assert results and results[0]["success"] is True
    audio = [e["audio"] for e in events if e["type"] == "audio"]
    assert audio and all(a.dtype == np.float32 for a in audio)
    assert kinds[-2:] == ["timings", "done"]
    assert session.past.get_seq_length() == session.n_tokens

    # second turn reuses the cache
    before = session.n_tokens
    script.extend(p.ids("Anything else?") + [p.im_end_id])
    list(session.respond(wav))
    assert session.n_tokens > before and session.past.get_seq_length() == session.n_tokens


def test_streaming_turn_matches_whole_utterance(agent):
    rng = np.random.default_rng(1)
    wav = np.concatenate([rng.standard_normal(24000).astype(np.float32) * 0.1, np.zeros(24000, np.float32)])
    res = make_reservation()
    s1 = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS)
    s2 = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS)
    turn = s1.stream_input()
    for i in range(0, len(wav), 1000):  # odd chunk size on purpose
        turn.feed(wav[i:i + 1000])
    whole = s2.a.codec.encode_latents([wav[: turn.latents.shape[0] * s2.a.codec.hop]])[0]
    assert torch.allclose(turn.latents, whole, atol=1e-4)
    assert s1.n_tokens == len(s1.prefix_ids) + 1 + whole.shape[0]


def test_ws_live_streams_a_turn(agent, monkeypatch):
    """Browser protocol end to end: mic PCM in, endpoint, streamed audio out, back to listening."""
    import json as _json

    from fastapi.testclient import TestClient

    import s2s.runtime.agent as agent_mod
    from s2s.cli.ws_live import build_app

    p = agent.thinker.prompts
    script = p.ids("Hello there.") + [p.im_end_id]
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    sr = agent.codec.sample_rate
    t = np.arange(sr) / sr
    speech = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    mic = np.concatenate([speech, np.zeros(2 * sr, dtype=np.float32)])
    pcm = (mic * 32767).astype("<i2")
    with TestClient(build_app(agent)).websocket_connect("/ws") as ws:
        ws.send_text(_json.dumps({"type": "hello", "sr": sr}))
        step = int(0.08 * sr)
        for i in range(0, len(pcm), step):
            ws.send_bytes(pcm[i:i + step].tobytes())
        states, logs, audio = [], [], 0
        while True:
            msg = ws.receive()
            if msg.get("bytes"):
                audio += len(msg["bytes"]) // 2
                continue
            m = _json.loads(msg["text"])
            if m["type"] == "state":
                states.append(m["state"])
            elif m["type"] == "log":
                logs.append(m["text"])
            elif m["type"] == "reply_done":
                break
        assert "thinking" in states and audio > 0
        assert any(line.startswith("ASSISTANT: Hello there.") for line in logs)
        ws.send_text(_json.dumps({"type": "played"}))
        while (m := _json.loads(ws.receive_text()))["type"] != "state":
            pass
        assert m["state"] == "listening"


def test_unclear_audio_gets_asked_again_without_tool_calls(agent, monkeypatch):
    """Parakeet's confidence below runtime.asr_min_confidence: a fixed 'didn't catch that' reply, no tool call,
    even if the thinker would have called a tool (garbled audio must never trigger an action)."""
    import s2s.runtime.agent as agent_mod

    class FakeParakeet:
        sample_rate = 24000

        def __init__(self, conf):
            self.conf = conf

        def encode(self, wavs):
            return agent.codec.encode_latents(wavs)

        def transcribe_with_confidence(self, feats):
            return [("order one ticket", self.conf)]

    p = agent.thinker.prompts
    call = '\n{"name": "order_product", "arguments": {"product": "towel", "quantity": 1}}\n'
    monkeypatch.setattr(agent, "input_encoder", FakeParakeet(0.3))
    agent.cfg.runtime.asr_min_confidence = 0.6
    script = [p.tool_call_start_id] + p.ids(call) + [p.tool_call_end_id, p.im_end_id]
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    res = make_reservation()
    backend = HotelBackend(res)
    session = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=backend)
    events = list(session.respond(np.zeros(24000, np.float32)))
    first = next(e for e in events if e["type"] == "user_transcript")
    assert first["unclear"] and first["asr"] == "order one ticket"
    assert [e["text"] for e in events if e["type"] == "assistant_text"] == ["Sorry, I didn't catch that. Could you say it again?"]
    assert not backend.orders and not any(e["type"] == "tool_call" for e in events)
    assert session.past.get_seq_length() == session.n_tokens

    monkeypatch.setattr(agent, "input_encoder", FakeParakeet(0.95))  # confident: normal behaviour
    events = list(session.respond(np.zeros(24000, np.float32)))
    assert backend.orders and any(e["type"] == "tool_call" for e in events)


def test_device_endpoint_raw_pcm(agent, monkeypatch):
    """/device: raw 16 kHz PCM in (no hello), raw PCM out at ?output=, transcript lines and __TURN_COMPLETE__
    after the reply has played, then the next turn works (like an ESP32 or the Gemini-bridge console)."""
    from fastapi.testclient import TestClient

    import s2s.runtime.agent as agent_mod
    from s2s.cli.ws_live import build_app

    p = agent.thinker.prompts
    script = (p.ids("Hello there.") + [p.im_end_id]) * 2
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    sr = 16000
    t = np.arange(sr) / sr
    mic = np.concatenate([0.1 * np.sin(2 * np.pi * 220 * t), np.zeros(2 * sr)]).astype(np.float32)
    pcm = (mic * 32767).astype("<i2")
    with TestClient(build_app(agent)).websocket_connect("/device?output=16000") as ws:
        for turn in range(2):
            for i in range(0, len(pcm), 2048):  # 128 ms chunks, like a ScriptProcessor(2048) at 16 kHz
                ws.send_bytes(pcm[i:i + 2048].tobytes())
            ws.send_text("ping")  # plain text from a device is ignored
            lines, audio = [], 0
            while True:
                msg = ws.receive()
                if msg.get("bytes"):
                    audio += len(msg["bytes"]) // 2
                    continue
                if msg["text"] == "__TURN_COMPLETE__":
                    break
                lines.append(msg["text"])
            assert audio > 0, f"turn {turn}: no audio"
            assert any(x.startswith("you: ") for x in lines) and "agent: Hello there." in lines
            assert not any(x.startswith("{") for x in lines)  # no JSON protocol messages on /device


def test_live_api_for_an_application_backend(agent, monkeypatch):
    """/live: setup with Gemini-style tool declarations, a text kickoff, then a spoken turn whose tool call is
    executed by the CLIENT (like a NestJS backend calling its own booking API) and answered with tool_response."""
    from fastapi.testclient import TestClient

    import s2s.runtime.agent as agent_mod
    from s2s.cli.ws_live import build_app

    p = agent.thinker.prompts
    call = '\n{"name": "place_product_order", "arguments": {"products": [{"productName": "Towel", "count": 2}]}}\n'
    script = (p.ids("Hi, how can I help?") + [p.im_end_id]
              + [p.tool_call_start_id] + p.ids(call) + [p.tool_call_end_id, p.im_end_id]
              + p.ids("Your towels are on the way.") + [p.im_end_id])
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    gemini_tools = [{"functionDeclarations": [
        {"name": "browse_products", "description": "List products."},  # no parameters at all
        {"name": "place_product_order", "description": "Order products.", "parameters": {
            "type": "OBJECT", "required": ["products"], "properties": {
                "message": {"type": "STRING"},
                "products": {"type": "ARRAY", "items": {"type": "OBJECT", "required": ["productName"], "properties": {
                    "productName": {"type": "STRING"}, "count": {"type": "NUMBER"}}}}}}}]}]
    client = TestClient(build_app(agent, live_token="secret"))

    def until(ws, kind):
        got, audio = [], 0
        while True:
            msg = ws.receive()
            if msg.get("bytes"):
                audio += len(msg["bytes"]) // 2
                continue
            m = json.loads(msg["text"])
            got.append(m)
            if m["type"] == kind:
                return got, audio

    with pytest.raises(Exception):  # wrong token: refused
        with client.websocket_connect("/live?token=nope") as ws:
            ws.receive()
    with client.websocket_connect("/live", headers={"Authorization": "Bearer secret"}) as ws:
        ws.send_text(json.dumps({"type": "setup", "system_prompt": "You are the concierge.", "tools": gemini_tools,
                                 "input_sample_rate": 16000, "output_sample_rate": 16000}))
        until(ws, "setup_complete")
        ws.send_text(json.dumps({"type": "text", "text": "Greet the guest."}))  # kickoff, like Gemini's sendClientContent
        got, audio = until(ws, "turn_complete")
        assert audio > 0 and {"type": "output_transcription", "text": "Hi, how can I help?"} in got
        assert not any(m["type"] == "input_transcription" for m in got)  # typed kickoffs are not transcribed

        sr = 16000
        t = np.arange(sr) / sr
        pcm = (np.concatenate([0.1 * np.sin(2 * np.pi * 220 * t), np.zeros(2 * sr)]) * 32767).astype("<i2")
        for i in range(0, len(pcm), 2048):
            ws.send_bytes(pcm[i:i + 2048].tobytes())
        got, _ = until(ws, "tool_call")
        assert any(m["type"] == "input_transcription" for m in got)
        tc = got[-1]
        assert tc["name"] == "place_product_order" and tc["args"]["products"][0]["count"] == 2
        ws.send_text(json.dumps({"type": "tool_response", "id": tc["id"], "response": {"order_id": 77}}))
        got, audio = until(ws, "turn_complete")
        assert audio > 0 and {"type": "output_transcription", "text": "Your towels are on the way."} in got


def test_external_tts_speaks_the_reply_instead_of_the_talker(agent, monkeypatch):
    """runtime.tts_url: the reply text goes to a TTS server in phrases; its audio (own sample rate) is emitted."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import s2s.runtime.agent as agent_mod
    from s2s.runtime.external_tts import ExternalTTS

    spoken = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            spoken.append(_json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"])
            pcm = (np.full(1600, 0.1) * 32767).astype("<i2").tobytes()
            self.send_response(200)
            self.send_header("X-Sample-Rate", "16000")
            self.send_header("Content-Length", str(len(pcm)))
            self.end_headers()
            self.wfile.write(pcm)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(agent, "tts", ExternalTTS(f"http://127.0.0.1:{srv.server_address[1]}/tts"))
    p = agent.thinker.prompts
    call = '\n{"name": "order_product", "arguments": {"product": "towel", "quantity": 2}}\n'
    script = (p.ids("Sure, one moment.") + [p.tool_call_start_id] + p.ids(call) + [p.tool_call_end_id, p.im_end_id]
              + p.ids("Two towels are on the way. Anything else?") + [p.im_end_id])
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    res = make_reservation()
    session = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=HotelBackend(res))
    try:
        events = list(session.respond(np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.05))
    finally:
        srv.shutdown()
    audio = [e for e in events if e["type"] == "audio"]
    assert spoken == ["Sure, one moment.", "Two towels are on the way.", "Anything else?"]
    assert len(audio) == 3 and all(e["sample_rate"] == 16000 and len(e["audio"]) == 1600 for e in audio)
    timings = next(e["ms"] for e in events if e["type"] == "timings")
    assert "first_audio_out" in timings and "tool_call" in timings


def test_warm_up_runs_a_throwaway_turn_and_keeps_the_tts(agent):
    agent.tts = sentinel = object()
    agent.warm_up(tokens=2)
    assert agent.tts is sentinel


def test_v3_calls_places_the_reply_after_the_detected_end_of_turn(agent, monkeypatch, tmp_path):
    """s2s.eval.v3_calls: a recorded call is fed as live; the reply lands after the detected end of the turn."""
    import json as _json
    import sys

    import soundfile as sf

    import s2s.runtime.agent as agent_mod
    from s2s.eval import v3_calls

    sr = 24000
    t = np.arange(sr) / sr
    wav = np.concatenate([np.zeros(sr // 2), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(int(sr * 3.5))]).astype(np.float32)
    sf.write(tmp_path / "c.wav", wav, sr)
    calls = [{"id": "c", "wav": str(tmp_path / "c.wav"), "prompt": "You work for a hotel.",
              "turns": [{"text": "what time is checkout", "start": 0.5, "end": 1.5, "keywords": ["11"]}]}]
    (tmp_path / "calls.json").write_text(_json.dumps(calls))
    p = agent.thinker.prompts
    script = p.ids("Checkout is at 11.") + [p.im_end_id]
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    monkeypatch.setattr(agent_mod, "VoiceAgent", lambda *a, **k: agent)
    monkeypatch.setattr(agent, "warm_up", lambda: None)
    monkeypatch.setattr(sys, "argv", ["v3_calls", "--config", os.path.join(ROOT, "configs", "default.yaml"),
                                      "--calls", str(tmp_path / "calls.json"), "--out-dir", str(tmp_path / "out")])
    v3_calls.main()
    run = _json.loads((tmp_path / "out" / "c.json").read_text())
    said = [x for x in run["text"] if x != "PAD"]
    assert len(said) == 1 and "Checkout is at 11." in said[0]
    start = run["text"].index(said[0]) * run["frame_s"]
    assert 1.5 < start < 4.0 and run["turns"][0]["end_detected_s"] > 1.5
    out, _ = sf.read(tmp_path / "out" / "c.wav")
    assert np.abs(out[: int(1.5 * sr)]).max() == 0 and np.abs(out).max() > 0


def test_static_cache_thinker_matches_the_normal_path(agent):
    """runtime.cuda_graphs: the fixed-size cache path gives the same logits as the normal path (the graph replay
    itself needs a GPU and is checked against this path when recorded); one conversation owns the cache."""
    from s2s.runtime.cuda_graph import StaticThinker

    th = agent.thinker
    st = StaticThinker(th.model, agent.layer_idx, need_hidden=True, max_tokens=64, device=agent.device, dtype=th.dtype)
    assert st.capture(lambda ids: th.embed(torch.tensor(ids)), th.prompts.ids("hello there")) is False  # CPU
    ids = th.prompts.ids("Hello, welcome to the hotel. How can I help?")
    emb = th.embed(torch.tensor(ids))
    with torch.no_grad():
        ref = th.model(inputs_embeds=emb[None, :-3], use_cache=True)
        past, ref_logits = ref.past_key_values, []
        for i in range(len(ids) - 3, len(ids)):
            out = th.model(inputs_embeds=emb[None, i:i + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            ref_logits.append(out.logits[0, -1])
    owner = object()
    st.reset(owner)
    st.forward(owner, emb[:-3])
    got = [st.forward(owner, emb[i:i + 1])[0] for i in range(len(ids) - 3, len(ids))]
    for a, b in zip(got, ref_logits):
        assert torch.allclose(a, b, atol=1e-4)
    st.reset(object())  # a new conversation takes the cache: the old one cannot continue
    with pytest.raises(RuntimeError):
        st.forward(owner, emb[:1])
    # a whole session on the static path says the same as on the normal path
    p = th.prompts
    wav = np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.05
    texts = []
    for static in (None, StaticThinker(th.model, agent.layer_idx, True, 512, agent.device, th.dtype)):
        agent.static = static
        session = agent.new_session(tools=None)
        texts.append([e["text"] for e in session.respond(wav) if e["type"] == "assistant_text"])
    agent.static = None
    assert texts[0] == texts[1] and p is not None


def test_a_turn_with_no_heard_words_gets_no_reply_and_is_forgotten(agent, monkeypatch):
    """Noise that passed the voice detector: the adapter hears no words -> no reply, the thinker's memory is
    rolled back to before the turn (normal and fixed-size cache)."""
    import s2s.runtime.agent as agent_mod
    from s2s.runtime.cuda_graph import StaticThinker

    monkeypatch.setattr(agent_mod, "ctc_greedy_decode", lambda ids: "")
    th = agent.thinker
    wav = np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.05
    for static in (None, StaticThinker(th.model, agent.layer_idx, True, 512, agent.device, th.dtype)):
        agent.static = static
        session = agent.new_session(tools=None)
        before = session.n_tokens
        events = list(session.respond(wav))
        assert [e["type"] for e in events] == ["user_transcript", "timings", "done"] and events[0]["ignored"]
        assert session.n_tokens == before and session.first_turn and not session.history
        if static is None:
            assert session.past.get_seq_length() == before
        else:
            assert static.n == before
    agent.static = None


def test_prompt_lookup_gives_the_same_reply_in_fewer_thinker_passes(agent, monkeypatch):
    """runtime.spec_lookup: (1) the real (tiny) thinker says exactly the same with and without it; (2) a reply
    that copies the prompt's facts is produced in fewer thinker passes than tokens."""
    import s2s.runtime.agent as agent_mod

    wav = np.random.default_rng(1).standard_normal(24000).astype(np.float32) * 0.05
    prompt = "You are a hotel concierge. The Wi-Fi password is sunset2026 and checkout is at 11 AM."
    replies = []
    for on in (False, True):
        monkeypatch.setitem(agent.cfg.runtime, "spec_lookup", on)
        session = agent.new_session(system_prompt=prompt, tools=None)
        events = list(session.respond(wav))
        replies.append(([e["text"] for e in events if e["type"] == "assistant_text"],
                        next(e["ms"] for e in events if e["type"] == "timings")))
    assert replies[0][0] == replies[1][0]
    assert replies[0][1]["reply_tokens"] == replies[0][1]["thinker_passes"]  # off: one pass per token

    # a thinker whose (greedy) reply copies the prompt: guessed tokens are checked against it
    th = agent.thinker
    script = th.prompts.ids("The Wi-Fi password is sunset2026 and checkout is at 11 AM.") + [th.prompts.im_end_id]
    want = list(script)

    def next_choice(*a, **k):
        return script.pop(0) if script else th.prompts.im_end_id

    monkeypatch.setattr(agent_mod, "sample_logits", next_choice)
    monkeypatch.setattr(agent_mod.VoiceSession, "_verify",
                        lambda self, logits, guess, temp: (lambda c: (c == guess, c))(next_choice()))
    monkeypatch.setitem(agent.cfg.runtime, "spec_lookup", True)
    session = agent.new_session(system_prompt=prompt, tools=None)
    events = list(session.respond(wav))
    text = next(e["text"] for e in events if e["type"] == "assistant_text")
    ms = next(e["ms"] for e in events if e["type"] == "timings")
    assert text == th.tokenizer.decode(want[:-1]).strip()
    assert ms["reply_tokens"] == len(want) - 1 and ms["thinker_passes"] < ms["reply_tokens"] / 2
    # the memory holds exactly the reply (rejected guesses were taken back out)
    assert session.text_ids[-len(want):] == want and session.n_tokens == session.past.get_seq_length()


def test_speech_text_keeps_spaces_before_numbers_and_says_times_naturally(agent, monkeypatch):
    """Qwen writes " 3" as " " + "3": the bare space must reach the voice ("at 3 PM", not "at3"); "3:00 PM" is
    said "3 PM"; a phrase never ends right after a number ("3:00 | PM")."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import s2s.runtime.agent as agent_mod
    from s2s.runtime.external_tts import ExternalTTS, clean_for_speech

    assert clean_for_speech("Check-in is at 3:00 PM.") == "Check-in is at 3 PM."
    assert clean_for_speech("Checkout 11:00 a.m. on Friday, gym 7:30pm, quiet hours 22:00.") == \
        "Checkout 11 AM on Friday, gym 7:30 PM, quiet hours 22 o'clock."
    spoken = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            import json as _json

            spoken.append(_json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"])
            pcm = np.zeros(160, "<i2").tobytes()
            self.send_response(200)
            self.send_header("X-Sample-Rate", "16000")
            self.send_header("Content-Length", str(len(pcm)))
            self.end_headers()
            self.wfile.write(pcm)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(agent, "tts", ExternalTTS(f"http://127.0.0.1:{srv.server_address[1]}/tts"))
    p = agent.thinker.prompts
    script = p.ids("Your check-in is scheduled for 3:00 PM on December 24, 2026.") + [p.im_end_id]
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    session = agent.new_session(tools=None)
    try:
        list(session.respond(np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.05))
    finally:
        srv.shutdown()
    assert " ".join(spoken) == "Your check-in is scheduled for 3 PM on December 24, 2026."
    assert not any(x.rstrip().endswith(("3", "3:00")) for x in spoken[:-1])


def test_hybrid_input_puts_the_listener_transcript_after_the_speech(agent, monkeypatch, tmp_path):
    """Hybrid input (hybrid.enabled): (A) training rows get the transcript right after the speech, inside the
    guest's turn, sometimes left out in training; (B) the live agent sends the same text to the 4B and reports it
    as what the guest said; (C) the Gate 1 test runs with it and reports the listener's word error rate."""
    import json as _json
    import sys

    import s2s.models.nemotron_text as nt
    from s2s.data.datasets import SpeechLLMDataset
    from s2s.models.nemotron_text import transcript_ids

    th, tmpl = agent.thinker, "\n(speech recogniser heard: {text})"
    heard = transcript_ids(th.prompts, "what time is checkout", tmpl)
    lat = tmp_path / "a.npy"
    np.save(lat, np.random.default_rng(0).standard_normal((30, 64)).astype(np.float32))
    row = {"id": "r1", "latent": str(lat), "text": "what time is checkout", "response": "It is at 11 AM.",
           "asr": "what time is checkout"}
    common = dict(prompts=th.prompts, system_prompt="You are a concierge.", transcribe_instruction="Repeat.",
                  transcribe_prob=0.0, max_trailing_frames=0, max_frames=100, max_target_tokens=64)
    plain = SpeechLLMDataset([row], train=False, **common)[0]
    hybrid = SpeechLLMDataset([row], train=False, hybrid_template=tmpl, **common)[0]
    assert hybrid["suffix"] == heard + list(plain["suffix"])  # transcript first, then the turn is closed
    assert SpeechLLMDataset([row], train=True, hybrid_template=tmpl, hybrid_drop=1.0, **common)[0]["suffix"] \
        == plain["suffix"]  # left out in training (drop)
    no_asr = {k: v for k, v in row.items() if k != "asr"}
    assert SpeechLLMDataset([no_asr], train=False, hybrid_template=tmpl, **common)[0]["suffix"] == plain["suffix"]

    class FakeTranscriber:  # stands in for Nemotron's text decoder (checked in test_units against transformers)
        def __call__(self, feats):
            return ["what time is checkout" for _ in feats]

    monkeypatch.setattr(agent, "transcriber", FakeTranscriber(), raising=False)
    monkeypatch.setattr(agent, "hybrid_template", tmpl, raising=False)
    session = agent.new_session(tools=None)
    wav = np.random.default_rng(1).standard_normal(24000).astype(np.float32) * 0.05
    events = list(session.respond(wav))
    said = next(e for e in events if e["type"] == "user_transcript")
    assert said["asr"] == "what time is checkout"
    ids = session.text_ids
    assert any(ids[i:i + len(heard)] == heard for i in range(len(ids)))  # the 4B got the transcript
    monkeypatch.setattr(agent, "transcriber", None)

    # (C) the Gate 1 test in hybrid mode
    monkeypatch.setattr(nt.NemotronTranscriber, "load", classmethod(lambda cls, name, device: FakeTranscriber()))
    man = tmp_path / "m.jsonl"
    man.write_text(_json.dumps({k: v for k, v in no_asr.items() if k != "response"}) + "\n")
    from s2s.eval import speech_vs_text

    cfg_args = ["--config", os.path.join(ROOT, "configs", "default.yaml"), "--set",
                f"thinker.model={agent.cfg.thinker.model}", "device=cpu", "hybrid.enabled=true"]
    monkeypatch.setattr(sys, "argv", ["speech_vs_text", *cfg_args, "--speech-llm-dir", str(tmp_path / "speech"),
                                      "--manifest", str(man), "--name", "h", "--max", "1", "--max-new", "4",
                                      "--out-dir", str(tmp_path / "out")])
    speech_vs_text.main()
    res = _json.loads((tmp_path / "out" / "h.json").read_text())
    assert res["summary"]["hybrid"] and res["summary"]["listener_wer"] == 0.0
    assert res["rows"][0]["listener_text"] == "what time is checkout"


def test_duplex_graph_frame_matches_normal_path(tiny_models, cpu):
    """The CUDA-graph frame (static cache, depth without a KV cache, Gumbel sampling) gives the same greedy text
    token and codes as the normal session step (here run eagerly: no GPU to record on)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from s2s.data.duplex import PAD_TOKEN, prefix_ids
    from s2s.models.duplex import DuplexModel
    from s2s.runtime.duplex import DuplexSession
    from s2s.runtime.duplex_graph import DuplexGraph

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(tiny_models["qwen"])
    llm = AutoModelForCausalLM.from_pretrained(tiny_models["qwen"], dtype=torch.float32)
    m = DuplexModel(llm, feat_dim=16, pad_id=tok.convert_tokens_to_ids(PAD_TOKEN), num_codebooks=4, card=64,
                    depth_d_model=32, depth_layers=2, depth_heads=4).eval()
    sess = DuplexSession(m, tok, "You are a concierge.", None, audio_temperature=0.0)
    g = DuplexGraph(m, 256, temperature=0.0, top_k=0)
    assert not g.capture(sess._embed([1, 2]), None)  # CPU: no graph, the frame runs eagerly
    g.run(sess._embed(prefix_ids(tok, "You are a concierge.", None)))
    feats = torch.randn(6, 16)
    for t in range(6):
        prev_text, prev_col = sess.prev_text, sess.prev_col.clone()
        out = sess.step(feats[t])
        text, col, _ = g.frame(feats[t], prev_text, prev_col, None, greedy=True)
        assert int(text) == out["text"]
        ref = sess.grid[-1].masked_fill(sess.grid[-1] < 0, -1)
        assert torch.equal(col[ref >= 0], ref[ref >= 0])
    # sampling: Gumbel noise over the top-k picks only top-k codes
    gs = DuplexGraph(m, 256, temperature=0.8, top_k=5)
    gs.run(sess._embed([1, 2, 3]))
    gen = torch.Generator().manual_seed(1)
    _, col, h = gs.frame(feats[0], m.pad_id, torch.full((4,), 64), gen)
    assert col.min() >= 0 and col.max() < 64 and gs.noise.abs().sum() > 0


def _hf_greedy(model, embeds: torch.Tensor, steps: int) -> list[int]:
    """Reference: the plain Hugging Face model with its own cache, one conversation, greedy."""
    from transformers import DynamicCache

    past = DynamicCache()
    out = model(inputs_embeds=embeds[None], past_key_values=past, use_cache=True)
    toks = []
    for _ in range(steps):
        tok = int(out.logits[0, -1].argmax())
        toks.append(tok)
        emb = model.get_input_embeddings()(torch.tensor([[tok]]))
        out = model(inputs_embeds=emb, past_key_values=past, use_cache=True)
    return toks


def test_batched_thinker_matches_hf_model(tiny_models, cpu):
    """Three conversations of different lengths in one slot cache, decoded together: same logits and greedy
    tokens as each run alone through the HF model (rows never mix, positions per row)."""
    from transformers import AutoModelForCausalLM

    from s2s.runtime.batch_engine import BatchedThinker

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(tiny_models["qwen"], dtype=torch.float32).eval()
    emb = model.get_input_embeddings()
    prompts = [torch.randint(0, 500, (n,)) for n in (7, 13, 4)]
    eng = BatchedThinker(model, slots=4, max_tokens=64, device=cpu, dtype=torch.float32)
    slots = [eng.alloc() for _ in prompts]
    eng.release(slots[0])                       # a freed slot is reused: its old rows must not leak
    slots[0] = eng.alloc()
    first = []
    for s, p in zip(slots, prompts):
        with torch.no_grad():
            logits = eng.forward([s], emb(p)[None])
            ref = model(inputs_embeds=emb(p)[None]).logits[0, -1]
        assert torch.allclose(logits[0], ref, atol=1e-4)
        first.append(int(logits[0].argmax()))
    got = [[t] for t in first]
    toks = first
    for _ in range(5):
        with torch.no_grad():
            logits = eng.forward(slots[::-1], emb(torch.tensor(toks[::-1]))[:, None])  # any row order
        toks = logits.argmax(-1).tolist()[::-1]
        for g, t in zip(got, toks):
            g.append(t)
    with torch.no_grad():
        refs = [_hf_greedy(model, emb(p), 6) for p in prompts]
    assert got == refs


def test_batch_scheduler_serves_concurrent_calls(tiny_models, cpu):
    """Calls submitted at once (text + speech pieces, several turns): every reply equals the same conversation
    run alone, and decoding really was batched."""
    import threading

    from transformers import AutoModelForCausalLM

    from s2s.models.adapter import SpeechAdapter
    from s2s.runtime.batch_engine import BatchedThinker, BatchScheduler, Job

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(tiny_models["qwen"], dtype=torch.float32).eval()
    emb = model.get_input_embeddings()
    hidden = model.config.hidden_size
    adapter = SpeechAdapter(latent_dim=16, llm_dim=hidden, d_model=32, n_layers=1, n_heads=4).eval()
    start, end = adapter.boundary_embeddings()

    def embed(ids):
        return emb(torch.tensor(ids))

    eng = BatchedThinker(model, slots=6, max_tokens=256, device=cpu, dtype=torch.float32)
    sched = BatchScheduler(eng, embed, adapter, (start.detach(), end.detach()), eos_ids=set(), im_end_id=None,
                           prefill_budget=8)
    sched.start()
    calls = [([1, 2, 3, 4, 5], torch.randn(9, 16), [7, 8]), ([9, 10], torch.randn(5, 16), [11]),
             ([12, 13, 14], torch.randn(12, 16), [15, 16, 17])]
    results = {}
    lock = threading.Lock()

    def run(i, prefix, speech, close):
        conv = sched.open()
        done = threading.Event()
        toks = []
        job = Job(conv, [("ids", prefix), ("speech", speech), ("ids", close)], max_new=6,
                  on_token=lambda t, now: toks.append(t), on_done=lambda st: done.set())
        sched.submit(job)
        assert done.wait(60)
        with lock:
            results[i] = toks

    threads = [threading.Thread(target=run, args=(i, *c)) for i, c in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sched.shutdown()
    assert sched.error is None
    with torch.no_grad():
        for i, (prefix, speech, close) in enumerate(calls):
            x = torch.cat([embed(prefix), start[None], adapter(speech[None])["embeds"][0], end[None], embed(close)])
            assert results[i] == _hf_greedy(model, x, 6), i
    assert sched.stats["max_batch"] >= 2


def test_batch_load_test_runs_end_to_end(tiny_models, tmp_path, monkeypatch):
    """The load-test driver on tiny models (CPU): simulated hotel calls with prompts, tools and stored speech
    features; every turn answered, results and summary written."""
    import json
    import sys

    import numpy as np

    from s2s.data.hotel import HOTEL_TOOLS
    from s2s.eval import batch_load_test
    from s2s.models.adapter import SpeechAdapter
    from s2s.models.thinker import Thinker

    th = Thinker(tiny_models["qwen"], torch.device("cpu"), torch.float32)
    ckpt = tmp_path / "speech_llm"
    ckpt.mkdir()
    SpeechAdapter(latent_dim=16, llm_dim=th.hidden_size, d_model=32, n_layers=1, n_heads=4).save(str(ckpt / "adapter.pt"))
    rows = []
    for i in range(4):
        lat = tmp_path / f"lat{i}.npy"
        np.save(lat, np.random.default_rng(i).standard_normal((6 + i, 16)).astype(np.float16))
        rows.append({"latent": str(lat), "system": f"You are the concierge of hotel {i}.", "tools": HOTEL_TOOLS[:2]})
    man = tmp_path / "rows.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "batch_load_test", "--config", "configs/default.yaml", "--speech-llm-dir", str(ckpt), "--manifest", str(man),
        "--levels", "1,3", "--turns", "2", "--max-new", "5", "--speech-speed", "0", "--stagger", "0",
        "--max-tokens", "2048", "--device", "cpu", "--out", str(out),
        "--set", f"thinker.model={tiny_models['qwen']}", "thinker.dtype=float32"])
    batch_load_test.main()
    res = json.loads((out / "results.json").read_text())
    assert [lv["calls"] for lv in res["levels"]] == [1, 3]
    assert all(lv["n_errors"] == 0 for lv in res["levels"]), res["levels"]
    assert res["levels"][1]["turns"] == 6
    assert (out / "summary.md").exists()


def _tiny_nemotron():
    """A small random-weight Nemotron streaming encoder with the real processor (needs the hub processor files)."""
    import types

    from transformers import AutoConfig, AutoProcessor
    from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
        NemotronAsrStreamingEncoder,
    )

    name = "nvidia/nemotron-speech-streaming-en-0.6b"
    try:
        proc = AutoProcessor.from_pretrained(name)
        cfg = AutoConfig.from_pretrained(name).encoder_config
    except OSError as e:
        pytest.skip(f"Nemotron processor unavailable: {e}")
    cfg = type(cfg)(**{**cfg.to_dict(), "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                       "intermediate_size": 128, "subsampling_conv_channels": 16})
    torch.manual_seed(0)
    enc = NemotronAsrStreamingEncoder(cfg).eval()
    return types.SimpleNamespace(enc=enc, proc=proc, device=torch.device("cpu"), dtype=torch.float32)


def test_batched_listener_matches_single_streams():
    """Calls that start at different times, with different chunk sizes pushed, encoded together: every call gets
    the same vectors as its own StreamingListener (filling and full sliding windows, first chunks together)."""
    import numpy as np

    from s2s.models.batched_listener import BatchedListener
    from s2s.models.streaming_listener import StreamingListener

    nem = _tiny_nemotron()
    bl = BatchedListener(nem, lookahead=1)
    rng = np.random.default_rng(0)
    n_calls, seconds = 4, 9
    audio = [rng.standard_normal(16000 * seconds).astype(np.float32) * 0.1 for _ in range(n_calls)]
    starts = [0, 0, 6400, 41600]                    # two start together, two later (one in another chunk phase)
    batched = [bl.new_stream() for _ in range(n_calls)]
    got = [[] for _ in range(n_calls)]
    step = 2560                                     # 160 ms pushes, like the live server
    for t in range(0, 16000 * seconds + max(starts), step):
        live = []
        for i in range(n_calls):
            a, b = t - starts[i], t - starts[i] + step
            if b <= 0 or a >= len(audio[i]):
                continue
            batched[i].push(audio[i][max(0, a):b])
            live.append(i)
        res = bl.step([batched[i] for i in live])
        for i in live:
            got[i].extend(res[id(batched[i])])
    assert bl.stats["max_batch"] >= 3
    for i in range(n_calls):
        ref = StreamingListener(nem, lookahead=1)
        ref.push(audio[i])
        exp = ref.new_frames()
        assert len(got[i]) == len(exp) > 100
        assert torch.allclose(torch.stack(got[i]), torch.stack(exp), atol=1e-4), i


def test_batch_live_serves_several_guests(tiny_models, tmp_path, monkeypatch):
    """The whole batched pipeline over the real /live protocol (CPU, tiny models, a stub voice): three guests at
    once speak (loud noise, then silence), each turn is detected, heard by the batched listener, answered by the
    batched thinker and spoken; every guest gets audio and turn_complete, twice."""
    import argparse
    import json
    import threading

    import numpy as np
    from fastapi.testclient import TestClient

    import s2s.models.encoders as encoders
    from s2s.cli import batch_live
    from s2s.models.adapter import SpeechAdapter
    from s2s.models.thinker import Thinker
    from s2s.runtime.external_tts import ExternalTTS

    nem = _tiny_nemotron()
    nem.lookahead = 1
    monkeypatch.setattr(encoders, "build_encoder", lambda spec, dev: nem)
    monkeypatch.setattr(ExternalTTS, "synth", lambda self, text: (np.full(2400, 0.1, np.float32), 24000))
    monkeypatch.setattr(batch_live.Pipeline, "transcript", lambda self, lat: "hello there")
    th = Thinker(tiny_models["qwen"], torch.device("cpu"), torch.float32)
    ckpt = tmp_path / "speech_llm"
    ckpt.mkdir()
    SpeechAdapter(latent_dim=64, llm_dim=th.hidden_size, d_model=32, n_layers=1, n_heads=4,
                  encoder="nemotron:tiny@1").save(str(ckpt / "adapter.pt"))
    args = argparse.Namespace(
        config="configs/qwen4b_v3.yaml", speech_llm_dir=str(ckpt), tts_url="http://a/tts,http://b/tts", max_calls=4,
        max_tokens=2048, prefill_budget=64, device="cpu", encoder_device=None,
        set=[f"thinker.model={tiny_models['qwen']}", "thinker.dtype=float32", "runtime.endpoint.vad=energy",
             "runtime.max_new_tokens=12", "runtime.temperature=1.5"])  # a random tiny model: sample real characters
    import s2s.runtime.agent as agent_mod

    monkeypatch.setattr(agent_mod, "speakable", lambda piece: True)
    pipe = batch_live.Pipeline(args)
    client = TestClient(batch_live.build_app(pipe))
    rng = np.random.default_rng(0)
    speech = (np.clip(rng.standard_normal(16000) * 0.3, -1, 1) * 32767).astype("<i2").tobytes()
    silence = bytes(2 * 16000 * 2)
    results, errors = {}, []

    def guest(i):
        try:
            with client.websocket_connect("/live") as ws:
                ws.send_text(json.dumps({"type": "setup", "system_prompt": f"Hotel {i} concierge.", "tools": []}))
                assert json.loads(ws.receive_text())["type"] == "setup_complete"
                got = []
                for _ in range(2):
                    pcm = speech + silence
                    for off in range(0, len(pcm), 3200):
                        ws.send_bytes(pcm[off: off + 3200])
                    audio, msgs = 0, []
                    while True:
                        m = ws.receive()
                        if m.get("bytes"):
                            audio += len(m["bytes"])
                            continue
                        msgs.append(json.loads(m["text"]))
                        if msgs[-1]["type"] in ("turn_complete", "error"):
                            break
                    got.append((audio, [x["type"] for x in msgs], msgs[-1].get("timings_ms")))
                results[i] = got
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=guest, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(240)
    pipe.stop = True
    pipe.sched.shutdown()
    assert not errors, errors
    assert sorted(results) == [0, 1, 2]
    for got in results.values():
        for audio, types, timings in got:
            assert audio > 0 and types[0] == "input_transcription" and types[-1] == "turn_complete", types
            assert "first_token" in timings and "first_audio" in timings
    assert pipe.listener.stats["max_batch"] >= 2 and pipe.stats["turns"] == 6
