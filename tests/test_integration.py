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
        "runtime.voice=talker"])  # the tiny talker; the Kokoro voice has its own (stubbed) test
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


def test_runtime_kokoro_voice_speaks_each_phrase(tiny_models, tmp_path, cpu, monkeypatch):
    """runtime.voice=kokoro: the thinker's streamed text is cut into phrases and each is spoken (TTS stubbed)."""
    import s2s.runtime.agent as agent_mod
    import s2s.runtime.tts as tts_mod
    from s2s.runtime.agent import VoiceAgent

    spoken = []

    class FakeKokoro:
        sample_rate = 24000

        def __init__(self, voice, speed, device):
            self.voice = voice

        def speak(self, text):
            spoken.append(text)
            return np.full(2400 * len(text.split()), 0.1, np.float32)

    monkeypatch.setattr(tts_mod, "KokoroVoice", FakeKokoro)
    th = Thinker(tiny_models["qwen"], cpu, torch.float32)
    speech_dir = tmp_path / "speech"
    os.makedirs(speech_dir)
    ad = SpeechAdapter(64, th.hidden_size, d_model=32, n_layers=1, n_heads=4)
    ad.init_scale(th.text_embedding_rms())
    ad.save(str(speech_dir / "adapter.pt"))
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"), [
        f"codec.model={tiny_models['mimi']}", f"thinker.model={tiny_models['qwen']}", "device=cpu",
        "runtime.endpoint.vad=energy", "runtime.voice=kokoro"])
    agent = VoiceAgent(cfg, str(speech_dir), str(tmp_path / "no_talker"))  # no talker needed
    assert agent.talker is None
    p = agent.thinker.prompts
    script = p.ids("Sure, one moment please. Two towels are on the way.") + [p.im_end_id]
    monkeypatch.setattr(agent_mod, "sample_logits", lambda *a, **k: script.pop(0) if script else p.im_end_id)
    session = agent.new_session(context="test", tools=None)
    events = list(session.respond(np.zeros(24000, np.float32)))
    audio = [e for e in events if e["type"] == "audio"]
    assert spoken == ["Sure, one moment please.", "Two towels are on the way."]
    assert [e["text"] for e in audio] == spoken and all(e["sample_rate"] == 24000 for e in audio)
    assert [e["text"] for e in events if e["type"] == "assistant_text"] == ["Sure, one moment please. Two towels are on the way."]


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
