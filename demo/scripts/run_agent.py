import json, random, time, numpy as np, soundfile as sf, torch
from s2s.config import load_config
from s2s.data.hotel import HOTEL_TOOLS, HotelBackend, make_reservation, reservation_context
from s2s.runtime.agent import VoiceAgent
torch.manual_seed(7); random.seed(7)
cfg = load_config("configs/small.yaml", ["device=cpu"])
agent = VoiceAgent(cfg, "checkpoints/speech_llm_pk3", "checkpoints/talker_v2")
res = make_reservation(random.Random(11))
session = agent.new_session(context=reservation_context(res), tools=HOTEL_TOOLS, backend=HotelBackend(res))
guest = json.load(open("/tmp/demo/guest.json"))
turns = []
for g in guest:
    wav, sr = sf.read(f"/tmp/demo/audio/guest_{g['i']}.wav", dtype="float32")
    t0 = time.time(); audio = []; tr = {"guest_text": g["text"], "guest_seconds": g["seconds"], "tool_calls": [], "tool_results": [], "assistant": []}
    for ev in session.respond(wav):
        t = ev["type"]
        if t == "audio": audio.append(ev["audio"])
        elif t == "user_transcript": tr["heard_ctc"] = ev["text"]
        elif t == "assistant_text" and ev["text"]: tr["assistant"].append(ev["text"])
        elif t == "tool_call": tr["tool_calls"].append(ev["call"])
        elif t == "tool_result": tr["tool_results"].append(ev["result"])
        elif t == "timings": tr["cpu_timings_ms"] = {k: round(v) for k, v in ev["ms"].items()}
    bot = np.concatenate(audio) if audio else np.zeros(2400, np.float32)
    sf.write(f"/tmp/demo/audio/bot_{g['i']}.wav", bot, 24000)
    tr["bot_seconds"] = round(len(bot) / 24000, 2)
    turns.append(tr); print(json.dumps(tr)[:400], f"(cpu {time.time()-t0:.0f}s)", flush=True)
json.dump({"reservation": res, "turns": turns}, open("/tmp/demo/conversation.json", "w"), indent=1)
