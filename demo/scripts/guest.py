import json, numpy as np, soundfile as sf
from kokoro import KPipeline
p = KPipeline(lang_code="a")
Q = ["Hi! Who are you?",
     "Could you send two extra towels to my room?",
     "The air conditioning isn't working.",
     "What time is breakfast served?",
     "What is my room number?"]
out = []
for i, q in enumerate(Q):
    wav = np.concatenate([a.numpy() for _, _, a in p(q, voice="am_michael", speed=1.0)])
    wav = np.concatenate([np.zeros(2400, np.float32), wav, np.zeros(4800, np.float32)])
    sf.write(f"/tmp/demo/audio/guest_{i}.wav", wav, 24000)
    out.append({"i": i, "text": q, "seconds": round(len(wav) / 24000, 2)})
json.dump(out, open("/tmp/demo/guest.json", "w"), indent=1)
print(out)
