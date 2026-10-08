# Using the model from a NestJS voice backend (in place of Gemini Live)

`s2s-live.client.ts` is a drop-in for `new GoogleGenAI({ apiKey }).live.connect(...)`. It talks to the
model server's `/live` WebSocket and gives back the **same message shapes Gemini Live does**
(`serverContent.modelTurn.parts[].inlineData`, `inputTranscription`, `outputTranscription`,
`turnComplete`, `toolCall.functionCalls`) and accepts the same calls (`sendRealtimeInput`,
`sendClientContent`, `sendToolResponse`, `close`). An existing Gemini Live message handler, tool runner
and transcript logging keep working; only the connect call changes.

## 1. Start the model server

On the GPU machine (inside the `rocm` container on the MI300X droplet):

```bash
cd ~/Voice-to-voice-model-training && git pull -q
export S2S_LIVE_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))"); echo "token: $S2S_LIVE_TOKEN"
python -m s2s.cli.ws_live --config configs/small.yaml \
    --speech-llm-dir checkpoints/speech_llm_pk4 --talker-dir checkpoints/talker_hifi_pre \
    --tunnel > ~/live.log 2>&1 &
grep -m1 trycloudflare ~/live.log      # -> https://<name>.trycloudflare.com
```

The application's endpoint is `wss://<name>.trycloudflare.com/live`. The system prompt and the tools come
from the application at connect time, so `--system-prompt-file` / `--tools` are not needed for `/live`.
`--talker-dir` selects the reply voice (our talker) and the thinker it was trained with.
Keep the token secret: anyone with the URL and token can use the GPU.

## 2. Swap the connect call

Copy `s2s-live.client.ts` next to the session service, add two settings (e.g. env vars
`S2S_LIVE_URL`, `S2S_LIVE_TOKEN`) and a switch (`VOICE_LIVE_PROVIDER=s2s`). Then, where the live session
is opened:

```ts
import { connectS2sLive } from './s2s-live.client';

const useS2s = process.env.VOICE_LIVE_PROVIDER === 's2s';

// isConfigured(): no Gemini key is needed for the s2s provider
return useS2s || !!resolveApiKey('primary') || !!resolveApiKey('secondary');

// in the key loop: the s2s provider has no keys to rotate
const apiKey = useS2s ? 's2s' : resolveApiKey(slot);

// the connect itself (same liveConfig, same callbacks object):
const opened = await connectWithTimeout(
  useS2s
    ? connectS2sLive({
        url: process.env.S2S_LIVE_URL!,            // wss://<name>.trycloudflare.com/live
        token: process.env.S2S_LIVE_TOKEN,
        config: liveConfig,                         // systemInstruction + tools are sent to the model
        inputSampleRate: VOICE_LIVE_CONFIG.sendSampleRate, // the PCM rate you forward (16000)
        outputSampleRate: 24000,                    // same as Gemini, so the existing resampler still applies
        callbacks,
      })
    : new GoogleGenAI({ apiKey }).live.connect({ model: runtimeConfig.model, config: liveConfig, callbacks }),
  VOICE_LIVE_CONFIG.geminiConnectTimeoutMs,
  (orphan) => { /* unchanged */ },
);
```

Nothing else changes: the greeting kickoff (`sendClientContent`), audio forwarding (`sendRealtimeInput`),
tool execution against your backend (`sendToolResponse`), transcript logging and `turnComplete`
forwarding all work as with Gemini. Gemini-style `functionDeclarations` (with `Type.OBJECT` etc.) are
accepted as they are; `behavior` and `scheduling` are ignored.

## Differences from Gemini Live

| | Gemini Live | this model |
|---|---|---|
| Turn-taking | full duplex, barge-in (`interrupted`) | half duplex: the guest's audio is ignored while the model thinks and speaks |
| `turnComplete` | when generation ends | when the reply has finished **playing** (the device can flush its buffer safely) |
| Tool calls | non-blocking, can talk meanwhile | the model waits for the tool result (20 s timeout), then speaks it |
| Reconnect | session resumption handle | a reconnect starts a fresh conversation (your greeting runs again) |
| Languages | many | English only |
| Voice | `speechConfig` | fixed on the server: our own talker (`--talker-dir`) |
| Usage | `usageMetadata` tokens | none (self-hosted: cost is the GPU) |

Tool-calling quality with your exact six tools improves once the thinker is trained on them (thinker v4).
This integration is the plumbing for testing the full path today.

## Protocol (for other clients)

See the docstring of `s2s/cli/live_api.py`: a `setup` JSON message (system prompt, tools, sample rates),
binary PCM16 both ways, `text` turns, `tool_call` / `tool_response`, transcriptions and `turn_complete`.
