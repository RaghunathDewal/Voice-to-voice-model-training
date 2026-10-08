// Drop-in replacement for `new GoogleGenAI({ apiKey }).live.connect(...)` that talks to our own
// speech-to-speech model server (s2s.cli.ws_live, endpoint /live) instead of Gemini Live.
//
// It emits the same message shapes Gemini Live does (serverContent.modelTurn.parts[].inlineData,
// inputTranscription, outputTranscription, turnComplete, toolCall.functionCalls), and accepts the same
// calls (sendRealtimeInput, sendClientContent, sendToolResponse, close), so an existing Gemini Live
// handler keeps working unchanged. Only `live.connect(...)` is swapped for `connectS2sLive(...)`.
//
// Differences from Gemini Live:
//   * half duplex: the guest's audio is ignored while the model thinks and speaks (no barge-in yet),
//     and turnComplete arrives once the reply has finished playing
//   * no session resumption / context compression: a reconnect starts a fresh conversation
//   * the voice is fixed on the server (speechConfig is ignored)

import WebSocket from 'ws';

export interface S2sLiveCallbacks {
  onopen?: () => void;
  onmessage?: (message: any) => void;
  onerror?: (error: any) => void;
  onclose?: (event: { code: number; reason: string }) => void;
}

export interface S2sLiveConnectParams {
  /** e.g. wss://<host>/live (the model server's /live endpoint) */
  url: string;
  /** the server's --live-token, if it was started with one */
  token?: string;
  /** the same liveConfig object passed to Gemini: systemInstruction and tools are used, the rest is ignored */
  config: { systemInstruction?: string | { parts?: { text?: string }[] }; tools?: any[] };
  /** PCM16 rate of the audio you send with sendRealtimeInput (default 16000) */
  inputSampleRate?: number;
  /** PCM16 rate of the audio you receive (default 24000, like Gemini) */
  outputSampleRate?: number;
  callbacks: S2sLiveCallbacks;
  /** reject if the server has not finished setup within this time (default 15 s) */
  connectTimeoutMs?: number;
}

export class S2sLiveSession {
  constructor(private readonly ws: WebSocket) {}

  /** Same call shape as Gemini: { audio: { data: <base64 PCM16>, mimeType: 'audio/pcm;rate=16000' } } */
  sendRealtimeInput(input: { audio?: { data?: string; mimeType?: string } }): void {
    const data = input?.audio?.data;
    if (!data || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(Buffer.from(data, 'base64'), { binary: true });
  }

  /** A typed user turn (e.g. the "greet the guest" kickoff). Same shape as Gemini's sendClientContent. */
  sendClientContent(content: { turns?: any; turnComplete?: boolean }): void {
    const turns = Array.isArray(content?.turns) ? content.turns : content?.turns ? [content.turns] : [];
    const text = turns
      .flatMap((t: any) => (typeof t === 'string' ? [t] : (t?.parts ?? []).map((p: any) => p?.text ?? '')))
      .join(' ')
      .trim();
    if (text && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'text', text }));
  }

  /** Same shape as Gemini: { functionResponses: [{ id, name, response }] }. Extra fields are ignored. */
  sendToolResponse(r: { functionResponses: Array<{ id?: string; name?: string; response?: any }> }): void {
    for (const fr of r?.functionResponses ?? []) {
      if (this.ws.readyState !== WebSocket.OPEN) return;
      this.ws.send(JSON.stringify({ type: 'tool_response', id: fr.id, name: fr.name, response: fr.response ?? {} }));
    }
  }

  close(): void {
    try {
      this.ws.close(1000, 'client closed');
    } catch {
      // already closed
    }
  }
}

function promptText(si: S2sLiveConnectParams['config']['systemInstruction']): string {
  if (!si) return '';
  if (typeof si === 'string') return si;
  return (si.parts ?? []).map((p) => p.text ?? '').join('\n');
}

export function connectS2sLive(params: S2sLiveConnectParams): Promise<S2sLiveSession> {
  const inputRate = params.inputSampleRate ?? 16000;
  const outputRate = params.outputSampleRate ?? 24000;
  const cb = params.callbacks ?? {};
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(params.url, {
      headers: params.token ? { Authorization: `Bearer ${params.token}` } : undefined,
    });
    const session = new S2sLiveSession(ws);
    let ready = false;
    const timer = setTimeout(() => {
      if (ready) return;
      reject(new Error(`s2s live setup timed out after ${params.connectTimeoutMs ?? 15000} ms`));
      ws.terminate();
    }, params.connectTimeoutMs ?? 15000);

    ws.on('open', () => {
      ws.send(
        JSON.stringify({
          type: 'setup',
          system_prompt: promptText(params.config?.systemInstruction),
          tools: params.config?.tools ?? [], // Gemini functionDeclarations are accepted as they are
          input_sample_rate: inputRate,
          output_sample_rate: outputRate,
        }),
      );
    });

    ws.on('message', (data: WebSocket.RawData, isBinary: boolean) => {
      if (isBinary) {
        const pcm = Buffer.isBuffer(data) ? data : Buffer.from(data as ArrayBuffer);
        cb.onmessage?.({
          serverContent: {
            modelTurn: { parts: [{ inlineData: { data: pcm.toString('base64'), mimeType: `audio/pcm;rate=${outputRate}` } }] },
          },
        });
        return;
      }
      let m: any;
      try {
        m = JSON.parse(data.toString());
      } catch {
        return;
      }
      switch (m.type) {
        case 'setup_complete':
          ready = true;
          clearTimeout(timer);
          cb.onopen?.();
          resolve(session);
          break;
        case 'input_transcription':
          cb.onmessage?.({ serverContent: { inputTranscription: { text: m.text } } });
          break;
        case 'output_transcription':
          cb.onmessage?.({ serverContent: { outputTranscription: { text: m.text } } });
          break;
        case 'tool_call':
          cb.onmessage?.({ toolCall: { functionCalls: [{ id: m.id, name: m.name, args: m.args ?? {} }] } });
          break;
        case 'turn_complete':
          cb.onmessage?.({ serverContent: { turnComplete: true } });
          break;
        case 'error':
          cb.onerror?.(new Error(m.message));
          break;
      }
    });

    ws.on('error', (err) => {
      if (!ready) {
        clearTimeout(timer);
        reject(err);
      } else cb.onerror?.(err);
    });

    ws.on('close', (code: number, reason: Buffer) => {
      clearTimeout(timer);
      if (!ready) reject(new Error(`s2s live closed before setup: ${code} ${reason.toString()}`));
      else cb.onclose?.({ code, reason: reason.toString() });
    });
  });
}
