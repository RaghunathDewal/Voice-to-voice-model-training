import React from 'react';
import {
  AbsoluteFill, Audio, Img, OffthreadVideo, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig,
} from 'remotion';
import storyboard from '../storyboard.json';

type Box = {x: number; y: number; width: number; height: number};
const sb = storyboard as any;
const GOLD = '#e8c27a';
const TEAL = '#5fd4c4';

const Fonts: React.FC = () => (
  <style>{`
    @font-face{font-family:'Fraunces';font-weight:500;src:url(${staticFile('fonts/f2.woff2')}) format('woff2')}
    @font-face{font-family:'Fraunces';font-weight:600;src:url(${staticFile('fonts/f5.woff2')}) format('woff2')}
    @font-face{font-family:'Inter';font-weight:400;src:url(${staticFile('fonts/f12.woff2')}) format('woff2')}
    @font-face{font-family:'Inter';font-weight:500;src:url(${staticFile('fonts/f19.woff2')}) format('woff2')}
    @font-face{font-family:'Inter';font-weight:600;src:url(${staticFile('fonts/f26.woff2')}) format('woff2')}
  `}</style>
);

const fadeInOut = (frame: number, dur: number, edge = 8) =>
  interpolate(frame, [0, edge, Math.max(edge + 1, dur - edge), dur], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

/* Gold ring + click ripple + label on the element that was clicked */
const Highlight: React.FC<{box: Box; label: string}> = ({box, label}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pop = spring({frame, fps, config: {damping: 12, stiffness: 160}});
  const opacity = interpolate(frame, [0, 4, 40, 54], [0, 1, 1, 0], {extrapolateRight: 'clamp'});
  const pad = 10;
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const ripple = interpolate(frame, [6, 30], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{opacity, pointerEvents: 'none'}}>
      <div style={{
        position: 'absolute', left: box.x - pad, top: box.y - pad, width: box.width + 2 * pad, height: box.height + 2 * pad,
        border: `4px solid ${GOLD}`, borderRadius: 18, transform: `scale(${0.85 + 0.15 * pop})`,
        boxShadow: `0 0 0 6px rgba(232,194,122,.18), 0 0 40px rgba(232,194,122,.55)`,
      }} />
      <div style={{
        position: 'absolute', left: cx - 60 * ripple, top: cy - 60 * ripple, width: 120 * ripple, height: 120 * ripple,
        borderRadius: '50%', border: `3px solid rgba(232,194,122,${1 - ripple})`,
      }} />
      <div style={{
        position: 'absolute', left: Math.min(box.x - pad, 1920 - 260), top: box.y + box.height + pad + 12,
        background: GOLD, color: '#1b1405', font: '600 20px Inter', padding: '8px 14px', borderRadius: 10,
        transform: `translateY(${(1 - pop) * 10}px)`, whiteSpace: 'nowrap',
      }}>👆 {label}</div>
    </AbsoluteFill>
  );
};

/* Lower-left caption card: step, caption, narration, who is speaking, latency */
const CaptionCard: React.FC<{scene: any; index: number; total: number}> = ({scene, index, total}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dur = Math.round((scene.end_s - scene.start_s) * fps);
  const opacity = fadeInOut(frame, dur, 10);
  const slide = interpolate(frame, [0, 12], [24, 0], {extrapolateRight: 'clamp'});
  const t = scene.start_s + frame / fps;
  const speaking = sb.audio.find((a: any) => t >= a.at_s && t < a.at_s + a.duration_s);
  const who = speaking ? (speaking.who === 'guest' ? {c: TEAL, txt: '🎙️ Guest speaking'} : {c: GOLD, txt: "🔊 Concierge — our model's voice"})
    : {c: '#8b7cf6', txt: '🤔 Model thinking…'};
  return (
    <div style={{
      position: 'absolute', left: 52, bottom: 26, width: 560, opacity, transform: `translateY(${slide}px)`,
      background: 'rgba(11,16,32,.92)', border: '1px solid rgba(255,255,255,.14)', borderRadius: 18, padding: '16px 20px',
      boxShadow: '0 20px 50px rgba(0,0,0,.45)', color: '#e9ecf5', fontFamily: 'Inter',
    }}>
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 15, color: '#8f98b3', marginBottom: 6}}>
        <span>Step {index} of {total}</span>
        {scene.latency_s ? <span style={{color: TEAL}}>⚡ first audio {scene.latency_s.toFixed(2)} s (T4)</span> : null}
      </div>
      <div style={{font: '600 23px Fraunces', marginBottom: 6, lineHeight: 1.25}}>{scene.caption}</div>
      <div style={{fontSize: 17, lineHeight: 1.4, color: '#c9cfe2'}}>{scene.narration}</div>
      <div style={{marginTop: 10, display: 'inline-block', fontSize: 15, fontWeight: 600, color: who.c,
        border: `1px solid ${who.c}55`, background: `${who.c}18`, padding: '5px 10px', borderRadius: 999}}>{who.txt}</div>
    </div>
  );
};

/* Title over the (still empty) conversation panel */
const IntroCard: React.FC<{scene: any}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dur = Math.round((scene.end_s - scene.start_s) * fps);
  const opacity = fadeInOut(frame, dur, 10);
  const s = spring({frame, fps, config: {damping: 14}});
  return (
    <div style={{position: 'absolute', left: 690, top: 360, width: 625, opacity, transform: `scale(${0.94 + 0.06 * s})`,
      textAlign: 'center', color: '#e9ecf5'}}>
      <div style={{font: '600 15px Inter', letterSpacing: 4, color: GOLD, marginBottom: 14}}>LIVE DEMO</div>
      <div style={{font: '600 54px Fraunces', lineHeight: 1.1}}>{scene.caption}</div>
      <div style={{font: '400 22px Inter', color: '#aeb6cf', marginTop: 16}}>{scene.subtitle}</div>
      <div style={{font: '400 18px Inter', color: '#8f98b3', marginTop: 22}}>{scene.narration}</div>
    </div>
  );
};

const OutroCard: React.FC<{scene: any}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dur = Math.round((scene.end_s - scene.start_s) * fps);
  const bg = interpolate(frame, [0, 14], [0, 0.78], {extrapolateRight: 'clamp'});
  const s = spring({frame: frame - 6, fps, config: {damping: 14}});
  return (
    <AbsoluteFill style={{background: `rgba(8,11,24,${bg})`, alignItems: 'center', justifyContent: 'center', opacity: fadeInOut(frame, dur + 20, 6)}}>
      <div style={{textAlign: 'center', color: '#e9ecf5', transform: `translateY(${(1 - s) * 30}px)`, opacity: s}}>
        <div style={{font: '600 60px Fraunces'}}>{scene.caption}</div>
        <div style={{font: '400 26px Inter', color: '#aeb6cf', marginTop: 18}}>{scene.subtitle}</div>
        <div style={{font: '400 20px Inter', color: GOLD, marginTop: 26}}>{scene.narration}</div>
      </div>
    </AbsoluteFill>
  );
};

const Progress: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return <div style={{position: 'absolute', left: 0, bottom: 0, height: 5, width: `${(100 * frame) / durationInFrames}%`,
    background: `linear-gradient(90deg, ${TEAL}, ${GOLD})`}} />;
};

export const VoiceDemo: React.FC = () => {
  const {fps, durationInFrames} = useVideoConfig();
  const f = (s: number) => Math.round(s * fps);
  const segs = sb.segments;
  const lastEnd = segs[segs.length - 1].at_s + (segs[segs.length - 1].to_s - segs[segs.length - 1].from_s);
  const turns = sb.scenes.filter((s: any) => s.id.startsWith('turn'));
  return (
    <AbsoluteFill style={{background: '#0b1020'}}>
      <Fonts />
      {segs.map((s: any, i: number) => (
        <Sequence key={`v${i}`} from={f(s.at_s)} durationInFrames={f(s.to_s - s.from_s)}>
          <OffthreadVideo src={staticFile(s.src)} startFrom={f(s.from_s)} muted />
        </Sequence>
      ))}
      <Sequence from={f(lastEnd)} durationInFrames={Math.max(1, durationInFrames - f(lastEnd))}>
        <Img src={staticFile('shots/10_end.png')} style={{width: 1920, height: 1080}} />
      </Sequence>
      {sb.audio.map((a: any, i: number) => (
        <Sequence key={`a${i}`} from={f(a.at_s)}>
          <Audio src={staticFile(a.src)} volume={a.who === 'guest' ? 0.9 : 1} />
        </Sequence>
      ))}
      {sb.highlights.map((h: any, i: number) => (
        <Sequence key={`h${i}`} from={Math.max(0, f(h.at_s) - 3)} durationInFrames={56}>
          <Highlight box={h.box} label={h.label} />
        </Sequence>
      ))}
      {sb.scenes.map((s: any) => (
        <Sequence key={s.id} from={f(s.start_s)} durationInFrames={Math.max(1, f(s.end_s - s.start_s))}>
          {s.id === 'intro' ? <IntroCard scene={s} /> : s.id === 'outro' ? <OutroCard scene={s} />
            : <CaptionCard scene={s} index={turns.indexOf(s) + 1} total={turns.length} />}
        </Sequence>
      ))}
      <Progress />
    </AbsoluteFill>
  );
};
