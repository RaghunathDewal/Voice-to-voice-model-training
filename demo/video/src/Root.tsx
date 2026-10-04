import React from 'react';
import {Composition} from 'remotion';
import {VoiceDemo} from './VoiceDemo';
import storyboard from '../storyboard.json';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="VoiceDemo"
    component={VoiceDemo}
    durationInFrames={Math.round(storyboard.duration_s * storyboard.fps)}
    fps={storyboard.fps}
    width={storyboard.width}
    height={storyboard.height}
  />
);
