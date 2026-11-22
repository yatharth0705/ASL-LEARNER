import { readFileSync } from 'node:fs';
import { describe, it } from 'vitest';
import { Segmenter } from './segmenter';
import { emptyFrame, type Frame } from './landmarks';

const rawClips = JSON.parse(readFileSync(new URL('../../public/replay/clips.json', import.meta.url), 'utf-8'));

const points: number[] = rawClips.points;
const clips = rawClips.clips;

function decodeFrame(flat: (number | null)[]): Frame {
  const f = emptyFrame();
  points.forEach((p, i) => {
    const x = flat[2 * i];
    const y = flat[2 * i + 1];
    if (x !== null && y !== null && typeof x === 'number' && typeof y === 'number') {
      f[p * 3] = x;
      f[p * 3 + 1] = y;
      f[p * 3 + 2] = 0;
    }
  });
  return f;
}

describe('Segmenter Real Clip Simulation without flush', () => {
  for (const clip of clips) {
    it(`simulates live webcam for '${clip.sign}' without flush`, () => {
      const decodedFrames: Frame[] = clip.frames.map((flat: any) => decodeFrame(flat));
      const seg = new Segmenter();
      const events: any[] = [];
      const stillFrame = decodedFrames[decodedFrames.length - 1]!;
      const fps = clip.fps || 30;
      const dt = 1000 / fps;
      let t = 1000;

      for (let i = 0; i < decodedFrames.length; i++) {
        const ev = seg.push(decodedFrames[i]!, t);
        if (ev) events.push({ ...ev, frameIdx: i, t });
        t += dt;
      }

      for (let i = 0; i < Math.ceil(600 / dt); i++) {
        const ev = seg.push(stillFrame, t);
        if (ev) events.push({ ...ev, stillIdx: i, t });
        t += dt;
      }

      console.log(`[SIM STILL] Clip '${clip.sign}' (${decodedFrames.length} frames): events:`, events.map(e => ({ type: e.type, reason: e.reason, segFrames: e.segment?.frames.length })));
      
      const seg2 = new Segmenter();
      const events2: any[] = [];
      t = 1000;
      for (let i = 0; i < decodedFrames.length; i++) {
        const ev = seg2.push(decodedFrames[i]!, t);
        if (ev) events2.push({ ...ev, frameIdx: i, t });
        t += dt;
      }
      for (let i = 0; i < Math.ceil(600 / dt); i++) {
        const ev = seg2.push(emptyFrame(), t);
        if (ev) events2.push({ ...ev, absentIdx: i, t });
        t += dt;
      }
      console.log(`[SIM ABSENT] Clip '${clip.sign}' (${decodedFrames.length} frames): events:`, events2.map(e => ({ type: e.type, reason: e.reason, segFrames: e.segment?.frames.length })));
    });
  }
});
