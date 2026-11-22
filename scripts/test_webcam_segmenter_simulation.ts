import { readFileSync } from 'node:fs';
import { Segmenter } from '../repos/GestureSign/web/src/ml/segmenter';
import { emptyFrame, N_POINTS, type Frame } from '../repos/GestureSign/web/src/ml/landmarks';
import { modelInput } from '../repos/GestureSign/web/src/ml/features';
import { parseMeta } from '../repos/GestureSign/web/src/ml/meta';

const rawClips = JSON.parse(readFileSync('repos/GestureSign/web/public/replay/clips.json', 'utf-8'));
const meta = parseMeta(JSON.parse(readFileSync('repos/GestureSign/web/public/models/model-meta.json', 'utf-8')));

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

console.log("=== Testing Real Clips with Live Segmenter (NO FLUSH) ===");

for (const clip of clips) {
  const decodedFrames: Frame[] = clip.frames.map((flat: any) => decodeFrame(flat));
  const seg = new Segmenter();
  const events: any[] = [];
  
  // Test 1: Clip followed by hands held still (user holds hand at end of sign for 500ms)
  const stillFrame = decodedFrames[decodedFrames.length - 1]!;
  const fps = clip.fps || 30;
  const dt = 1000 / fps;
  let t = 1000; // start at 1000ms

  // Feed active signing frames
  for (let i = 0; i < decodedFrames.length; i++) {
    const ev = seg.push(decodedFrames[i]!, t);
    if (ev) events.push({ ...ev, frameIdx: i, t });
    t += dt;
  }

  // Follow with 600ms of still frames (still in view, but stationary)
  for (let i = 0; i < Math.ceil(600 / dt); i++) {
    const ev = seg.push(stillFrame, t);
    if (ev) events.push({ ...ev, stillIdx: i, t });
    t += dt;
  }

  console.log(`\nClip '${clip.sign}': ${decodedFrames.length} frames at ${fps} fps`);
  console.log(`Events when followed by STILL hold:`, events.map(e => ({ type: e.type, reason: e.reason, segFrames: e.segment?.frames.length })));
  
  // Check classify condition on any segment
  for (const ev of events) {
    if (ev.type === 'end') {
      const input = modelInput(ev.segment.frames, ev.segment.fps, meta);
      console.log(`  -> Input frames after resample: ${input.frames} (minFrames: ${meta.minFrames}, dim: ${input.dim})`);
      if (input.frames < meta.minFrames) {
        console.log(`  *** WARNING: input.frames (${input.frames}) < minFrames (${meta.minFrames}) -> WOULD BE DISCARDED! ***`);
      }
    }
  }

  // Test 2: Clip followed by hands lowered / absent
  const seg2 = new Segmenter();
  const events2: any[] = [];
  t = 1000;
  for (let i = 0; i < decodedFrames.length; i++) {
    const ev = seg2.push(decodedFrames[i]!, t);
    if (ev) events2.push({ ...ev, frameIdx: i, t });
    t += dt;
  }
  // Follow with 600ms of absent frames (hands lowered)
  for (let i = 0; i < Math.ceil(600 / dt); i++) {
    const ev = seg2.push(emptyFrame(), t);
    if (ev) events2.push({ ...ev, absentIdx: i, t });
    t += dt;
  }
  console.log(`Events when followed by LOWERING hands:`, events2.map(e => ({ type: e.type, reason: e.reason, segFrames: e.segment?.frames.length })));
}

