import { readFileSync } from 'node:fs';
import { describe, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import { Recognizer } from '../src/ml/recognizer.ts';
import { parseMeta, parseLabels } from '../src/ml/meta.ts';
import { emptyFrame } from '../src/ml/landmarks.ts';

const meta = parseMeta(JSON.parse(readFileSync('./public/models/model-meta.json', 'utf-8')));
const labels = parseLabels(JSON.parse(readFileSync('./public/models/labels.json', 'utf-8')), meta.classes);
const clipsData = JSON.parse(readFileSync('./public/replay/clips.json', 'utf-8'));

ort.env.wasm.numThreads = 1;

describe('User Webcam Simulation Scenarios', () => {
  it('tests various user signing durations and frame rates', async () => {
    const session = await ort.InferenceSession.create('./public/models/model.onnx', { executionProviders: ['wasm'] });

    const infer = async (input) => {
      const tensor = new ort.Tensor('float32', input.data, [1, input.frames, input.dim]);
      const outputs = await session.run({ frames: tensor });
      return {
        logits: outputs.logits.data,
        embedding: outputs.embedding.data,
        ms: 5,
      };
    };

    const points = clipsData.points;
    function decodeFrame(flat) {
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

    const helloClip = clipsData.clips.find(c => c.sign === 'hello');
    const helloFrames = helloClip.frames.map(decodeFrame);

    console.log('\n--- Scenario A: User raises hand, makes brief sign (sampled at 15 fps) ---');
    // Subsample helloFrames to 15 fps (every 2nd frame = 9 frames)
    const frames15fps = helloFrames.filter((_, i) => i % 2 === 0);
    console.log(`Subsampled hello to 15 fps: ${frames15fps.length} frames`);

    const eventsA = [];
    const recA = new Recognizer(meta, labels, infer, (e) => eventsA.push(e));
    let t = 1000;
    const dt15 = 1000 / 15; // 66.7ms per frame

    // 1 second with no hands (just sitting in view)
    for (let i = 0; i < 15; i++) {
      recA.push({ frame: emptyFrame(), t, poses: 1 });
      t += dt15;
    }

    // 9 frames of signing
    for (let i = 0; i < frames15fps.length; i++) {
      recA.push({ frame: frames15fps[i], t, poses: 1 });
      t += dt15;
    }

    // Hand lowers (empty frames)
    for (let i = 0; i < 15; i++) {
      recA.push({ frame: emptyFrame(), t, poses: 1 });
      t += dt15;
    }

    await new Promise(r => setTimeout(r, 60));
    console.log('Events in Scenario A (15 fps, 9 frames):', eventsA.map(e => ({ type: e.type, reason: e.reason, kind: e.result?.decision?.kind })));

    console.log('\n--- Scenario B: User makes quick sign (only 7 frames at 15 fps) ---');
    const frames7 = frames15fps.slice(0, 7);
    const eventsB = [];
    const recB = new Recognizer(meta, labels, infer, (e) => eventsB.push(e));
    t = 1000;
    for (let i = 0; i < 15; i++) {
      recB.push({ frame: emptyFrame(), t, poses: 1 });
      t += dt15;
    }
    for (let i = 0; i < frames7.length; i++) {
      recB.push({ frame: frames7[i], t, poses: 1 });
      t += dt15;
    }
    for (let i = 0; i < 15; i++) {
      recB.push({ frame: emptyFrame(), t, poses: 1 });
      t += dt15;
    }
    await new Promise(r => setTimeout(r, 60));
    console.log('Events in Scenario B (15 fps, 7 frames):', eventsB.map(e => ({ type: e.type, reason: e.reason, kind: e.result?.decision?.kind })));

    console.log('\n--- Scenario C: User raises hand, pauses briefly (holds still), then starts sign ---');
    const eventsC = [];
    const recC = new Recognizer(meta, labels, infer, (e) => eventsC.push(e));
    t = 1000;
    // 1s idle
    for (let i = 0; i < 15; i++) {
      recC.push({ frame: emptyFrame(), t, poses: 1 });
      t += dt15;
    }
    // Hand enters and moves up (2 frames)
    recC.push({ frame: helloFrames[0], t, poses: 1 }); t += dt15;
    recC.push({ frame: helloFrames[1], t, poses: 1 }); t += dt15;
    // User pauses for 350ms (5 frames still) before starting sign motion!
    for (let i = 0; i < 5; i++) {
      recC.push({ frame: helloFrames[1], t, poses: 1 });
      t += dt15;
    }
    // Now user performs the sign (frames 2 to 16)
    for (let i = 2; i < helloFrames.length; i++) {
      recC.push({ frame: helloFrames[i], t, poses: 1 });
      t += dt15;
    }
    // Lower hand
    for (let i = 0; i < 15; i++) {
      recC.push({ frame: emptyFrame(), t, poses: 1 });
      t += dt15;
    }
    await new Promise(r => setTimeout(r, 60));
    console.log('Events in Scenario C (hand raised -> paused -> signed):', eventsC.map(e => ({ type: e.type, reason: e.reason, kind: e.result?.decision?.kind })));
  });
});

