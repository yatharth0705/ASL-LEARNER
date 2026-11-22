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

describe('Live Pipeline Full Diagnostic', () => {
  it('runs diagnostic on all modes with real ONNX model', async () => {
    const session = await ort.InferenceSession.create('./public/models/model.onnx', { executionProviders: ['wasm'] });
    console.log('ONNX Model Loaded. Inputs:', session.inputNames, 'Outputs:', session.outputNames);

    const infer = async (input) => {
      const tensor = new ort.Tensor('float32', input.data, [1, input.frames, input.dim]);
      const t0 = performance.now();
      const outputs = await session.run({ frames: tensor });
      const ms = performance.now() - t0;
      return {
        logits: outputs.logits.data,
        embedding: outputs.embedding.data,
        ms,
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

    console.log('\n================================================================');
    console.log('DIAGNOSTIC TEST 1: Replay Mode with flush()');
    console.log('================================================================');

    for (const clip of clipsData.clips) {
      const events = [];
      const rec = new Recognizer(meta, labels, infer, (e) => events.push(e));
      const frames = clip.frames.map(decodeFrame);
      const dt = 1000 / clip.fps;
      let t = 1000;
      for (let i = 0; i < frames.length; i++) {
        rec.push({ frame: frames[i], t, poses: 1 });
        t += dt;
      }
      for (let i = 0; i < Math.ceil(450 / dt); i++) {
        rec.push({ frame: emptyFrame(), t, poses: 0 });
        t += dt;
      }
      rec.flush();

      await new Promise(r => setTimeout(r, 60));

      const result = events.find(e => e.type === 'result');
      const discarded = events.find(e => e.type === 'discarded');
      const error = events.find(e => e.type === 'error');

      if (result) {
        const top = result.result.decision.top[0];
        const topLabel = labels[top.index]?.label;
        const topProb = (top.probability * 100).toFixed(1);
        const kind = result.result.decision.kind;
        console.log(`[PASS] Clip '${clip.sign}': Pred = ${topLabel} (${topProb}%), Decision = ${kind}`);
      } else {
        console.log(`[FAIL] Clip '${clip.sign}': NO RESULT! Discarded: ${Boolean(discarded)}, Error: ${error?.message}`);
      }
    }

    console.log('\n================================================================');
    console.log('DIAGNOSTIC TEST 2: Live Webcam Simulation (NO flush, hand held still)');
    console.log('================================================================');

    for (const clip of clipsData.clips) {
      const events = [];
      const rec = new Recognizer(meta, labels, infer, (e) => events.push(e));
      const frames = clip.frames.map(decodeFrame);
      const stillFrame = frames[frames.length - 1];
      const dt = 1000 / clip.fps;
      let t = 1000;

      for (let i = 0; i < frames.length; i++) {
        rec.push({ frame: frames[i], t, poses: 1 });
        t += dt;
      }
      for (let i = 0; i < Math.ceil(700 / dt); i++) {
        rec.push({ frame: stillFrame, t, poses: 1 });
        t += dt;
      }

      await new Promise(r => setTimeout(r, 60));

      const result = events.find(e => e.type === 'result');
      const discarded = events.filter(e => e.type === 'discarded');
      const error = events.find(e => e.type === 'error');

      if (result) {
        const top = result.result.decision.top[0];
        const topLabel = labels[top.index]?.label;
        const topProb = (top.probability * 100).toFixed(1);
        const kind = result.result.decision.kind;
        console.log(`[PASS] Live Clip '${clip.sign}' (still hold): Pred = ${topLabel} (${topProb}%), Decision = ${kind}`);
      } else {
        console.log(`[FAIL] Live Clip '${clip.sign}' (still hold): NO RESULT! Discarded count: ${discarded.length}, Error: ${error?.message}`);
      }
    }

    console.log('\n================================================================');
    console.log('DIAGNOSTIC TEST 3: Live Webcam Simulation (NO flush, hands lowered)');
    console.log('================================================================');

    for (const clip of clipsData.clips) {
      const events = [];
      const rec = new Recognizer(meta, labels, infer, (e) => events.push(e));
      const frames = clip.frames.map(decodeFrame);
      const dt = 1000 / clip.fps;
      let t = 1000;

      for (let i = 0; i < frames.length; i++) {
        rec.push({ frame: frames[i], t, poses: 1 });
        t += dt;
      }
      for (let i = 0; i < Math.ceil(700 / dt); i++) {
        rec.push({ frame: emptyFrame(), t, poses: 1 });
        t += dt;
      }

      await new Promise(r => setTimeout(r, 60));

      const result = events.find(e => e.type === 'result');
      const discarded = events.filter(e => e.type === 'discarded');
      const error = events.find(e => e.type === 'error');

      if (result) {
        const top = result.result.decision.top[0];
        const topLabel = labels[top.index]?.label;
        const topProb = (top.probability * 100).toFixed(1);
        const kind = result.result.decision.kind;
        console.log(`[PASS] Live Clip '${clip.sign}' (hands lowered): Pred = ${topLabel} (${topProb}%), Decision = ${kind}`);
      } else {
        console.log(`[FAIL] Live Clip '${clip.sign}' (hands lowered): NO RESULT! Discarded count: ${discarded.length}, Error: ${error?.message}`);
      }
    }
  }, 25000);
});

