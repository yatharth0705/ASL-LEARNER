import { describe, expect, it } from 'vitest';
import { TemporalAggregator } from './temporal-aggregator';
import {
  emptyFrame,
  RIGHT_HAND_START,
  POSE_START,
  POSE_LEFT_SHOULDER,
  POSE_RIGHT_SHOULDER,
  POSE_RIGHT_WRIST,
} from '../landmarks';

describe('TemporalAggregator', () => {
  it('accumulates distinct recognized signs', () => {
    const agg = new TemporalAggregator(0.35, 800);
    const r1 = agg.processRecognition({ sign: 'hello', confidence: 0.92 }, 'still', 1000);
    expect(r1.status).toBe('accepted');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);

    const r2 = agg.processRecognition({ sign: 'police', confidence: 0.88 }, 'absent', 1500);
    expect(r2.status).toBe('accepted');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO', 'POLICE']);
  });

  it('debounces duplicate identical signs when no neutral reset occurred (GIRAFFE lowering case)', () => {
    const agg = new TemporalAggregator(0.35, 800);

    // Segment 1: Main sign (ends in still)
    const r1 = agg.processRecognition({ sign: 'giraffe', confidence: 0.95 }, 'still', 1000);
    expect(r1.status).toBe('accepted');
    expect(agg.getConfirmedGlosses()).toEqual(['GIRAFFE']);

    // Segment 2: Lowering tail arriving 350ms later with end_reason='absent'
    const r2 = agg.processRecognition({ sign: 'giraffe', confidence: 0.89 }, 'absent', 1350);
    expect(r2.status).toBe('debounced');
    // Buffer must remain single GIRAFFE
    expect(agg.getConfirmedGlosses()).toEqual(['GIRAFFE']);
  });

  it('accepts intentional repeated signs when neutral reset is confirmed', () => {
    const agg = new TemporalAggregator(0.35, 800);

    // First HELLO
    agg.processRecognition({ sign: 'hello', confidence: 0.90 }, 'still', 1000);
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);

    // Hand returns to neutral rest during inter-sign pause
    agg.notifyNeutralReset();
    expect(agg.getNeutralStatus()).toBe(true);

    // Second intentional HELLO after pause (e.g. t = 1800ms)
    const r2 = agg.processRecognition({ sign: 'hello', confidence: 0.91 }, 'absent', 1800);
    expect(r2.status).toBe('accepted');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO', 'HELLO']);
  });

  it('suppresses transition epenthesis with low confidence within 400ms', () => {
    const agg = new TemporalAggregator(0.35, 800, 400);

    agg.processRecognition({ sign: 'hello', confidence: 0.90 }, 'still', 1000);
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);

    // Epenthesis false motion (e.g. 'mad' at 40% confidence arriving at 250ms without neutral reset)
    const r2 = agg.processRecognition({ sign: 'mad', confidence: 0.40 }, 'absent', 1250);
    expect(r2.status).toBe('debounced');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);
  });

  it('filters low-confidence recognitions (< minConfidence)', () => {
    const agg = new TemporalAggregator(0.35, 800);
    const r = agg.processRecognition({ sign: 'water', confidence: 0.20 }, 'absent', 1000);
    expect(r.status).toBe('low_confidence');
    expect(agg.getConfirmedGlosses()).toEqual([]);
  });

  it('supports interactive manual editing (CRUD)', () => {
    const agg = new TemporalAggregator();

    // 1. Automatic recognition
    agg.processRecognition({ sign: 'hello', confidence: 0.9 }, 'still', 1000);
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);

    // 2. Manual insertion of OOV sign (e.g. FRIEND)
    const manualEntry = agg.addManualGloss('friend');
    expect(manualEntry?.origin).toBe('manual');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO', 'FRIEND']);

    // 3. Edit gloss
    if (manualEntry) {
      agg.replaceGloss(manualEntry.id, 'teacher');
      expect(agg.getConfirmedGlosses()).toEqual(['HELLO', 'TEACHER']);
    }

    // 4. Delete gloss
    if (manualEntry) {
      agg.deleteGloss(manualEntry.id);
      expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);
    }

    // 5. Clear
    agg.clear();
    expect(agg.getConfirmedGlosses()).toEqual([]);
  });

  it('evaluates spatial neutral zone from frame landmarks', () => {
    const agg = new TemporalAggregator();
    const frame = emptyFrame();

    // Set shoulders: left (11) at x=0.6, y=0.5; right (12) at x=0.4, y=0.5
    // midpoint y = 0.5, scale = 0.2, neutral line = 0.5 + 0.15*0.2 = 0.53
    const lSh = (POSE_START + POSE_LEFT_SHOULDER) * 3;
    const rSh = (POSE_START + POSE_RIGHT_SHOULDER) * 3;
    frame[lSh] = 0.6; frame[lSh+1] = 0.5; frame[lSh+2] = 0;
    frame[rSh] = 0.4; frame[rSh+1] = 0.5; frame[rSh+2] = 0;

    // Hand wrist at y = 0.3 (raised high, above shoulder)
    frame[RIGHT_HAND_START * 3] = 0.4;
    frame[RIGHT_HAND_START * 3 + 1] = 0.3;
    frame[RIGHT_HAND_START * 3 + 2] = 0;

    agg.notifyFrame(frame, 1000);
    // Should NOT be neutral
    expect(agg.getNeutralStatus()).toBe(true); // Initially true

    // Now accept a sign (sets neutral to false)
    agg.processRecognition({ sign: 'giraffe', confidence: 0.9 }, 'still', 1050);
    expect(agg.getNeutralStatus()).toBe(false);

    // Frame with hand still high (y=0.35)
    frame[RIGHT_HAND_START * 3 + 1] = 0.35;
    agg.notifyFrame(frame, 1100);
    expect(agg.getNeutralStatus()).toBe(false);

    // Frame with hand lowered to neutral (y=0.65 > neutral_y 0.53)
    frame[RIGHT_HAND_START * 3 + 1] = 0.65;
    agg.notifyFrame(frame, 1200);
    expect(agg.getNeutralStatus()).toBe(true);
  });

  it('handles one-handed signing correctly: elevated hand prevents reset, lowered hand enables reset', () => {
    const agg = new TemporalAggregator();
    agg.processRecognition({ sign: 'hello', confidence: 0.9 }, 'still', 1000);
    expect(agg.getNeutralStatus()).toBe(false);

    const frame = emptyFrame();
    // Valid shoulders
    const lSh = (POSE_START + POSE_LEFT_SHOULDER) * 3;
    const rSh = (POSE_START + POSE_RIGHT_SHOULDER) * 3;
    frame[lSh] = 0.6; frame[lSh + 1] = 0.5; frame[lSh + 2] = 0;
    frame[rSh] = 0.4; frame[rSh + 1] = 0.5; frame[rSh + 2] = 0;

    // Left hand completely absent (NaN). Right hand active signing at y = 0.25 (elevated)
    frame[RIGHT_HAND_START * 3] = 0.4;
    frame[RIGHT_HAND_START * 3 + 1] = 0.25;
    frame[RIGHT_HAND_START * 3 + 2] = 0;

    agg.notifyFrame(frame, 1050);
    // Left hand is missing, but right hand is raised -> must NOT trigger neutral reset
    expect(agg.getNeutralStatus()).toBe(false);

    // Right hand lowered to neutral position (y = 0.70)
    frame[RIGHT_HAND_START * 3 + 1] = 0.70;
    agg.notifyFrame(frame, 1150);
    // Lowered one hand with other absent -> valid neutral rest
    expect(agg.getNeutralStatus()).toBe(true);
  });

  it('does not trigger neutral reset when wrist landmarks are missing/unreliable while hand is in frame', () => {
    const agg = new TemporalAggregator();
    agg.processRecognition({ sign: 'hello', confidence: 0.9 }, 'still', 1000);
    expect(agg.getNeutralStatus()).toBe(false);

    const frame = emptyFrame();
    // Hand is present (e.g. finger tip at index 8 is present, but wrist 0 is NaN)
    frame[(RIGHT_HAND_START + 8) * 3] = 0.4;
    frame[(RIGHT_HAND_START + 8) * 3 + 1] = 0.3;
    frame[(RIGHT_HAND_START + 8) * 3 + 2] = 0;

    // Notify frame: wrists are not valid numbers
    agg.notifyFrame(frame, 1050);
    // Must NOT falsely set neutral reset
    expect(agg.getNeutralStatus()).toBe(false);
  });

  it('handles temporary hand dropouts: < 350ms does not reset, >= 350ms resets', () => {
    const agg = new TemporalAggregator();
    agg.processRecognition({ sign: 'hello', confidence: 0.9 }, 'still', 1000);
    expect(agg.getNeutralStatus()).toBe(false);

    // Short flicker/dropout (100ms)
    agg.notifyFrame(emptyFrame(), 1050);
    expect(agg.getNeutralStatus()).toBe(false);
    agg.notifyFrame(emptyFrame(), 1150);
    expect(agg.getNeutralStatus()).toBe(false);

    // After 350ms total absence (1050ms + 350ms = 1400ms)
    agg.notifyFrame(emptyFrame(), 1401);
    expect(agg.getNeutralStatus()).toBe(true);
  });

  it('fuses pose skeleton wrist when hand mesh wrist is temporarily absent', () => {
    const agg = new TemporalAggregator();
    agg.processRecognition({ sign: 'hello', confidence: 0.9 }, 'still', 1000);
    expect(agg.getNeutralStatus()).toBe(false);

    const frame = emptyFrame();
    // Valid shoulders at y = 0.5
    const lSh = (POSE_START + POSE_LEFT_SHOULDER) * 3;
    const rSh = (POSE_START + POSE_RIGHT_SHOULDER) * 3;
    frame[lSh] = 0.6; frame[lSh + 1] = 0.5; frame[lSh + 2] = 0;
    frame[rSh] = 0.4; frame[rSh + 1] = 0.5; frame[rSh + 2] = 0;

    // Hand mesh wrist (522) is missing (NaN), but pose right wrist (505) is tracked at y = 0.25 (raised)
    const pwR = (POSE_START + POSE_RIGHT_WRIST) * 3;
    frame[pwR] = 0.4;
    frame[pwR + 1] = 0.25;
    frame[pwR + 2] = 0;

    // A finger point is present to indicate hand detected
    frame[(RIGHT_HAND_START + 4) * 3] = 0.4;
    frame[(RIGHT_HAND_START + 4) * 3 + 1] = 0.25;

    agg.notifyFrame(frame, 1050);
    // Pose wrist detected as raised -> neutral reset remains false!
    expect(agg.getNeutralStatus()).toBe(false);

    // Now pose wrist drops to neutral (y = 0.70)
    frame[pwR + 1] = 0.70;
    agg.notifyFrame(frame, 1150);
    expect(agg.getNeutralStatus()).toBe(true);
  });

  it('correctly tracks chest-level signs (e.g. POLICE, FINE) as active signing space', () => {
    const agg = new TemporalAggregator(0.45, 800, 500);
    agg.processRecognition({ sign: 'police', confidence: 0.90 }, 'still', 1000);
    expect(agg.getNeutralStatus()).toBe(false);

    const frame = emptyFrame();
    // Realistic webcam coordinates: shoulders at y=0.582, scale = 0.484
    // Threshold line = 0.582 + 0.55 * 0.484 = 0.848
    const lSh = (POSE_START + POSE_LEFT_SHOULDER) * 3;
    const rSh = (POSE_START + POSE_RIGHT_SHOULDER) * 3;
    frame[lSh] = 0.65; frame[lSh + 1] = 0.582; frame[lSh + 2] = 0;
    frame[rSh] = 0.17; frame[rSh + 1] = 0.582; frame[rSh + 2] = 0;

    // Wrist at chest level (y = 0.75 < 0.848)
    frame[RIGHT_HAND_START * 3] = 0.40;
    frame[RIGHT_HAND_START * 3 + 1] = 0.75;
    frame[RIGHT_HAND_START * 3 + 2] = 0;

    agg.notifyFrame(frame, 1050);
    // MUST NOT reset to neutral because hand is active on the chest!
    expect(agg.getNeutralStatus()).toBe(false);

    // Repeated chest segment (e.g. hesitation or duplicate FINE) without neutral reset
    const r2 = agg.processRecognition({ sign: 'police', confidence: 0.85 }, 'absent', 1400);
    expect(r2.status).toBe('debounced');
    expect(agg.getConfirmedGlosses()).toEqual(['POLICE']);

    // Now hand drops down to lap / desk (y = 0.88 > 0.848)
    frame[RIGHT_HAND_START * 3 + 1] = 0.88;
    agg.notifyFrame(frame, 1600);
    expect(agg.getNeutralStatus()).toBe(true);

    // Intentional repeat after neutral return
    const r3 = agg.processRecognition({ sign: 'police', confidence: 0.88 }, 'absent', 1900);
    expect(r3.status).toBe('accepted');
    expect(agg.getConfirmedGlosses()).toEqual(['POLICE', 'POLICE']);
  });

  it('rejects lowering tail transitions (e.g. PRETEND 55.4%) within epenthesis window', () => {
    const agg = new TemporalAggregator(0.45, 800, 500);

    // 1. Initial sign: HELLO accepted
    const r1 = agg.processRecognition({ sign: 'hello', confidence: 0.92 }, 'still', 1000);
    expect(r1.status).toBe('accepted');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);

    // 2. Lowering tail: Hand dropping from temple, ONNX classifies as PRETEND (55.4%) at 350ms
    const r2 = agg.processRecognition(
      { sign: 'pretend', confidence: 0.554, durationMs: 250 },
      'absent',
      1350
    );
    // Must be suppressed as transition epenthesis
    expect(r2.status).toBe('debounced');
    expect(agg.getConfirmedGlosses()).toEqual(['HELLO']);
  });

  it('bounds diagnostic log to 50 events and exports comprehensive telemetry JSON', () => {
    const agg = new TemporalAggregator(0.45, 800, 500);

    // Generate 60 events
    for (let i = 0; i < 60; i++) {
      agg.processRecognition(
        { sign: `sign_${i}`, confidence: 0.8, rawFrames: 15, durationMs: 400 },
        'absent',
        1000 + i * 1000
      );
    }

    const log = agg.getPipelineLog();
    // Bounded to 50
    expect(log.length).toBe(50);
    expect(log[log.length - 1]?.predictedSign).toBe('SIGN_59');

    // Also record discarded segment
    agg.logDiscardedSegment({
      capturedFrames: 4,
      keptFrames: 4,
      discardReason: 'too-short',
      durationMs: 120,
    });

    const updatedLog = agg.getPipelineLog();
    expect(updatedLog.length).toBe(50);
    expect(updatedLog[updatedLog.length - 1]?.status).toBe('discarded');
    expect(updatedLog[updatedLog.length - 1]?.rejectionReason).toBe('segmenter_discard_too-short');

    // Verify JSON export structure
    const jsonStr = agg.exportPipelineLog();
    const parsed = JSON.parse(jsonStr);
    expect(parsed).toHaveProperty('exportedAt');
    expect(parsed).toHaveProperty('summary');
    expect(parsed.summary.totalEvents).toBe(50);
    expect(Array.isArray(parsed.events)).toBe(true);
  });
});

