/**
 * temporal-aggregator.ts
 * Manages chronological gloss accumulation, confidence filtering,
 * spatial neutral-zone reset debouncing, transition epenthesis rejection,
 * and interactive user editing (manual insert, edit, delete, clear).
 */

import {
  LEFT_HAND_START,
  RIGHT_HAND_START,
  POSE_START,
  POSE_LEFT_SHOULDER,
  POSE_RIGHT_SHOULDER,
  POSE_LEFT_WRIST,
  POSE_RIGHT_WRIST,
  type Frame,
} from '../landmarks';

export interface GlossCandidate {
  label: string;
  probability: number;
}

export interface GlossEntry {
  id: number;
  sign: string;
  confidence: number;
  timestamp: number;
  durationMs: number;
  status: 'accepted' | 'low_confidence' | 'edited' | 'discarded';
  origin: 'recognized' | 'manual' | 'edited';
  top3: GlossCandidate[];
  endReason: string;
}

export interface AggregatorDiagnostic {
  action: 'accepted' | 'debounced' | 'flagged_low_confidence' | 'discarded';
  sign: string;
  confidence: number;
  timestamp: number;
  endReason?: string | undefined;
  reason?: string | undefined;
}

export interface PipelineDiagnosticEvent {
  id: number;
  timestamp: number;
  timeString: string;
  rawFrames: number;
  resampledFrames?: number | undefined;
  durationMs: number;
  endReason: string;
  top3: GlossCandidate[];
  predictedSign: string;
  confidence: number;
  status: 'accepted' | 'debounced' | 'low_confidence' | 'discarded';
  rejectionReason?: string | undefined;
  neutralResetState: boolean;
  debounceTimerActive: boolean;
  lastAcceptedSign: string | null;
}

export class TemporalAggregator {
  private static readonly MAX_LOG_SIZE = 50;
  private minConfidence: number;
  private holdDebounceMs: number;
  private epenthesisWindowMs: number;

  private entries: GlossEntry[] = [];
  private nextId = 1;
  private nextLogId = 1;
  private lastAcceptedSign: string | null = null;
  private lastAcceptedTime: number | null = null;
  private hasNeutralReset = true;
  private lastHandsAbsentTime: number | null = null;
  private diagnosticHistory: AggregatorDiagnostic[] = [];
  private pipelineLog: PipelineDiagnosticEvent[] = [];

  constructor(
    minConfidence = 0.45,
    holdDebounceMs = 800.0,
    epenthesisWindowMs = 500.0
  ) {
    this.minConfidence = minConfidence;
    this.holdDebounceMs = holdDebounceMs;
    this.epenthesisWindowMs = epenthesisWindowMs;
  }

  reset(): void {
    this.entries = [];
    this.nextId = 1;
    this.lastAcceptedSign = null;
    this.lastAcceptedTime = null;
    this.hasNeutralReset = true;
    this.lastHandsAbsentTime = null;
    this.diagnosticHistory = [];
  }

  clearPipelineLog(): void {
    this.pipelineLog = [];
    this.diagnosticHistory = [];
  }

  getPipelineLog(): PipelineDiagnosticEvent[] {
    return [...this.pipelineLog];
  }

  exportPipelineLog(): string {
    return JSON.stringify(
      {
        exportedAt: new Date().toISOString(),
        summary: {
          totalEvents: this.pipelineLog.length,
          accepted: this.pipelineLog.filter((e) => e.status === 'accepted').length,
          debounced: this.pipelineLog.filter((e) => e.status === 'debounced').length,
          lowConfidence: this.pipelineLog.filter((e) => e.status === 'low_confidence').length,
          discarded: this.pipelineLog.filter((e) => e.status === 'discarded').length,
        },
        confirmedGlosses: this.getConfirmedGlosses(),
        events: this.pipelineLog,
      },
      null,
      2
    );
  }

  private addPipelineLog(event: Omit<PipelineDiagnosticEvent, 'id'>): void {
    const entry: PipelineDiagnosticEvent = {
      id: this.nextLogId++,
      ...event,
    };
    this.pipelineLog.push(entry);
    if (this.pipelineLog.length > TemporalAggregator.MAX_LOG_SIZE) {
      this.pipelineLog.shift();
    }
  }

  getConfirmedGlosses(): string[] {
    return this.entries
      .filter((e) => e.status === 'accepted' || e.status === 'edited')
      .map((e) => e.sign.toUpperCase());
  }

  getEntries(): GlossEntry[] {
    return [...this.entries];
  }

  getDiagnosticHistory(): AggregatorDiagnostic[] {
    return [...this.diagnosticHistory];
  }

  getNeutralStatus(): boolean {
    return this.hasNeutralReset;
  }

  /**
   * Evaluates real-time landmark coordinates to track spatial neutral rest return.
   * Uses dual wrist tracking (both hand mesh 468/522 and pose skeleton 504/505).
   */
  notifyFrame(frame: Frame, timestamp?: number): void {
    const now = timestamp ?? performance.now();

    // Shoulders: pose landmarks 11 & 12
    const lShIdx = (POSE_START + POSE_LEFT_SHOULDER) * 3;
    const rShIdx = (POSE_START + POSE_RIGHT_SHOULDER) * 3;
    const lx = frame[lShIdx] ?? Number.NaN;
    const ly = frame[lShIdx + 1] ?? Number.NaN;
    const rx = frame[rShIdx] ?? Number.NaN;
    const ry = frame[rShIdx + 1] ?? Number.NaN;

    const shouldersValid =
      Number.isFinite(lx) &&
      Number.isFinite(ly) &&
      Number.isFinite(rx) &&
      Number.isFinite(ry);

    const neutralThresholdY = shouldersValid
      ? (ly + ry) / 2.0 + 0.55 * Math.max(Math.hypot(lx - rx, ly - ry), 0.1)
      : 0.82;

    // Hand mesh wrists (468, 522)
    const hwLy = frame[LEFT_HAND_START * 3 + 1];
    const hwRy = frame[RIGHT_HAND_START * 3 + 1];

    // Pose skeleton wrists (504, 505)
    const pwLy = frame[(POSE_START + POSE_LEFT_WRIST) * 3 + 1];
    const pwRy = frame[(POSE_START + POSE_RIGHT_WRIST) * 3 + 1];

    // Take higher elevation (minimum y) between hand mesh and pose wrist
    const leftYs = [hwLy, pwLy].filter((y): y is number => y !== undefined && Number.isFinite(y));
    const rightYs = [hwRy, pwRy].filter((y): y is number => y !== undefined && Number.isFinite(y));

    const lwValid = leftYs.length > 0;
    const rwValid = rightYs.length > 0;
    const lwy = lwValid ? Math.min(...leftYs) : Number.NaN;
    const rwy = rwValid ? Math.min(...rightYs) : Number.NaN;

    const anyHandElevated =
      (lwValid && lwy <= neutralThresholdY) ||
      (rwValid && rwy <= neutralThresholdY);

    if (anyHandElevated) {
      this.lastHandsAbsentTime = null;
      return;
    }

    // Neither hand is elevated in the signing space.
    // Case 1: At least one wrist is tracked and resting in the neutral zone.
    if (lwValid || rwValid) {
      this.lastHandsAbsentTime = null;
      this.hasNeutralReset = true;
      return;
    }

    // Case 2: Wrists are completely absent from view.
    // Require >= 350ms of absence (strictly greater than segmenter gap_ms of 300ms)
    // to ensure segment close is processed before neutral reset activates.
    if (this.lastHandsAbsentTime === null) {
      this.lastHandsAbsentTime = now;
    }
    if (now - this.lastHandsAbsentTime >= 350.0) {
      this.hasNeutralReset = true;
    }
  }

  notifyNeutralReset(): void {
    this.hasNeutralReset = true;
  }

  /**
   * Process a new isolated sign recognition event.
   * Debounces duplicate lowering tails and transition epenthesis.
   */
  processRecognition(
    recognition: {
      sign: string;
      confidence: number;
      durationMs?: number;
      top3?: GlossCandidate[];
      rawFrames?: number;
      resampledFrames?: number;
    },
    endReason = 'absent',
    timestamp?: number
  ): { status: 'accepted' | 'debounced' | 'low_confidence'; entry: GlossEntry | null } {
    const now = timestamp ?? performance.now();
    const sign = recognition.sign;
    const confidence = recognition.confidence;
    const durationMs = recognition.durationMs ?? 0;
    const top3 = recognition.top3 ?? [];
    const rawFrames = recognition.rawFrames ?? 0;
    const resampledFrames = recognition.resampledFrames;

    const neutralResetState = this.hasNeutralReset;
    const debounceTimerActive =
      this.lastAcceptedTime !== null &&
      now - this.lastAcceptedTime < this.holdDebounceMs;
    const lastAccepted = this.lastAcceptedSign;
    const timeStr = new Date().toISOString().substring(11, 23);

    // 1. Debounce duplicate identical sign without neutral reset
    // Handles BOTH end_reason == 'still' (hold) and end_reason == 'absent' (hand lowering exit)
    if (
      this.lastAcceptedSign === sign &&
      this.lastAcceptedTime !== null &&
      now - this.lastAcceptedTime < this.holdDebounceMs
    ) {
      if (!this.hasNeutralReset) {
        const last = this.entries[this.entries.length - 1];
        if (last && last.status === 'accepted') {
          last.durationMs += durationMs;
        }
        const diag: AggregatorDiagnostic = {
          action: 'debounced',
          sign,
          confidence,
          timestamp: now,
          endReason,
          reason: 'identical_sign_without_neutral_reset',
        };
        this.diagnosticHistory.push(diag);
        this.addPipelineLog({
          timestamp: now,
          timeString: timeStr,
          rawFrames,
          resampledFrames,
          durationMs,
          endReason,
          top3,
          predictedSign: sign.toUpperCase(),
          confidence,
          status: 'debounced',
          rejectionReason: 'identical_sign_without_neutral_reset',
          neutralResetState,
          debounceTimerActive,
          lastAcceptedSign: lastAccepted,
        });
        return { status: 'debounced', entry: last ?? null };
      }
    }

    // 2. Filter Transitional Movement Epenthesis
    // Transitions within epenthesisWindowMs (500ms) without neutral reset
    // require confidence >= 0.60 to reject lowering tails and transit artifacts.
    if (
      this.lastAcceptedTime !== null &&
      now - this.lastAcceptedTime < this.epenthesisWindowMs &&
      !this.hasNeutralReset &&
      confidence < 0.60
    ) {
      const diag: AggregatorDiagnostic = {
        action: 'debounced',
        sign,
        confidence,
        timestamp: now,
        endReason,
        reason: 'transition_epenthesis_suppressed',
      };
      this.diagnosticHistory.push(diag);
      this.addPipelineLog({
        timestamp: now,
        timeString: timeStr,
        rawFrames,
        resampledFrames,
        durationMs,
        endReason,
        top3,
        predictedSign: sign.toUpperCase(),
        confidence,
        status: 'debounced',
        rejectionReason: `transition_epenthesis_suppressed (conf ${(confidence * 100).toFixed(1)}% < 60%)`,
        neutralResetState,
        debounceTimerActive,
        lastAcceptedSign: lastAccepted,
      });
      return { status: 'debounced', entry: null };
    }

    // 3. Confidence Thresholding
    if (confidence < this.minConfidence) {
      const entry: GlossEntry = {
        id: this.nextId++,
        sign: sign.toUpperCase(),
        confidence,
        timestamp: now,
        durationMs,
        status: 'low_confidence',
        origin: 'recognized',
        top3,
        endReason,
      };
      this.entries.push(entry);
      this.diagnosticHistory.push({
        action: 'flagged_low_confidence',
        sign,
        confidence,
        timestamp: now,
      });
      this.addPipelineLog({
        timestamp: now,
        timeString: timeStr,
        rawFrames,
        resampledFrames,
        durationMs,
        endReason,
        top3,
        predictedSign: sign.toUpperCase(),
        confidence,
        status: 'low_confidence',
        rejectionReason: `confidence ${(confidence * 100).toFixed(1)}% < min ${(this.minConfidence * 100).toFixed(0)}%`,
        neutralResetState,
        debounceTimerActive,
        lastAcceptedSign: lastAccepted,
      });
      return { status: 'low_confidence', entry };
    }

    // 4. Accepted Distinct Sign
    const entry: GlossEntry = {
      id: this.nextId++,
      sign: sign.toUpperCase(),
      confidence,
      timestamp: now,
      durationMs,
      status: 'accepted',
      origin: 'recognized',
      top3,
      endReason,
    };
    this.entries.push(entry);
    this.lastAcceptedSign = sign;
    this.lastAcceptedTime = now;
    this.hasNeutralReset = false; // Require neutral reset before duplicate identical sign can be accepted

    this.diagnosticHistory.push({
      action: 'accepted',
      sign,
      confidence,
      timestamp: now,
      endReason,
    });
    this.addPipelineLog({
      timestamp: now,
      timeString: timeStr,
      rawFrames,
      resampledFrames,
      durationMs,
      endReason,
      top3,
      predictedSign: sign.toUpperCase(),
      confidence,
      status: 'accepted',
      neutralResetState,
      debounceTimerActive,
      lastAcceptedSign: lastAccepted,
    });

    return { status: 'accepted', entry };
  }

  logDiscardedSegment(telemetry: {
    capturedFrames?: number | undefined;
    keptFrames?: number | undefined;
    resampledFrames?: number | undefined;
    discardReason?: string | null | undefined;
    durationMs?: number | undefined;
    fps?: number | undefined;
    timestamp?: Date | undefined;
  }): void {
    const now = performance.now();
    const timeStr = telemetry.timestamp
      ? telemetry.timestamp.toISOString().substring(11, 23)
      : new Date().toISOString().substring(11, 23);
    const rawFrames = telemetry.capturedFrames ?? telemetry.keptFrames ?? 0;
    this.addPipelineLog({
      timestamp: now,
      timeString: timeStr,
      rawFrames,
      resampledFrames: telemetry.resampledFrames,
      durationMs: telemetry.durationMs ?? 0,
      endReason: telemetry.discardReason ?? 'discarded',
      top3: [],
      predictedSign: 'NONE',
      confidence: 0,
      status: 'discarded',
      rejectionReason: `segmenter_discard_${telemetry.discardReason ?? 'noise'}`,
      neutralResetState: this.hasNeutralReset,
      debounceTimerActive:
        this.lastAcceptedTime !== null &&
        now - this.lastAcceptedTime < this.holdDebounceMs,
      lastAcceptedSign: this.lastAcceptedSign,
    });
  }

  logUncertainRecognition(info: {
    top3: GlossCandidate[];
    durationMs?: number | undefined;
    rawFrames?: number | undefined;
    resampledFrames?: number | undefined;
    endReason?: string | undefined;
    partial?: boolean | undefined;
    reason?: string | undefined;
  }): void {
    const now = performance.now();
    const top = info.top3[0];
    this.addPipelineLog({
      timestamp: now,
      timeString: new Date().toISOString().substring(11, 23),
      rawFrames: info.rawFrames ?? 0,
      resampledFrames: info.resampledFrames,
      durationMs: info.durationMs ?? 0,
      endReason: info.endReason ?? 'absent',
      top3: info.top3,
      predictedSign: top?.label.toUpperCase() ?? 'NOT_SURE',
      confidence: top?.probability ?? 0,
      status: 'low_confidence',
      rejectionReason:
        info.reason ??
        (info.partial ? 'partial_framing' : 'decision_kind_not_sure'),
      neutralResetState: this.hasNeutralReset,
      debounceTimerActive:
        this.lastAcceptedTime !== null &&
        now - this.lastAcceptedTime < this.holdDebounceMs,
      lastAcceptedSign: this.lastAcceptedSign,
    });
  }

  // Interactive Buffer Operations
  addManualGloss(sign: string): GlossEntry | null {
    const trimmed = sign.trim().toUpperCase();
    if (!trimmed) return null;
    const entry: GlossEntry = {
      id: this.nextId++,
      sign: trimmed,
      confidence: 1.0,
      timestamp: performance.now(),
      durationMs: 0,
      status: 'accepted',
      origin: 'manual',
      top3: [{ label: trimmed.toLowerCase(), probability: 1.0 }],
      endReason: 'manual',
    };
    this.entries.push(entry);
    this.hasNeutralReset = true;
    return entry;
  }

  replaceGloss(id: number, newSign: string): boolean {
    const trimmed = newSign.trim().toUpperCase();
    if (!trimmed) return false;
    const idx = this.entries.findIndex((e) => e.id === id);
    if (idx === -1) return false;
    const entry = this.entries[idx];
    if (!entry) return false;
    entry.sign = trimmed;
    entry.status = 'edited';
    entry.origin = 'edited';
    return true;
  }

  deleteGloss(id: number): boolean {
    const idx = this.entries.findIndex((e) => e.id === id);
    if (idx === -1) return false;
    this.entries.splice(idx, 1);
    return true;
  }

  deleteLast(): GlossEntry | null {
    return this.entries.pop() ?? null;
  }

  clear(): void {
    this.entries = [];
    this.lastAcceptedSign = null;
    this.lastAcceptedTime = null;
    this.hasNeutralReset = true;
  }
}
