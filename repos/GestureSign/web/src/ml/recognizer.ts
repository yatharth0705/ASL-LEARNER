/**
 * The live pipeline, independent of where frames come from (camera, Replay or
 * the e2e injection hook): frame status -> segmenter -> features -> model ->
 * decision. Inference is injected so the logic is unit-testable without
 * workers or WASM.
 */
import { decide, type Decision } from './decision';
import { modelInput, type ModelInput } from './features';
import type { Frame } from './landmarks';
import type { ModelMeta, SignLabel } from './meta';
import { Segmenter, frameStatus, type FrameStatus, type Segment, type SegmenterOptions, type EndReason } from './segmenter';

export interface Inference {
  logits: Float32Array;
  embedding: Float32Array;
  ms?: number;
}

export type Infer = (input: ModelInput) => Promise<Inference>;

export interface RecognitionResult {
  decision: Decision;
  inference: Inference;
  segment: Segment;
  /** True when the signer's shoulders were out of view for most of the sign. */
  partial: boolean;
  endReason?: EndReason | undefined;
}

export interface DiagnosticTelemetry {
  capturedFrames: number;
  keptFrames: number;
  resampledFrames: number;
  inferenceCalled: boolean;
  latencyMs?: number | undefined;
  topGuesses: Array<{ label: string; probability: number }>;
  decisionKind: string;
  discardReason?: string | null | undefined;
  fps: number;
  durationMs: number;
  timestamp: Date;
}

export type RecognizerEvent =
  | { type: 'status'; status: FrameStatus }
  | { type: 'signing' }
  | { type: 'discarded'; telemetry?: DiagnosticTelemetry }
  | { type: 'result'; result: RecognitionResult; telemetry?: DiagnosticTelemetry }
  | { type: 'error'; message: string };

export interface TrackedFrame {
  frame: Frame;
  /** Milliseconds, monotonic. */
  t: number;
  /** People the pose landmarker found (0, 1 or more). */
  poses: number;
}

export class Recognizer {
  private readonly segmenter: Segmenter;
  private statuses: FrameStatus[] = [];
  private lastStatus: FrameStatus | null = null;
  private generation = 0;
  private signingFrames = 0;

  constructor(
    private readonly meta: ModelMeta,
    private readonly labels: readonly SignLabel[],
    private readonly infer: Infer,
    private readonly emit: (event: RecognizerEvent) => void,
    segmenterOptions?: Partial<SegmenterOptions>,
  ) {
    this.segmenter = new Segmenter(segmenterOptions);
  }

  get shipped(): boolean[] {
    return this.labels.map((l) => l.shipped);
  }

  reset(): void {
    this.generation++;
    this.segmenter.reset();
    this.statuses = [];
    this.lastStatus = null;
    this.signingFrames = 0;
  }

  push({ frame, t, poses }: TrackedFrame): void {
    const status = frameStatus(frame, poses);
    if (status !== this.lastStatus) {
      this.lastStatus = status;
      this.emit({ type: 'status', status });
    }
    const event = this.segmenter.push(frame, t, { crowded: status === 'crowded' });
    if (this.segmenter.state === 'signing') {
      this.statuses.push(status);
      this.signingFrames++;
    }
    this.handle(event);
  }

  /** End an open sign now, e.g. when a replay clip finishes. */
  flush(): void {
    this.handle(this.segmenter.flush());
  }

  private handle(event: ReturnType<Segmenter['push']>): void {
    if (!event) return;
    console.log('[Recognizer.handle] Event:', event.type, event);
    if (event.type === 'start') {
      this.statuses = [];
      this.signingFrames = 1;
      this.emit({ type: 'signing' });
      return;
    }
    const statuses = this.statuses;
    const capturedTotal = this.signingFrames;
    this.statuses = [];
    this.signingFrames = 0;
    if (event.type === 'discard') {
      console.warn('[Recognizer.handle] Discarding segment: reason =', event.reason);
      const telemetry: DiagnosticTelemetry = {
        capturedFrames: capturedTotal,
        keptFrames: this.segmenter.lastKeptCount,
        resampledFrames: 0,
        inferenceCalled: false,
        topGuesses: [],
        decisionKind: 'discarded',
        discardReason: event.reason,
        fps: 0,
        durationMs: 0,
        timestamp: new Date(),
      };
      this.emit({ type: 'discarded', telemetry });
      return;
    }
    // Vote only over frames where the signer's hands were seen: frames without
    // hands (the gap that ends a sign, short dropouts) say nothing about framing.
    const partialFrames = statuses.filter((s) => s === 'partial').length;
    const framedFrames = partialFrames + statuses.filter((s) => s === 'ok').length;
    const partial = partialFrames > framedFrames / 2;
    console.log(
      `[Recognizer.handle] Segment ended (${event.reason}): rawFrames=${event.segment.frames.length}, fps=${event.segment.fps.toFixed(1)}, duration=${event.segment.durationMs.toFixed(0)}ms, partial=${partial}`,
    );
    void this.classify(event.segment, partial, this.generation, capturedTotal, event.reason);
  }

  private async classify(
    segment: Segment,
    partial: boolean,
    generation: number,
    capturedTotal = 0,
    endReason?: EndReason,
  ): Promise<void> {
    const input = modelInput(segment.frames, segment.fps, this.meta);
    console.log(
      `[Recognizer.classify] input.frames=${input.frames}, minFrames=${this.meta.minFrames}, dim=${input.dim}`,
    );
    if (input.frames < this.meta.minFrames) {
      console.warn(
        `[Recognizer.classify] DISCARDED: input.frames (${input.frames}) < minFrames (${this.meta.minFrames})`,
      );
      const telemetry: DiagnosticTelemetry = {
        capturedFrames: capturedTotal || segment.frames.length,
        keptFrames: segment.frames.length,
        resampledFrames: input.frames,
        inferenceCalled: false,
        topGuesses: [],
        decisionKind: 'discarded',
        discardReason: 'too-short',
        fps: segment.fps,
        durationMs: segment.durationMs,
        timestamp: new Date(),
      };
      this.emit({ type: 'discarded', telemetry });
      return;
    }
    try {
      console.log(
        `[Recognizer.classify] INFERENCE REQUEST: input tensor shape = [1, ${input.frames}, ${input.dim}]`,
      );
      const inference = await this.infer(input);
      console.log(
        `[Recognizer.classify] INFERENCE RESPONSE: logits shape = [1, ${inference.logits.length}], embedding shape = [1, ${inference.embedding.length}], latency = ${inference.ms != null ? inference.ms.toFixed(1) + 'ms' : 'n/a'}`,
      );
      if (generation !== this.generation) {
        console.warn(`[Recognizer.classify] Stale inference ignored (generation ${generation} !== ${this.generation})`);
        return; // the mode or source changed meanwhile
      }
      const decision = decide(inference.logits, this.shipped, this.meta.calibration, { partial });
      const top3Formatted = decision.top.map(
        (t, i) => `${i + 1}. ${this.labels[t.index]?.label ?? `#${t.index}`}: ${(t.probability * 100).toFixed(1)}%`,
      );
      console.log(
        `[Recognizer.classify] DECISION: kind=${decision.kind}, reason=${decision.reason ?? 'none'}, Top-3: ${top3Formatted.join(', ')}`,
      );
      const telemetry: DiagnosticTelemetry = {
        capturedFrames: capturedTotal || segment.frames.length,
        keptFrames: segment.frames.length,
        resampledFrames: input.frames,
        inferenceCalled: true,
        latencyMs: inference.ms,
        topGuesses: decision.top.map((t) => ({
          label: this.labels[t.index]?.label ?? '#' + t.index,
          probability: t.probability,
        })),
        decisionKind: decision.kind,
        discardReason: null,
        fps: segment.fps,
        durationMs: segment.durationMs,
        timestamp: new Date(),
      };
      this.emit({ type: 'result', result: { decision, inference, segment, partial, endReason }, telemetry });
    } catch (error) {
      console.error('[Recognizer.classify] ERROR during inference:', error);
      this.emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
  }
}
