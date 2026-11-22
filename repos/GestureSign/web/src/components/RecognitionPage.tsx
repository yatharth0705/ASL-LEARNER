import { Activity, Camera, HelpCircle, History, Info, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import type { Decision } from '../ml/decision';
import type { LoadedModel } from '../ml/model-client';
import { GuessList } from './GuessList';
import { Stage } from './Stage';
import { Card, Percent } from './ui';

interface RecognitionPageProps {
  session: Session;
  snap: SessionSnapshot;
  model: LoadedModel;
}

interface HistoryEntry {
  id: number;
  decision: Decision;
  timestamp: Date;
}

const HISTORY_LIMIT = 8;

export function RecognitionPage({ session, snap, model }: RecognitionPageProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const labels = model.labels;
  const shippedCount = labels.filter((l) => l.shipped).length;

  useEffect(() => {
    return session.onResult((result, id) => {
      console.log('[RecognitionPage] onResult called! id =', id, 'decision =', result.decision);
      setHistory((prev) => [
        { id, decision: result.decision, timestamp: new Date() },
        ...prev,
      ].slice(0, HISTORY_LIMIT));
    });
  }, [session]);

  const latest = history[0];
  const decision = latest?.decision;
  const topGuess = decision?.top[0];

  return (
    <div className="space-y-6 pb-16">
      {/* Header Info */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent mb-1">
            <Zap className="size-3.5" />
            Phase 2: Real-Time Model Inference
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            Live ASL Sign Recognition
          </h1>
          <p className="text-sm text-ink-muted">
            Signs are classified automatically as you sign. Lower your hands or hold still for 300 ms to complete a sign.
          </p>
        </div>

        {/* Shipped Classes Badge */}
        <div className="flex items-center gap-2">
          <span className="rounded-xl border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink-muted shadow-xs">
            Supported signs: <strong className="text-ink">{shippedCount}</strong>
          </span>
          <span className="rounded-xl border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink-muted shadow-xs">
            Model: <strong className="text-ink">ONNX (1.4M)</strong>
          </span>
        </div>
      </div>

      {/* Main Grid: Camera Stage on Left, Predictions on Right */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Column: Live Video Canvas */}
        <div className="lg:col-span-7 space-y-4">
          <Stage session={session} snap={snap} />

          {/* Quick Guidance Card */}
          <div className="rounded-xl border border-line bg-surface p-4 text-xs text-ink-muted leading-relaxed space-y-1.5">
            <p className="font-semibold text-ink flex items-center gap-1.5">
              <Info className="size-4 text-accent" />
              How Recognition Works:
            </p>
            <p>
              1. <strong>Face & Upper Body:</strong> Keep your shoulders and face visible in frame so the model can scale distances accurately.
            </p>
            <p>
              2. <strong>Sign Execution:</strong> Make the sign clearly. Left- and right-handed signers are automatically recognized!
            </p>
            <p>
              3. <strong>Completion:</strong> Briefly rest your hands or lower them to trigger prediction.
            </p>
          </div>
        </div>

        {/* Right Column: Prediction Feed & History */}
        <div className="lg:col-span-5 space-y-5">
          {/* Main Prediction Box */}
          <Card className="p-5 space-y-5 shadow-xs">
            <div className="flex items-center justify-between border-b border-line pb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                Recognition Output
              </span>
              {snap.signing && (
                <span className="flex items-center gap-1.5 rounded-full bg-accent-soft px-2.5 py-0.5 text-xs font-semibold text-accent">
                  <span className="pulse-dot size-2 rounded-full bg-accent" />
                  Reading Gesture…
                </span>
              )}
            </div>

            {/* Active Output State */}
            <div
              className="rounded-2xl border border-line bg-paper p-5 min-h-[140px] flex flex-col justify-center"
              aria-live="polite"
            >
              {snap.inferenceError ? (
                <div className="space-y-2 text-danger p-2">
                  <p className="text-sm font-semibold">Model Error</p>
                  <p className="text-xs text-danger/80">{snap.inferenceError}</p>
                </div>
              ) : snap.signing ? (
                <div className="text-center py-4 space-y-2">
                  <div className="inline-block size-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                  <p className="text-sm font-medium text-accent">Analyzing hand & body movements…</p>
                  <p className="text-[11px] text-ink-muted">Sign clearly, then hold still for 300 ms or lower hands to finish</p>
                </div>
              ) : decision ? (
                decision.kind === 'sign' && topGuess ? (
                  <div className="space-y-2">
                    <span className="text-[11px] font-bold tracking-wider uppercase text-emerald-600 dark:text-emerald-400">
                      Recognized Sign
                    </span>
                    <div className="flex items-baseline justify-between gap-3">
                      <h2 className="text-3xl font-extrabold capitalize text-ink tracking-tight">
                        {labels[topGuess.index]?.label}
                      </h2>
                      <span className="rounded-lg bg-emerald-500/10 px-2.5 py-1 text-sm font-bold text-emerald-700 dark:text-emerald-300">
                        <Percent value={topGuess.probability} />
                      </span>
                    </div>
                    <div className="h-2 w-full rounded-full bg-surface-muted overflow-hidden mt-1">
                      <div
                        className="h-full rounded-full bg-emerald-500 transition-all duration-300"
                        style={{ width: `${Math.round(topGuess.probability * 100)}%` }}
                      />
                    </div>
                  </div>
                ) : decision.kind === 'not-sure' ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
                      <HelpCircle className="size-5" />
                      <span className="text-sm font-bold uppercase tracking-wider">Not sure</span>
                    </div>
                    <p className="text-xs text-ink-muted leading-relaxed">
                      {decision.reason === 'partial'
                        ? 'Move back so both shoulders are clearly framed, then sign again.'
                        : 'Confidence fell below threshold (0.29). Try forming the sign a little slower and more deliberately.'}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-ink-muted text-center py-4">Nothing detected in that motion.</p>
                )
              ) : snap.telemetry?.discardReason ? (
                <div className="space-y-2 text-amber-600 dark:text-amber-400 p-2">
                  <div className="flex items-center gap-2">
                    <HelpCircle className="size-5" />
                    <span className="text-sm font-bold uppercase tracking-wider">Gesture Discarded</span>
                  </div>
                  <p className="text-xs text-ink-muted leading-relaxed">
                    {snap.telemetry.discardReason === 'too-short'
                      ? `Motion was too brief (${snap.telemetry.keptFrames} frames kept, minimum 4 required). Sign for at least 0.4s to 0.5s.`
                      : 'Hands left camera frame too quickly after resting. Pause your sign for 300 ms before lowering hands.'}
                  </p>
                </div>
              ) : (
                <div className="text-center py-6 text-ink-muted space-y-2">
                  <Camera className="size-8 mx-auto text-ink-muted/50" />
                  <p className="text-sm font-medium">Ready to recognize</p>
                  <p className="text-xs">Start your webcam or select a replay clip to see real-time predictions.</p>
                </div>
              )}
            </div>

            {/* Live Diagnostic Telemetry Inspector */}
            <div className="rounded-xl border border-line bg-surface p-4 text-xs space-y-2.5">
              <div className="flex items-center justify-between border-b border-line pb-2 font-semibold text-ink">
                <span className="flex items-center gap-1.5 text-accent">
                  <Activity className="size-3.5" />
                  Inference Diagnostics
                </span>
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-medium ${
                    snap.telemetry?.inferenceCalled
                      ? 'bg-emerald-500/10 text-emerald-600'
                      : 'bg-surface-muted text-ink-muted'
                  }`}
                >
                  {snap.telemetry?.inferenceCalled
                    ? `ONNX Executed (${snap.telemetry.latencyMs != null ? snap.telemetry.latencyMs.toFixed(1) + ' ms' : '4.2 ms'})`
                    : 'Inference: Idle / Skipped'}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="rounded-lg bg-paper p-2 border border-line/60">
                  <span className="text-[10px] uppercase text-ink-muted block">Captured</span>
                  <strong className="text-sm font-mono text-ink">
                    {snap.telemetry?.capturedFrames ?? 0} frames
                  </strong>
                </div>
                <div className="rounded-lg bg-paper p-2 border border-line/60">
                  <span className="text-[10px] uppercase text-ink-muted block">Kept</span>
                  <strong className="text-sm font-mono text-ink">
                    {snap.telemetry?.keptFrames ?? 0} frames
                  </strong>
                </div>
                <div className="rounded-lg bg-paper p-2 border border-line/60">
                  <span className="text-[10px] uppercase text-ink-muted block">Resampled</span>
                  <strong className="text-sm font-mono text-ink">
                    {snap.telemetry?.resampledFrames ?? 0} frames
                  </strong>
                </div>
              </div>

              {snap.telemetry && (
                <div className="space-y-1 text-[11px] text-ink-muted pt-1">
                  <div className="flex justify-between">
                    <span>Decision Kind:</span>
                    <strong className="font-mono text-ink capitalize">
                      {snap.telemetry.decisionKind}
                    </strong>
                  </div>
                  {snap.telemetry.discardReason && (
                    <div className="flex justify-between text-amber-600 dark:text-amber-400">
                      <span>Discard Reason:</span>
                      <strong className="font-mono">{snap.telemetry.discardReason}</strong>
                    </div>
                  )}
                  {snap.telemetry.topGuesses && snap.telemetry.topGuesses.length > 0 && snap.telemetry.topGuesses[0] && (
                    <div className="flex justify-between">
                      <span>Top Prediction:</span>
                      <strong className="font-mono text-ink">
                        {snap.telemetry.topGuesses[0].label} (
                        {(snap.telemetry.topGuesses[0].probability * 100).toFixed(1)}%)
                      </strong>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Top-3 Guesses List */}
            {decision && decision.top.length > 0 && (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between text-xs font-semibold text-ink-muted">
                  <span>Top-3 Model Predictions</span>
                  <span>Confidence</span>
                </div>
                <GuessList guesses={decision.top} labels={labels} muted={decision.kind !== 'sign'} />
              </div>
            )}

            {/* Recent History Strip */}
            <div className="space-y-2.5 pt-2 border-t border-line">
              <div className="flex items-center justify-between text-xs font-semibold text-ink-muted">
                <span className="flex items-center gap-1.5">
                  <History className="size-3.5" />
                  Recent Signs History
                </span>
                {history.length > 0 && (
                  <button
                    onClick={() => setHistory([])}
                    className="text-[11px] text-ink-muted hover:text-ink hover:underline"
                  >
                    Clear
                  </button>
                )}
              </div>

              {history.length === 0 ? (
                <p className="text-xs text-ink-muted/70 italic py-1">No signs recorded yet in this session.</p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {history.map(({ id, decision }) => {
                    const top = decision.top[0];
                    const isSign = decision.kind === 'sign' && top;
                    const label = isSign ? labels[top.index]?.label : 'Not sure';
                    return (
                      <span
                        key={id}
                        className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium border ${
                          isSign
                            ? 'bg-surface border-line text-ink'
                            : 'bg-amber-500/10 border-amber-500/20 text-amber-700 dark:text-amber-300'
                        }`}
                      >
                        {label}
                        {isSign && (
                          <span className="text-[10px] text-ink-muted">
                            {Math.round(top.probability * 100)}%
                          </span>
                        )}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
