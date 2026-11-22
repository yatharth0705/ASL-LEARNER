import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
  Info,
  Languages,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
  Volume2,
  X,
  Zap,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import type { LoadedModel } from '../ml/model-client';
import {
  TemporalAggregator,
  type GlossEntry,
  type PipelineDiagnosticEvent,
} from '../ml/continuous/temporal-aggregator';
import { GlossaBARTClient } from '../ml/continuous/translator-client';
import { Stage } from './Stage';
import { Card } from './ui';

interface ContinuousTranslationPageProps {
  session: Session;
  snap: SessionSnapshot;
  model: LoadedModel;
}

export function ContinuousTranslationPage({
  session,
  snap,
  model,
}: ContinuousTranslationPageProps) {
  const aggregatorRef = useRef<TemporalAggregator>(new TemporalAggregator(0.45, 800, 500));
  const translatorRef = useRef<GlossaBARTClient>(new GlossaBARTClient());

  const [entries, setEntries] = useState<GlossEntry[]>([]);
  const [diagnosticLogs, setDiagnosticLogs] = useState<PipelineDiagnosticEvent[]>([]);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [diagCopied, setDiagCopied] = useState(false);

  const [manualInput, setManualInput] = useState('');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingText, setEditingText] = useState('');
  const [translation, setTranslation] = useState<string>('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Poll server health on mount
  useEffect(() => {
    let mounted = true;
    const checkServer = async () => {
      const ok = await translatorRef.current.checkHealth();
      if (mounted) setServerOnline(ok);
    };
    void checkServer();
    const interval = setInterval(() => void checkServer(), 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // Wire frame landmarks to spatial neutral-zone tracker
  useEffect(() => {
    return session.onFrame((frame) => {
      aggregatorRef.current.notifyFrame(frame);
    });
  }, [session]);

  // Wire recognition results and discarded events to temporal aggregator and diagnostics
  useEffect(() => {
    const unsubResult = session.onResult((result) => {
      const top3Candidates = result.decision.top.map((g) => ({
        label: model.labels[g.index]?.label ?? 'unknown',
        probability: g.probability,
      }));
      const endReason = result.endReason ?? 'absent';
      const rawFrames = result.segment.frames.length;
      const durationMs = result.segment.durationMs;

      if (result.decision.kind === 'sign') {
        const top = top3Candidates[0];
        if (top && top.label !== 'unknown') {
          aggregatorRef.current.processRecognition(
            {
              sign: top.label,
              confidence: top.probability,
              top3: top3Candidates,
              durationMs,
              rawFrames,
            },
            endReason
          );

          setEntries(aggregatorRef.current.getEntries());
          setDiagnosticLogs(aggregatorRef.current.getPipelineLog());
        }
      } else {
        // decision.kind === 'not-sure'
        aggregatorRef.current.logUncertainRecognition({
          top3: top3Candidates,
          durationMs,
          rawFrames,
          endReason,
          partial: result.partial,
          reason: result.decision.reason,
        });
        setDiagnosticLogs(aggregatorRef.current.getPipelineLog());
      }
    });

    const unsubDiscard = session.onDiscard((telemetry) => {
      aggregatorRef.current.logDiscardedSegment(telemetry);
      setDiagnosticLogs(aggregatorRef.current.getPipelineLog());
    });

    return () => {
      unsubResult();
      unsubDiscard();
    };
  }, [session, model]);

  const handleAddManual = (signToAdd?: string) => {
    const text = signToAdd ?? manualInput;
    if (!text.trim()) return;
    aggregatorRef.current.addManualGloss(text);
    setEntries(aggregatorRef.current.getEntries());
    setManualInput('');
  };

  const handleStartEdit = (id: number, currentSign: string) => {
    setEditingId(id);
    setEditingText(currentSign);
  };

  const handleSaveEdit = (id: number) => {
    if (editingText.trim()) {
      aggregatorRef.current.replaceGloss(id, editingText.trim());
      setEntries(aggregatorRef.current.getEntries());
    }
    setEditingId(null);
    setEditingText('');
  };

  const handleDeleteGloss = (id: number) => {
    aggregatorRef.current.deleteGloss(id);
    setEntries(aggregatorRef.current.getEntries());
  };

  const handleClearBuffer = () => {
    aggregatorRef.current.clear();
    setEntries([]);
    setTranslation('');
  };

  const handleTranslate = async () => {
    const glosses = aggregatorRef.current.getConfirmedGlosses();
    if (glosses.length === 0) return;

    setIsTranslating(true);
    const res = await translatorRef.current.translate(glosses, true);
    setIsTranslating(false);

    if (res.available && res.englishTranslation) {
      setTranslation(res.englishTranslation);
    } else if (res.error) {
      setTranslation(`[Accumulator Mode: ${glosses.join(' ')}] (${res.error})`);
    }
  };

  const handleSpeak = () => {
    if (!translation) return;
    setIsSpeaking(true);
    translatorRef.current.speakEnglish(translation);
    setTimeout(() => setIsSpeaking(false), 2000);
  };

  const handleCopy = () => {
    const text = translation || aggregatorRef.current.getConfirmedGlosses().join(' ');
    if (!text) return;
    void navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleCopyDiagnostics = () => {
    const jsonStr = aggregatorRef.current.exportPipelineLog();
    void navigator.clipboard.writeText(jsonStr);
    setDiagCopied(true);
    setTimeout(() => setDiagCopied(false), 2000);
  };

  const handleDownloadDiagnostics = () => {
    const jsonStr = aggregatorRef.current.exportPipelineLog();
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `continuous-asl-diagnostics-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleClearDiagnostics = () => {
    aggregatorRef.current.clearPipelineLog();
    setDiagnosticLogs([]);
  };

  const confirmedGlosses = aggregatorRef.current.getConfirmedGlosses();
  const totalEvents = diagnosticLogs.length;
  const acceptedCount = diagnosticLogs.filter((e) => e.status === 'accepted').length;
  const debouncedCount = diagnosticLogs.filter((e) => e.status === 'debounced').length;
  const lowConfCount = diagnosticLogs.filter((e) => e.status === 'low_confidence').length;
  const discardedCount = diagnosticLogs.filter((e) => e.status === 'discarded').length;

  return (
    <div className="space-y-6 pb-16">
      {/* Header Info */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent mb-1">
            <Sparkles className="size-3.5" />
            Experimental Feature — Continuous ASL Pipeline
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            Continuous ASL Translation
          </h1>
          <p className="text-sm text-ink-muted">
            Signs are accumulated chronologically into an interactive gloss buffer with spatial neutral-zone reset and translated to English.
          </p>
        </div>

        {/* Server Status Pill */}
        <div className="flex items-center gap-2">
          {serverOnline ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-success/30 bg-success-soft px-3 py-1 text-xs font-medium text-success shadow-xs">
              <CheckCircle2 className="size-3.5" />
              Glossa-BART Online (Port 5005)
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-600 shadow-xs">
              <AlertCircle className="size-3.5" />
              Gloss Accumulator Mode (Offline)
            </span>
          )}
        </div>
      </div>

      {/* Main Grid: Camera Stage on Left, Gloss Buffer & Translation on Right */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Column: Camera Stage */}
        <div className="lg:col-span-7 space-y-4">
          <Stage session={session} snap={snap} />

          {/* Real-time Instructions Card */}
          <Card className="p-4 bg-surface-muted/50 border-line">
            <div className="flex items-start gap-3">
              <Info className="size-5 text-accent shrink-0 mt-0.5" />
              <div className="text-xs text-ink-muted space-y-1">
                <p className="font-semibold text-ink">How Continuous Signing Works:</p>
                <p>1. Sign an ASL sign to the camera. It is recognized and added to the gloss buffer below.</p>
                <p>2. To sign identical consecutive signs, lower your hands briefly below shoulder level (spatial neutral reset).</p>
                <p>3. You can manually add vocabulary (e.g. FRIEND, WANT, LEARN) that the 250-sign vision model cannot classify.</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Interactive Gloss Buffer & English Output */}
        <div className="lg:col-span-5 space-y-4">
          {/* Card 1: Interactive Gloss Buffer */}
          <Card className="p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-line pb-3">
              <div className="flex items-center gap-2">
                <Zap className="size-4 text-accent" />
                <h2 className="text-base font-bold text-ink">ASL Gloss Buffer</h2>
                <span className="rounded-full bg-surface-muted px-2 py-0.5 text-xs font-medium text-ink-muted">
                  {confirmedGlosses.length} tokens
                </span>
              </div>
              {confirmedGlosses.length > 0 && (
                <button
                  onClick={handleClearBuffer}
                  className="flex items-center gap-1 text-xs text-danger hover:underline font-medium"
                >
                  <Trash2 className="size-3" />
                  Clear
                </button>
              )}
            </div>

            {/* Gloss Pills Stream */}
            <div className="min-h-24 max-h-48 overflow-y-auto rounded-xl border border-line bg-paper/60 p-3">
              {entries.length === 0 ? (
                <div className="flex h-20 items-center justify-center text-xs text-ink-muted italic">
                  Sign to camera or type below to build an ASL gloss sequence…
                </div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {entries.map((e) => (
                    <div
                      key={e.id}
                      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-semibold shadow-xs transition ${
                        e.origin === 'manual'
                          ? 'border-purple-300 bg-purple-50 text-purple-900'
                          : e.origin === 'edited'
                            ? 'border-amber-300 bg-amber-50 text-amber-900'
                            : 'border-accent/30 bg-accent-soft text-accent'
                      }`}
                    >
                      {editingId === e.id ? (
                        <form
                          onSubmit={(ev) => {
                            ev.preventDefault();
                            handleSaveEdit(e.id);
                          }}
                          className="inline-flex items-center gap-1"
                        >
                          <input
                            type="text"
                            autoFocus
                            value={editingText}
                            onChange={(ev) => setEditingText(ev.target.value)}
                            onBlur={() => handleSaveEdit(e.id)}
                            className="w-20 rounded border border-accent bg-paper px-1 py-0.5 text-xs font-semibold uppercase text-ink outline-hidden"
                          />
                        </form>
                      ) : (
                        <button
                          type="button"
                          onClick={() => handleStartEdit(e.id, e.sign)}
                          className="hover:underline cursor-pointer"
                          title="Click to edit sign"
                        >
                          {e.sign}
                        </button>
                      )}
                      <span className="text-[10px] opacity-70">
                        {e.origin === 'manual'
                          ? '[Manual]'
                          : e.origin === 'edited'
                            ? '[Edited]'
                            : `${Math.round(e.confidence * 100)}%`}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleDeleteGloss(e.id)}
                        className="opacity-50 hover:opacity-100"
                        title="Remove sign"
                      >
                        <X className="size-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Manual Insert Input */}
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Type any ASL gloss (e.g. FRIEND, WANT)…"
                value={manualInput}
                onChange={(e) => setManualInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAddManual()}
                className="flex-1 rounded-xl border border-line bg-paper px-3 py-1.5 text-xs text-ink focus:border-accent focus:outline-hidden"
              />
              <button
                onClick={() => handleAddManual()}
                className="flex items-center gap-1 rounded-xl bg-accent px-3 py-1.5 text-xs font-medium text-accent-ink hover:opacity-90"
              >
                <Plus className="size-3.5" />
                Add
              </button>
            </div>

            {/* Quick Suggestions for Vocabulary Outside GestureSign */}
            <div className="flex flex-wrap items-center gap-1.5 pt-1 text-xs">
              <span className="text-ink-muted text-[11px]">Quick add OOV:</span>
              {['FRIEND', 'WANT', 'LEARN', 'TEACHER', 'NAME'].map((w) => (
                <button
                  key={w}
                  onClick={() => handleAddManual(w)}
                  className="rounded-md border border-line bg-surface px-2 py-0.5 text-[11px] font-medium text-ink-muted hover:bg-surface-muted hover:text-ink transition"
                >
                  +{w}
                </button>
              ))}
            </div>
          </Card>

          {/* Card 2: English Translation Output */}
          <Card className="p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-line pb-3">
              <div className="flex flex-col gap-0.5 sm:flex-row sm:items-center sm:gap-2">
                <div className="flex items-center gap-2">
                  <Languages className="size-4 text-accent" />
                  <h2 className="text-base font-bold text-ink">English Translation</h2>
                </div>
                <span className="text-[11px] font-medium text-ink-muted">
                  (AI-Generated Interpretation)
                </span>
              </div>
              <button
                onClick={handleTranslate}
                disabled={isTranslating || confirmedGlosses.length === 0}
                className="flex items-center gap-1.5 rounded-xl bg-accent px-3.5 py-1.5 text-xs font-bold text-accent-ink shadow-xs transition hover:opacity-90 disabled:opacity-50"
              >
                {isTranslating ? (
                  <>
                    <RefreshCw className="size-3.5 animate-spin" />
                    Translating…
                  </>
                ) : (
                  <>
                    <Languages className="size-3.5" />
                    Translate
                  </>
                )}
              </button>
            </div>

            {/* English Sentence Display */}
            <div className="rounded-xl border border-line bg-paper p-4 min-h-24 flex flex-col justify-between">
              {translation ? (
                <p className="text-base font-semibold leading-relaxed text-ink">
                  "{translation}"
                </p>
              ) : (
                <p className="text-xs text-ink-muted italic self-center my-auto">
                  Click 'Translate' to generate an English sentence from the gloss sequence.
                </p>
              )}

              {/* Action Buttons for Translation */}
              {translation && (
                <div className="flex items-center justify-end gap-2 pt-3 border-t border-line/50 mt-3">
                  <button
                    onClick={handleSpeak}
                    className="flex items-center gap-1 rounded-lg border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface-muted"
                    title="Speak with text-to-speech"
                  >
                    <Volume2 className={`size-3.5 ${isSpeaking ? 'text-accent animate-pulse' : ''}`} />
                    Speak
                  </button>
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1 rounded-lg border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface-muted"
                  >
                    <Copy className="size-3.5" />
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
              )}
            </div>

            {/* Transparent Translation Disclaimer */}
            <div className="rounded-lg bg-surface-muted/60 p-2.5 text-[11px] text-ink-muted leading-normal">
              <strong>Model Notice:</strong> Glossa-BART provides an AI-generated statistical English interpretation of the accumulated ASL glosses. It is an experimental sequence-to-sequence neural model and not an authoritative, certified human translator. Translations may hallucinate on compounding or inverted grammar. Always inspect the underlying gloss sequence above.
            </div>
          </Card>
        </div>
      </div>

      {/* Diagnostic Inspector Card (Bounded 50-Event Log) */}
      <Card className="p-5 space-y-4 border-line">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-line pb-3">
          <div className="flex items-center gap-2">
            <Activity className="size-4 text-accent" />
            <h2 className="text-base font-bold text-ink">
              Live Recognition & Segmentation Diagnostics
            </h2>
            <span className="rounded-full bg-surface-muted px-2 py-0.5 text-xs font-medium text-ink-muted">
              {totalEvents} / 50 events
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {totalEvents > 0 && (
              <>
                <button
                  type="button"
                  onClick={handleCopyDiagnostics}
                  className="flex items-center gap-1 rounded-lg border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface-muted transition"
                  title="Copy JSON diagnostic log to clipboard"
                >
                  <Copy className="size-3" />
                  {diagCopied ? 'Copied JSON!' : 'Copy Log'}
                </button>
                <button
                  type="button"
                  onClick={handleDownloadDiagnostics}
                  className="flex items-center gap-1 rounded-lg border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface-muted transition"
                  title="Download JSON diagnostic file"
                >
                  <Download className="size-3" />
                  Download JSON
                </button>
                <button
                  type="button"
                  onClick={handleClearDiagnostics}
                  className="flex items-center gap-1 text-xs text-danger hover:underline font-medium px-2 py-1"
                >
                  <Trash2 className="size-3" />
                  Clear
                </button>
              </>
            )}
            <button
              type="button"
              onClick={() => setShowDiagnostics((prev) => !prev)}
              className="flex items-center gap-1 rounded-lg border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface-muted transition ml-1"
            >
              {showDiagnostics ? (
                <>
                  <ChevronUp className="size-3.5" /> Hide Table
                </>
              ) : (
                <>
                  <ChevronDown className="size-3.5" /> Show Table
                </>
              )}
            </button>
          </div>
        </div>

        {/* Telemetry Summary Badges */}
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="inline-flex items-center gap-1 rounded-md bg-success-soft px-2.5 py-1 font-medium text-success">
            Accepted: {acceptedCount}
          </span>
          <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2.5 py-1 font-medium text-amber-600">
            Debounced: {debouncedCount}
          </span>
          <span className="inline-flex items-center gap-1 rounded-md bg-surface-muted px-2.5 py-1 font-medium text-ink-muted">
            Low Confidence: {lowConfCount}
          </span>
          <span className="inline-flex items-center gap-1 rounded-md bg-danger-soft px-2.5 py-1 font-medium text-danger">
            Discarded (Noise/Short): {discardedCount}
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-line bg-paper px-2.5 py-1 font-mono text-[11px] text-ink-muted">
            Neutral State: {aggregatorRef.current.getNeutralStatus() ? 'RESET (lap/idle)' : 'ELEVATED (signing)'}
          </span>
        </div>

        {/* Diagnostic Table */}
        {showDiagnostics && (
          <div className="overflow-x-auto rounded-xl border border-line bg-paper">
            {diagnosticLogs.length === 0 ? (
              <div className="p-6 text-center text-xs text-ink-muted italic">
                No segmentation or recognition events recorded yet. Start signing to inspect events.
              </div>
            ) : (
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-line bg-surface-muted/60 font-semibold text-ink">
                    <th className="p-2.5">Time</th>
                    <th className="p-2.5">Frames</th>
                    <th className="p-2.5">Duration</th>
                    <th className="p-2.5">End Reason</th>
                    <th className="p-2.5">Top Prediction</th>
                    <th className="p-2.5">Prob</th>
                    <th className="p-2.5">Status</th>
                    <th className="p-2.5">Rejection / Debounce Reason</th>
                    <th className="p-2.5">Neutral Reset</th>
                    <th className="p-2.5">Top-3 Candidates</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/70 font-mono text-[11px]">
                  {[...diagnosticLogs].reverse().map((ev) => (
                    <tr key={ev.id} className="hover:bg-surface-muted/30 transition">
                      <td className="p-2.5 text-ink-muted whitespace-nowrap">{ev.timeString}</td>
                      <td className="p-2.5 text-ink">{ev.rawFrames}</td>
                      <td className="p-2.5 text-ink-muted">{Math.round(ev.durationMs)}ms</td>
                      <td className="p-2.5">
                        <span className="rounded bg-surface-muted px-1.5 py-0.5 text-[10px] text-ink font-medium">
                          {ev.endReason}
                        </span>
                      </td>
                      <td className="p-2.5 font-bold text-ink">{ev.predictedSign}</td>
                      <td className="p-2.5 text-ink">
                        {ev.confidence > 0 ? `${(ev.confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="p-2.5">
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                            ev.status === 'accepted'
                              ? 'bg-success-soft text-success'
                              : ev.status === 'debounced'
                                ? 'bg-amber-500/10 text-amber-600'
                                : ev.status === 'low_confidence'
                                  ? 'bg-purple-100 text-purple-700'
                                  : 'bg-danger-soft text-danger'
                          }`}
                        >
                          {ev.status}
                        </span>
                      </td>
                      <td className="p-2.5 text-ink-muted max-w-xs truncate" title={ev.rejectionReason ?? 'none'}>
                        {ev.rejectionReason ?? '—'}
                      </td>
                      <td className="p-2.5 text-center">
                        {ev.neutralResetState ? (
                          <span className="text-success font-bold" title="Neutral reset was ACTIVE">✓</span>
                        ) : (
                          <span className="text-amber-500 font-bold" title="Hands in signing space">✗</span>
                        )}
                      </td>
                      <td className="p-2.5 text-ink-muted whitespace-nowrap">
                        {ev.top3 && ev.top3.length > 0
                          ? ev.top3
                              .slice(0, 3)
                              .map((c) => `${c.label} (${(c.probability * 100).toFixed(0)}%)`)
                              .join(', ')
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}

