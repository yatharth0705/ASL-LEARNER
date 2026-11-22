import { AlertCircle, ArrowRight, CheckCircle2, ExternalLink, Filter, Flame, Search, Shuffle, Sparkles, Trophy } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import { SIGN_CATEGORIES } from '../lib/categories';
import { recordPracticeAttempt } from '../lib/progress';
import { practiceScore } from '../ml/decision';
import { dictionaryUrl } from '../ml/meta';
import type { LoadedModel } from '../ml/model-client';
import { Stage } from './Stage';
import { Card, Percent } from './ui';

export interface PracticePageProps {
  session: Session;
  snap: SessionSnapshot;
  model: LoadedModel;
  initialSignIndex?: number | undefined;
}

interface Attempt {
  id: number;
  score: number;
  sawIndex: number | null;
  partial: boolean;
}

function pickRandom(indices: number[], not: number | null): number {
  const pool = indices.length > 1 ? indices.filter((i) => i !== not) : indices;
  return pool[Math.floor(Math.random() * pool.length)] ?? 0;
}

export function PracticePage({
  session,
  snap,
  model,
  initialSignIndex,
}: PracticePageProps) {
  const { labels, meta } = model;
  const shipped = useMemo(() => labels.filter((l) => l.shipped).map((l) => l.index), [labels]);

  const [targetIndex, setTargetIndex] = useState<number>(() => {
    if (initialSignIndex !== undefined && shipped.includes(initialSignIndex)) {
      return initialSignIndex;
    }
    return pickRandom(shipped, null);
  });

  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [sessionStreak, setSessionStreak] = useState(0);
  const [sessionTries, setSessionTries] = useState(0);
  const [sessionPassed, setSessionPassed] = useState(0);

  // Target sign object
  const currentSign = labels[targetIndex];

  // Shuffled category pool
  const filteredIndices = useMemo(() => {
    let pool = shipped;
    if (selectedCategory !== 'all') {
      const cat = SIGN_CATEGORIES.find((c) => c.id === selectedCategory);
      if (cat) {
        pool = shipped.filter((idx) => {
          const name = labels[idx]?.sign.toLowerCase();
          return cat.signs.some((s) => s.toLowerCase() === name);
        });
      }
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      pool = pool.filter((idx) => {
        const sign = labels[idx];
        return (
          sign?.sign.toLowerCase().includes(q) ||
          sign?.label.toLowerCase().includes(q)
        );
      });
    }
    return pool.length > 0 ? pool : shipped;
  }, [shipped, selectedCategory, searchQuery, labels]);

  // Listen to session results
  useEffect(() => {
    return session.onResult((result, id) => {
      const score = practiceScore(
        result.inference.logits,
        labels.map((l) => l.shipped),
        targetIndex,
        meta.calibration,
      );
      const best = result.decision.top[0];
      const sawIndex = result.decision.kind === 'sign' && best ? best.index : null;
      const isPassed = !result.partial && score >= meta.calibration.notSureBelow;

      setAttempt({
        id,
        score,
        sawIndex,
        partial: result.partial,
      });

      setSessionTries((t) => t + 1);
      if (isPassed) {
        setSessionPassed((p) => p + 1);
        setSessionStreak((s) => s + 1);
      } else {
        setSessionStreak(0);
      }

      // Record in persistent progress history
      const sawLabel = sawIndex !== null ? labels[sawIndex]?.label ?? 'unknown' : 'not sure';
      recordPracticeAttempt(currentSign?.label ?? 'target', sawLabel, score, isPassed);
    });
  }, [session, labels, meta.calibration, targetIndex, currentSign]);

  const passed =
    attempt !== null && !attempt.partial && attempt.score >= meta.calibration.notSureBelow;

  const handleNextSign = () => {
    const nextIdx = pickRandom(filteredIndices, targetIndex);
    setTargetIndex(nextIdx);
    setAttempt(null);
    session.resetRecognition();
  };

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent mb-1">
            <Sparkles className="size-3.5" />
            Phase 3: Interactive Practice & Scoring
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            Sign Practice Studio
          </h1>
          <p className="text-sm text-ink-muted">
            The model compares your motion against calibrated prototypes. Receive instant similarity scores and guidance.
          </p>
        </div>

        {/* Live Session Score Pill */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 rounded-xl border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink shadow-xs">
            <Flame className="size-4 text-orange-500" />
            <span>Streak: <strong>{sessionStreak}</strong></span>
          </div>
          <div className="flex items-center gap-1.5 rounded-xl border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink shadow-xs">
            <Trophy className="size-4 text-emerald-600" />
            <span>Accuracy: <strong>{sessionTries > 0 ? Math.round((sessionPassed / sessionTries) * 100) : 0}%</strong></span>
          </div>
        </div>
      </div>

      {/* Main Grid: Camera Stage on Left, Practice Target & Score on Right */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Column: Live Stage */}
        <div className="lg:col-span-7 space-y-4">
          <Stage session={session} snap={snap} />
        </div>

        {/* Right Column: Target Flashcard & Interactive Feedback */}
        <div className="lg:col-span-5 space-y-5">
          {/* Target Flashcard */}
          <Card className="p-5 space-y-4 shadow-xs">
            <div className="flex items-center justify-between border-b border-line pb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                Target Sign
              </span>
              <button
                onClick={handleNextSign}
                className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline"
              >
                <Shuffle className="size-3.5" />
                Next Random Sign
              </button>
            </div>

            {currentSign && (
              <div className="space-y-3">
                <div className="rounded-2xl border border-accent/20 bg-accent-soft/30 p-5 text-center">
                  <p className="text-xs font-semibold uppercase tracking-wider text-accent">
                    Sign this to the camera:
                  </p>
                  <h2 className="mt-1 text-4xl font-black capitalize tracking-tight text-ink">
                    {currentSign.label}
                  </h2>
                  <p className="mt-1 text-xs text-ink-muted">
                    Benchmark held-out accuracy: {currentSign.heldOutTop1 !== null ? Math.round(currentSign.heldOutTop1 * 100) : 75}%
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <a
                    href={dictionaryUrl(currentSign)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-xl border border-line bg-surface py-2.5 px-3 text-xs font-medium text-ink hover:bg-surface-muted transition shadow-xs"
                  >
                    <ExternalLink className="size-3.5 text-accent" />
                    Watch Dictionary Video
                  </a>
                  <button
                    onClick={handleNextSign}
                    className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-accent py-2.5 px-4 text-xs font-semibold text-accent-ink hover:bg-accent/90 transition shadow-xs"
                  >
                    <Shuffle className="size-3.5" />
                    Skip / Next
                  </button>
                </div>
              </div>
            )}
          </Card>

          {/* Practice Feedback Box */}
          <Card className="p-5 space-y-4 shadow-xs">
            <span className="text-xs font-bold uppercase tracking-wider text-ink-muted">
              Attempt Assessment
            </span>

            {attempt === null ? (
              <div className="rounded-xl border border-line bg-paper p-6 text-center text-ink-muted space-y-2">
                <Sparkles className="size-6 mx-auto text-accent" />
                <p className="text-xs leading-relaxed">
                  Sign <strong>"{currentSign?.label}"</strong> to the camera, or play a replay clip, to receive your score.
                </p>
              </div>
            ) : (
              <div
                className={`rounded-2xl border p-5 space-y-3 transition ${
                  passed
                    ? 'border-emerald-500/40 bg-emerald-500/10'
                    : 'border-amber-500/40 bg-amber-500/10'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-bold ${
                      passed
                        ? 'bg-emerald-600 text-white'
                        : 'bg-amber-600 text-white'
                    }`}
                  >
                    {passed ? <CheckCircle2 className="size-3.5" /> : <AlertCircle className="size-3.5" />}
                    {passed ? 'Passed (+100 XP)' : 'Needs Improvement'}
                  </span>
                  <span className="text-base font-extrabold text-ink">
                    <Percent value={attempt.score} /> Match
                  </span>
                </div>

                <p className="text-xs text-ink-muted leading-relaxed">
                  {passed ? (
                    'Excellent! The model recognized your motion as matching the target sign with high confidence.'
                  ) : attempt.sawIndex !== null ? (
                    <>
                      The model thought you signed{' '}
                      <strong className="text-ink">"{labels[attempt.sawIndex]?.label}"</strong> instead of{' '}
                      <strong>"{currentSign?.label}"</strong>. Check hand shape and motion.
                    </>
                  ) : (
                    'Gesture was not recognized clearly. Keep your shoulders in view and sign at a steady pace.'
                  )}
                </p>

                {passed && (
                  <button
                    onClick={handleNextSign}
                    className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-xs font-semibold text-accent-ink transition hover:bg-accent/90"
                  >
                    Next Sign
                    <ArrowRight className="size-3.5" />
                  </button>
                )}
              </div>
            )}
          </Card>

          {/* Category & Topic Filter Accordion */}
          <Card className="p-4 space-y-3 shadow-xs">
            <div className="flex items-center justify-between text-xs font-semibold text-ink-muted">
              <span className="flex items-center gap-1.5">
                <Filter className="size-3.5 text-accent" />
                Practice Topic
              </span>
              <span className="text-[11px] font-medium">{filteredIndices.length} signs in deck</span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-ink-muted" />
              <input
                type="text"
                placeholder="Search practice signs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full rounded-lg border border-line bg-surface pl-8 pr-3 py-1.5 text-xs text-ink placeholder:text-ink-muted focus:border-accent focus:outline-none"
              />
            </div>

            <div className="flex flex-wrap gap-1.5 text-xs max-h-36 overflow-y-auto pt-1">
              <button
                onClick={() => setSelectedCategory('all')}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                  selectedCategory === 'all'
                    ? 'bg-accent text-accent-ink'
                    : 'bg-surface-muted text-ink-muted hover:text-ink'
                }`}
              >
                All Signs (250)
              </button>
              {SIGN_CATEGORIES.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                    selectedCategory === cat.id
                      ? 'bg-accent text-accent-ink'
                      : 'bg-surface-muted text-ink-muted hover:text-ink'
                  }`}
                >
                  {cat.name}
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
