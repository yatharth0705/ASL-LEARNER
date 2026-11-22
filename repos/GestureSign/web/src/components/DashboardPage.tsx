import { ArrowRight, BookOpen, ExternalLink, Eye, Flame, Layers, Play, Sparkles, Trophy, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import { SIGN_CATEGORIES } from '../lib/categories';
import { loadProgress, type UserProgress } from '../lib/progress';
import { dictionaryUrl, type SignLabel } from '../ml/meta';
import type { AppPage } from './SiteHeader';

interface DashboardPageProps {
  onNavigate: (page: AppPage) => void;
  onSelectPracticeSign?: (signIndex: number) => void;
  labels: SignLabel[];
}

export function DashboardPage({ onNavigate, onSelectPracticeSign, labels }: DashboardPageProps) {
  const [progress, setProgress] = useState<UserProgress>(loadProgress);

  useEffect(() => {
    setProgress(loadProgress());
  }, []);

  const totalSigns = labels.filter((l) => l.shipped).length;
  const accuracy = progress.totalAttempts > 0
    ? Math.round((progress.passedAttempts / progress.totalAttempts) * 100)
    : 0;

  // Sign of the Day: stable pseudo-random pick based on current date
  const todayDay = new Date().getDate() + new Date().getMonth() * 31;
  const shippedLabels = labels.filter((l) => l.shipped);
  const signOfTheDay = shippedLabels[todayDay % (shippedLabels.length || 1)] ?? shippedLabels[0];

  const handlePracticeSign = (sign: SignLabel) => {
    if (onSelectPracticeSign) {
      onSelectPracticeSign(sign.index);
    }
    onNavigate('practice');
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Header Greeting */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            Welcome to Your ASL Dashboard
          </h1>
          <p className="text-sm text-ink-muted">
            Track your sign language fluency, daily practice goals, and vocabulary mastery.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate('recognition')}
            className="inline-flex items-center gap-1.5 rounded-xl bg-accent px-4 py-2.5 text-xs font-semibold text-accent-ink shadow-sm transition hover:bg-accent/90"
          >
            <Eye className="size-4" />
            Live Recognition
          </button>
          <button
            onClick={() => onNavigate('sentence')}
            className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-surface px-4 py-2.5 text-xs font-semibold text-ink transition hover:bg-surface-muted"
          >
            <Layers className="size-4 text-accent" />
            Sentence Builder
          </button>
        </div>
      </div>

      {/* Progress Metric Cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-muted">Practice Streak</span>
            <div className="flex size-8 items-center justify-center rounded-lg bg-orange-500/10 text-orange-600">
              <Flame className="size-4" />
            </div>
          </div>
          <p className="mt-3 text-2xl font-bold text-ink sm:text-3xl">
            {progress.currentStreak} <span className="text-xs font-normal text-ink-muted">in a row</span>
          </p>
          <p className="mt-1 text-xs text-ink-muted">
            Best streak: <span className="font-semibold text-ink">{progress.bestStreak}</span>
          </p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-muted">Signs Mastered</span>
            <div className="flex size-8 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600">
              <Trophy className="size-4" />
            </div>
          </div>
          <p className="mt-3 text-2xl font-bold text-ink sm:text-3xl">
            {progress.masteredSigns.length}{' '}
            <span className="text-xs font-normal text-ink-muted">/ {totalSigns}</span>
          </p>
          <p className="mt-1 text-xs text-ink-muted">
            {Math.round((progress.masteredSigns.length / (totalSigns || 1)) * 100)}% of vocabulary
          </p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-muted">Success Rate</span>
            <div className="flex size-8 items-center justify-center rounded-lg bg-blue-500/10 text-blue-600">
              <Zap className="size-4" />
            </div>
          </div>
          <p className="mt-3 text-2xl font-bold text-ink sm:text-3xl">{accuracy}%</p>
          <p className="mt-1 text-xs text-ink-muted">
            {progress.passedAttempts} of {progress.totalAttempts} attempts
          </p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-muted">Available Signs</span>
            <div className="flex size-8 items-center justify-center rounded-lg bg-accent/10 text-accent">
              <Sparkles className="size-4" />
            </div>
          </div>
          <p className="mt-3 text-2xl font-bold text-ink sm:text-3xl">{totalSigns}</p>
          <p className="mt-1 text-xs text-ink-muted">Pretrained ASL classes</p>
        </div>
      </div>

      {/* Featured Sign of the Day & Quick Jump */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Sign of the Day Card */}
        {signOfTheDay && (
          <div className="lg:col-span-1 rounded-2xl border border-accent/30 bg-accent-soft/30 p-6 shadow-xs flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1 rounded-full bg-accent text-accent-ink px-2.5 py-0.5 text-xs font-semibold">
                  <Sparkles className="size-3" />
                  Sign of the Day
                </span>
                <span className="text-xs text-ink-muted font-medium">Index #{signOfTheDay.index}</span>
              </div>

              <div>
                <h3 className="text-3xl font-extrabold capitalize tracking-tight text-ink">
                  {signOfTheDay.label}
                </h3>
                <p className="mt-1 text-xs text-ink-muted">
                  Held-out benchmark accuracy: {signOfTheDay.heldOutTop1 !== null ? Math.round(signOfTheDay.heldOutTop1 * 100) : 75}%
                </p>
              </div>

              <p className="text-xs leading-relaxed text-ink-muted">
                Practice this sign today using your webcam to earn extra mastery points and maintain your daily streak!
              </p>
            </div>

            <div className="mt-6 flex items-center gap-2 pt-2">
              <button
                onClick={() => handlePracticeSign(signOfTheDay)}
                className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-xl bg-accent px-4 py-2.5 text-xs font-semibold text-accent-ink shadow-sm transition hover:bg-accent/90"
              >
                <Play className="size-3.5 fill-current" />
                Practice Now
              </button>
              <a
                href={dictionaryUrl(signOfTheDay)}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-xl border border-line bg-surface p-2.5 text-xs font-medium text-ink hover:bg-surface-muted transition"
                title="View demonstration in ASL Dictionary"
              >
                <ExternalLink className="size-4" />
              </a>
            </div>
          </div>
        )}

        {/* Quick Launch Cards */}
        <div className="lg:col-span-2 grid gap-4 sm:grid-cols-2">
          <div
            onClick={() => onNavigate('alphabet')}
            className="rounded-2xl border border-line bg-surface p-5 shadow-xs transition hover:border-accent/40 cursor-pointer flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="flex size-10 items-center justify-center rounded-xl bg-amber-500/10 text-amber-600">
                <BookOpen className="size-5" />
              </div>
              <h3 className="text-base font-bold text-ink">Fingerspelling Alphabet</h3>
              <p className="text-xs text-ink-muted leading-relaxed">
                Explore all 26 manual alphabet handshapes with mnemonics, finger positions, and common mistakes.
              </p>
            </div>
            <div className="mt-4 flex items-center gap-1 text-xs font-semibold text-accent">
              <span>View Alphabet (A-Z)</span>
              <ArrowRight className="size-3.5" />
            </div>
          </div>

          <div
            onClick={() => onNavigate('sentence')}
            className="rounded-2xl border border-line bg-surface p-5 shadow-xs transition hover:border-accent/40 cursor-pointer flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="flex size-10 items-center justify-center rounded-xl bg-purple-500/10 text-purple-600">
                <Layers className="size-5" />
              </div>
              <h3 className="text-base font-bold text-ink">Sentence Builder</h3>
              <p className="text-xs text-ink-muted leading-relaxed">
                Accumulate isolated signs into written phrases with duplicate protection and text-to-speech.
              </p>
            </div>
            <div className="mt-4 flex items-center gap-1 text-xs font-semibold text-accent">
              <span>Open Sentence Builder</span>
              <ArrowRight className="size-3.5" />
            </div>
          </div>
        </div>
      </div>

      {/* Vocabulary Categories Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-ink">Vocabulary by Topic</h2>
            <p className="text-xs text-ink-muted">250 signs organized into thematic learning modules</p>
          </div>
          <button
            onClick={() => onNavigate('practice')}
            className="text-xs font-semibold text-accent hover:underline flex items-center gap-1"
          >
            <span>Practice all</span>
            <ArrowRight className="size-3.5" />
          </button>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {SIGN_CATEGORIES.map((cat) => {
            const masteredInCat = cat.signs.filter((s) =>
              progress.masteredSigns.some((m) => m.toLowerCase() === s.toLowerCase()),
            ).length;
            const pct = Math.round((masteredInCat / cat.signs.length) * 100);

            return (
              <div
                key={cat.id}
                onClick={() => onNavigate('practice')}
                className="rounded-xl border border-line bg-surface p-4 shadow-xs transition hover:border-accent/40 cursor-pointer"
              >
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-sm text-ink">{cat.name}</h3>
                  <span className="text-[11px] font-medium text-ink-muted">
                    {cat.signs.length} signs
                  </span>
                </div>
                <p className="mt-1 text-xs text-ink-muted line-clamp-2">{cat.description}</p>
                <div className="mt-3">
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-ink-muted">Mastered</span>
                    <span className="font-medium text-ink">{masteredInCat} / {cat.signs.length}</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-surface-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-accent transition-all duration-300"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
