import { ArrowRight, CheckCircle2, History, Trash2, Trophy, XCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { SIGN_CATEGORIES } from '../lib/categories';
import { loadProgress, resetProgress, type UserProgress } from '../lib/progress';
import type { SignLabel } from '../ml/meta';
import type { AppPage } from './SiteHeader';
import { Percent } from './ui';

interface ProgressPageProps {
  onNavigate: (page: AppPage) => void;
  labels: SignLabel[];
}

export function ProgressPage({ onNavigate, labels }: ProgressPageProps) {
  const [progress, setProgress] = useState<UserProgress>(loadProgress);

  useEffect(() => {
    setProgress(loadProgress());
  }, []);

  const totalSigns = labels.filter((l) => l.shipped).length;
  const accuracy = progress.totalAttempts > 0
    ? Math.round((progress.passedAttempts / progress.totalAttempts) * 100)
    : 0;

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset all practice history and streaks?')) {
      setProgress(resetProgress());
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent mb-1">
            <History className="size-3.5" />
            Learning Progress & Analytics
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            Your ASL Learning Journey
          </h1>
          <p className="text-sm text-ink-muted">
            All practice history and mastery achievements are persisted locally in your browser.
          </p>
        </div>

        <button
          onClick={handleReset}
          className="inline-flex items-center gap-1.5 self-start rounded-xl border border-danger/30 bg-danger-soft px-3.5 py-2 text-xs font-semibold text-danger hover:bg-danger/20 transition"
        >
          <Trash2 className="size-3.5" />
          Reset History
        </button>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <span className="text-xs font-medium text-ink-muted">Total Practice Attempts</span>
          <p className="mt-2 text-3xl font-extrabold text-ink">{progress.totalAttempts}</p>
          <p className="mt-1 text-xs text-ink-muted">{progress.passedAttempts} successful evaluations</p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <span className="text-xs font-medium text-ink-muted">Accuracy Rate</span>
          <p className="mt-2 text-3xl font-extrabold text-ink">{accuracy}%</p>
          <p className="mt-1 text-xs text-ink-muted">Across all attempts</p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <span className="text-xs font-medium text-ink-muted">Current / Best Streak</span>
          <p className="mt-2 text-3xl font-extrabold text-ink">
            {progress.currentStreak} <span className="text-lg font-bold text-ink-muted">/ {progress.bestStreak}</span>
          </p>
          <p className="mt-1 text-xs text-ink-muted">Consecutive passes</p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-5 shadow-xs">
          <span className="text-xs font-medium text-ink-muted">Mastered Signs</span>
          <p className="mt-2 text-3xl font-extrabold text-ink">
            {progress.masteredSigns.length} <span className="text-lg font-bold text-ink-muted">/ {totalSigns}</span>
          </p>
          <p className="mt-1 text-xs text-ink-muted">Passed at least twice</p>
        </div>
      </div>

      {/* Category Progress Breakdown */}
      <div className="rounded-2xl border border-line bg-surface p-6 shadow-xs space-y-5">
        <div>
          <h2 className="text-lg font-bold text-ink">Category Mastery Breakdown</h2>
          <p className="text-xs text-ink-muted">Your proficiency progress across different ASL topics</p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          {SIGN_CATEGORIES.map((cat) => {
            const masteredCount = cat.signs.filter((s) =>
              progress.masteredSigns.some((m) => m.toLowerCase() === s.toLowerCase()),
            ).length;
            const pct = Math.round((masteredCount / cat.signs.length) * 100);

            return (
              <div key={cat.id} className="rounded-xl border border-line bg-paper p-4 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-ink">{cat.name}</span>
                  <span className="font-medium text-ink-muted">{masteredCount} of {cat.signs.length} mastered ({pct}%)</span>
                </div>
                <div className="h-2 w-full rounded-full bg-surface-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-accent transition-all duration-300"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Mastered Signs Pill Gallery */}
      <div className="rounded-2xl border border-line bg-surface p-6 shadow-xs space-y-4">
        <div>
          <h2 className="text-lg font-bold text-ink">Mastered Signs Portfolio</h2>
          <p className="text-xs text-ink-muted">Signs successfully practiced and retained in memory</p>
        </div>

        {progress.masteredSigns.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line p-8 text-center text-xs text-ink-muted space-y-2">
            <Trophy className="size-8 mx-auto text-ink-muted/50" />
            <p>No signs mastered yet. Pass a sign at least twice in Practice Mode to add it here!</p>
            <button
              onClick={() => onNavigate('practice')}
              className="mt-2 inline-flex items-center gap-1 font-semibold text-accent hover:underline"
            >
              Start Practice Session
              <ArrowRight className="size-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {progress.masteredSigns.map((sign) => (
              <span
                key={sign}
                className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-800 dark:text-emerald-300"
              >
                <CheckCircle2 className="size-3.5 text-emerald-600" />
                {sign}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Recent Practice Log Table */}
      <div className="rounded-2xl border border-line bg-surface p-6 shadow-xs space-y-4">
        <div>
          <h2 className="text-lg font-bold text-ink">Recent Practice Log</h2>
          <p className="text-xs text-ink-muted">Detailed log of recent webcam evaluation attempts</p>
        </div>

        {progress.history.length === 0 ? (
          <p className="text-xs text-ink-muted/70 italic py-4">No recent practice attempts recorded.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-line text-ink-muted">
                  <th className="py-2.5 font-semibold">Target Sign</th>
                  <th className="py-2.5 font-semibold">Detected Sign</th>
                  <th className="py-2.5 font-semibold">Similarity Score</th>
                  <th className="py-2.5 font-semibold">Outcome</th>
                  <th className="py-2.5 font-semibold">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {progress.history.map((record) => (
                  <tr key={record.id} className="hover:bg-surface-muted/50 transition">
                    <td className="py-2.5 font-semibold text-ink capitalize">{record.target}</td>
                    <td className="py-2.5 text-ink-muted capitalize">{record.sign}</td>
                    <td className="py-2.5 font-mono text-ink">
                      <Percent value={record.score} />
                    </td>
                    <td className="py-2.5">
                      <span
                        className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-semibold ${
                          record.passed
                            ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                            : 'bg-amber-500/10 text-amber-700 dark:text-amber-300'
                        }`}
                      >
                        {record.passed ? <CheckCircle2 className="size-3" /> : <XCircle className="size-3" />}
                        {record.passed ? 'Passed' : 'Missed'}
                      </span>
                    </td>
                    <td className="py-2.5 text-ink-muted">
                      {new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
