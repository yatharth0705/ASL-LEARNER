import { Cpu, Lock } from 'lucide-react';
import { useState, useSyncExternalStore } from 'react';
import type { Session } from './app/session';
import { AboutPage } from './components/AboutPage';
import { AlphabetPage } from './components/AlphabetPage';
import { ContinuousTranslationPage } from './components/ContinuousTranslationPage';
import { DashboardPage } from './components/DashboardPage';
import { LandingPage } from './components/LandingPage';
import { ModelGate } from './components/ModelGate';
import { PracticePage } from './components/PracticePage';
import { ProgressPage } from './components/ProgressPage';
import { RecognitionPage } from './components/RecognitionPage';
import { SentenceBuilderPage } from './components/SentenceBuilderPage';
import { SiteHeader, type AppPage } from './components/SiteHeader';
import { DATASET_CREDIT, REPO_URL } from './links';

export function App({ session }: { session: Session }) {
  const snap = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const [page, setPage] = useState<AppPage>('landing');
  const [selectedPracticeSignIndex, setSelectedPracticeSignIndex] = useState<number | undefined>(undefined);

  const model = snap.model;
  const isModelReady = model.status === 'ready';
  const shippedLabels = isModelReady ? model.model.labels.filter((l) => l.shipped) : [];
  const shippedCount = shippedLabels.length || 250;

  const handleNavigate = (nextPage: AppPage) => {
    setPage(nextPage);
    session.resetRecognition();
    const cameraPages: AppPage[] = ['recognition', 'continuous', 'practice', 'sentence'];
    if (!cameraPages.includes(nextPage)) {
      session.stopCamera();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleSelectPracticeSign = (signIndex: number) => {
    setSelectedPracticeSignIndex(signIndex);
    handleNavigate('practice');
  };

  return (
    <div className="min-h-screen flex flex-col bg-paper text-ink font-sans selection:bg-accent selection:text-accent-ink">
      {/* Global Navigation Header */}
      <SiteHeader currentPage={page} onNavigate={handleNavigate} snap={snap} />

      {/* Main Content Container */}
      <main id="main" className="flex-1 mx-auto w-full max-w-7xl px-4 pt-6 pb-12 sm:px-6">
        {page === 'landing' && (
          <LandingPage onNavigate={handleNavigate} signsCount={shippedCount} />
        )}

        {page === 'dashboard' && (
          <DashboardPage
            onNavigate={handleNavigate}
            onSelectPracticeSign={handleSelectPracticeSign}
            labels={isModelReady ? model.model.labels : []}
          />
        )}

        {page === 'recognition' && (
          <ModelGate state={model} onRetry={() => void session.loadModel()}>
            {(loadedModel) => (
              <RecognitionPage session={session} snap={snap} model={loadedModel} />
            )}
          </ModelGate>
        )}

        {page === 'continuous' && (
          <ModelGate state={model} onRetry={() => void session.loadModel()}>
            {(loadedModel) => (
              <ContinuousTranslationPage session={session} snap={snap} model={loadedModel} />
            )}
          </ModelGate>
        )}

        {page === 'practice' && (
          <ModelGate state={model} onRetry={() => void session.loadModel()}>
            {(loadedModel) => (
              <PracticePage
                session={session}
                snap={snap}
                model={loadedModel}
                initialSignIndex={selectedPracticeSignIndex}
              />
            )}
          </ModelGate>
        )}

        {page === 'sentence' && (
          <ModelGate state={model} onRetry={() => void session.loadModel()}>
            {(loadedModel) => (
              <SentenceBuilderPage session={session} snap={snap} model={loadedModel} />
            )}
          </ModelGate>
        )}

        {page === 'alphabet' && <AlphabetPage onNavigate={handleNavigate} />}

        {page === 'progress' && (
          <ProgressPage
            onNavigate={handleNavigate}
            labels={isModelReady ? model.model.labels : []}
          />
        )}

        {page === 'about' && <AboutPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-line bg-surface py-8 text-xs text-ink-muted">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div className="space-y-1">
            <p className="font-semibold text-ink">
              ASL-Learner • Pretrained Deep Learning Sign Language Education
            </p>
            <p>
              Pretrained model from{' '}
              <a
                href={REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent underline underline-offset-2 hover:opacity-80"
              >
                GestureSign
              </a>{' '}
              by Shashank Yadav (MIT License). Benchmarked on Kaggle Google ASL Signs dataset ({DATASET_CREDIT}).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <span className="inline-flex items-center gap-1.5 text-emerald-600 font-medium">
              <Lock className="size-3.5" />
              100% On-Device Privacy
            </span>
            <span className="inline-flex items-center gap-1.5 text-ink-muted">
              <Cpu className="size-3.5" />
              WASM SIMD Accelerated
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
