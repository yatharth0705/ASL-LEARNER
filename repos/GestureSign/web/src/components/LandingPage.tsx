import { Activity, ArrowRight, BookOpen, Cpu, Eye, History, Layers, Lock, ShieldCheck, Sparkles, Video } from 'lucide-react';
import type { AppPage } from './SiteHeader';

interface LandingPageProps {
  onNavigate: (page: AppPage) => void;
  signsCount: number;
}

export function LandingPage({ onNavigate, signsCount }: LandingPageProps) {
  return (
    <div className="space-y-16 pb-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl border border-line bg-surface p-8 sm:p-12 md:p-16 shadow-xs">
        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent-soft px-3.5 py-1 text-xs font-semibold text-accent">
            <Sparkles className="size-3.5" />
            <span>Pretrained ONNX Deep Learning Model • 250 Isolated Signs</span>
          </div>

          <h1 className="text-4xl font-extrabold tracking-tight sm:text-6xl text-ink">
            Learn American Sign Language with{' '}
            <span className="text-accent">Private On-Device AI</span>
          </h1>

          <p className="text-lg leading-relaxed text-ink-muted sm:text-xl">
            Real-time isolated sign recognition, fingerspelling guides, interactive scoring, and sentence building. 
            Runs 100% inside your browser tab powered by MediaPipe and a 1.4M-parameter Conv-Transformer model.
          </p>

          {/* Privacy & Hardware Highlights */}
          <div className="flex flex-wrap gap-3 pt-2">
            {[
              { icon: Lock, label: '100% Private (No video leaves laptop)' },
              { icon: Cpu, label: '4.4 ms CPU Latency (Intel i5 Optimized)' },
              { icon: ShieldCheck, label: 'Pretrained ONNX (No training needed)' },
            ].map(({ icon: Icon, label }) => (
              <span
                key={label}
                className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface-muted px-3 py-1.5 text-xs font-medium text-ink-muted"
              >
                <Icon className="size-3.5 text-accent" />
                {label}
              </span>
            ))}
          </div>

          {/* Call to Actions */}
          <div className="flex flex-wrap items-center gap-3 pt-4">
            <button
              onClick={() => onNavigate('recognition')}
              className="inline-flex items-center gap-2 rounded-xl bg-accent px-6 py-3.5 text-sm font-semibold text-accent-ink shadow-md transition hover:bg-accent/90"
            >
              <Eye className="size-4" />
              Start Live Recognition
              <ArrowRight className="size-4" />
            </button>
            <button
              onClick={() => onNavigate('practice')}
              className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface px-5 py-3.5 text-sm font-semibold text-ink transition hover:bg-surface-muted"
            >
              <Sparkles className="size-4 text-accent" />
              Practice Signs
            </button>
            <button
              onClick={() => onNavigate('alphabet')}
              className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface px-5 py-3.5 text-sm font-semibold text-ink transition hover:bg-surface-muted"
            >
              <BookOpen className="size-4 text-ink-muted" />
              ASL Alphabet
            </button>
          </div>
        </div>

        {/* Decorative background glow */}
        <div className="pointer-events-none absolute -right-20 -bottom-20 size-96 rounded-full bg-accent/10 blur-3xl" />
      </section>

      {/* Feature Grid */}
      <section className="space-y-6">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <h2 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            Complete Toolkit for ASL Learners
          </h2>
          <p className="text-sm text-ink-muted sm:text-base">
            Engineered around the verified GestureSign pretrained ONNX model with exact feature calibration.
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {[
            {
              id: 'recognition',
              icon: Eye,
              title: 'Real-Time Sign Recognition',
              description: `Recognizes ${signsCount} isolated ASL signs in real-time with top-3 guesses, calibrated confidence thresholds, and "Not sure" fallback when hands rest.`,
              badge: 'MediaPipe Holistic',
            },
            {
              id: 'practice',
              icon: Sparkles,
              title: 'Interactive Practice Mode',
              description: 'Select target signs, receive instant cosine-similarity scoring, view dictionary video demonstrations, and build your confidence score.',
              badge: 'Instant Feedback',
            },
            {
              id: 'sentence',
              icon: Layers,
              title: 'Isolated-Sign Sentence Builder',
              description: 'Sequence isolated sign predictions into text sentences with hold deduplication, pause controls, and built-in speech synthesis.',
              badge: 'Experimental Tool',
            },
            {
              id: 'alphabet',
              icon: BookOpen,
              title: 'Fingerspelling Alphabet (A-Z)',
              description: 'Visual reference guide covering all 26 ASL handshapes, palm orientations, formation rules, mnemonics, and common mistakes.',
              badge: '26 Letters',
            },
            {
              id: 'progress',
              icon: History,
              title: 'Progress & Mastery Tracking',
              description: 'Track daily practice streaks, success rates across categories (Everyday, Animals, Food, Actions), and review your complete practice history.',
              badge: 'Local Persistence',
            },
            {
              id: 'recognition',
              icon: Video,
              title: 'Offline Replay Mode',
              description: 'No webcam required! Test the pipeline using 6 genuine held-out signer landmark clips from the Google ASL Signs benchmark.',
              badge: 'Camera-Free',
            },
          ].map(({ id, icon: Icon, title, description, badge }) => (
            <div
              key={title}
              onClick={() => onNavigate(id as AppPage)}
              className="group flex flex-col justify-between rounded-2xl border border-line bg-surface p-6 shadow-xs transition hover:border-accent/40 hover:shadow-md cursor-pointer"
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex size-11 items-center justify-center rounded-xl bg-accent-soft text-accent group-hover:scale-105 transition">
                    <Icon className="size-5" />
                  </div>
                  <span className="rounded-md border border-line px-2 py-0.5 text-xs font-medium text-ink-muted bg-surface-muted">
                    {badge}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-ink group-hover:text-accent transition">
                  {title}
                </h3>
                <p className="text-sm leading-relaxed text-ink-muted">
                  {description}
                </p>
              </div>

              <div className="mt-5 flex items-center gap-1.5 text-xs font-semibold text-accent">
                <span>Explore feature</span>
                <ArrowRight className="size-3.5 transition group-hover:translate-x-1" />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Model & Architecture Overview Card */}
      <section className="rounded-2xl border border-line bg-surface p-6 sm:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl space-y-2">
            <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent">
              <Activity className="size-4" />
              Pretrained Model Specifications
            </div>
            <h3 className="text-xl font-bold text-ink sm:text-2xl">
              GestureSign Conv-Transformer Architecture
            </h3>
            <p className="text-sm text-ink-muted leading-relaxed">
              Trained on 94,477 clips from the Kaggle Google ASL Signs benchmark. Exports a 1.4M parameter 
              depthwise-convolutional transformer with cosine-similarity logits and 128-dimensional embedding head.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { label: 'Parameters', val: '1.4 Million' },
              { label: 'Feature Dim', val: '460 features' },
              { label: 'CPU Latency', val: '~4.4 ms' },
              { label: 'Top-5 Accuracy', val: '88.1%' },
            ].map(({ label, val }) => (
              <div key={label} className="rounded-xl border border-line bg-paper p-3.5 text-center">
                <p className="text-lg font-bold text-ink">{val}</p>
                <p className="text-xs text-ink-muted">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
