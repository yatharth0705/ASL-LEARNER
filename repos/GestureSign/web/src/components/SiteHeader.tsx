import { BookOpen, CheckCircle2, Cpu, Eye, History, Home, Info, Languages, Layers, LayoutDashboard, Menu, Sparkles, X } from 'lucide-react';
import { useState } from 'react';
import type { SessionSnapshot } from '../app/session';

export type AppPage =
  | 'landing'
  | 'dashboard'
  | 'recognition'
  | 'practice'
  | 'continuous'
  | 'sentence'
  | 'alphabet'
  | 'progress'
  | 'about';

interface SiteHeaderProps {
  currentPage: AppPage;
  onNavigate: (page: AppPage) => void;
  snap: SessionSnapshot;
}

export function SiteHeader({ currentPage, onNavigate, snap }: SiteHeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const modelStatus = snap.model.status;

  const navItems: { id: AppPage; label: string; icon: typeof Home }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'recognition', label: 'Live Recognition', icon: Eye },
    { id: 'continuous', label: 'Continuous (Exp)', icon: Languages },
    { id: 'practice', label: 'Practice Mode', icon: Sparkles },
    { id: 'sentence', label: 'Sentence Builder', icon: Layers },
    { id: 'alphabet', label: 'ASL Alphabet', icon: BookOpen },
    { id: 'progress', label: 'Progress', icon: History },
    { id: 'about', label: 'About & Model', icon: Info },
  ];

  const handleNav = (page: AppPage) => {
    onNavigate(page);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-paper/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Brand Logo */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleNav('landing')}
            className="flex items-center gap-2.5 text-left font-semibold tracking-tight transition hover:opacity-85"
            aria-label="ASL-Learner Home"
          >
            <div className="flex size-9 items-center justify-center rounded-xl bg-accent text-accent-ink shadow-sm">
              <span className="text-lg font-black tracking-tighter">ASL</span>
            </div>
            <div>
              <span className="text-base font-bold text-ink">ASL-Learner</span>
              <span className="ml-1.5 hidden rounded-md bg-accent-soft px-1.5 py-0.5 text-[10px] font-semibold text-accent sm:inline-block">
                PRETRAINED AI
              </span>
            </div>
          </button>
        </div>

        {/* Desktop Navigation */}
        <nav aria-label="Main Navigation" className="hidden lg:flex items-center gap-1 text-sm font-medium">
          {navItems.map(({ id, label, icon: Icon }) => {
            const isActive = currentPage === id;
            return (
              <button
                key={id}
                onClick={() => handleNav(id)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-2 transition ${
                  isActive
                    ? 'bg-accent text-accent-ink font-semibold shadow-xs'
                    : 'text-ink-muted hover:bg-surface-muted hover:text-ink'
                }`}
              >
                <Icon className={`size-4 ${isActive ? 'text-accent-ink' : 'text-ink-muted'}`} />
                {label}
              </button>
            );
          })}
        </nav>

        {/* Header Right: Model Status & Mobile Toggle */}
        <div className="flex items-center gap-2">
          {/* Live Model Status Pill */}
          <div
            className={`hidden sm:inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium shadow-xs ${
              modelStatus === 'ready'
                ? 'border-success/30 bg-success-soft text-success'
                : modelStatus === 'loading'
                  ? 'border-accent/30 bg-accent-soft text-accent animate-pulse'
                  : 'border-danger/30 bg-danger-soft text-danger'
            }`}
            title="Pretrained Conv-Transformer (1.4M params) running on-device via WebAssembly"
          >
            {modelStatus === 'ready' ? (
              <>
                <CheckCircle2 className="size-3.5" />
                <span>ONNX 250 Signs Ready</span>
              </>
            ) : modelStatus === 'loading' ? (
              <>
                <Cpu className="size-3.5 animate-spin" />
                <span>Loading ONNX Model…</span>
              </>
            ) : (
              <span>Model Offline</span>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="rounded-lg p-2 text-ink-muted hover:bg-surface-muted hover:text-ink lg:hidden"
            aria-label="Toggle menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X className="size-6" /> : <Menu className="size-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Drawer */}
      {mobileMenuOpen && (
        <div className="border-b border-line bg-surface px-4 pt-2 pb-4 shadow-lg lg:hidden">
          <div className="mb-3 flex items-center justify-between px-2 pt-1 pb-2 border-b border-line">
            <span className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Navigation</span>
            <span
              className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                modelStatus === 'ready'
                  ? 'bg-success-soft text-success'
                  : 'bg-accent-soft text-accent'
              }`}
            >
              <Cpu className="size-3" />
              {modelStatus === 'ready' ? 'Model Ready' : 'Loading Model'}
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <button
              onClick={() => handleNav('landing')}
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-left ${
                currentPage === 'landing' ? 'bg-accent text-accent-ink font-semibold' : 'hover:bg-surface-muted text-ink'
              }`}
            >
              <Home className="size-4" />
              Home / Overview
            </button>
            {navItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => handleNav(id)}
                className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-left ${
                  currentPage === id ? 'bg-accent text-accent-ink font-semibold' : 'hover:bg-surface-muted text-ink'
                }`}
              >
                <Icon className="size-4" />
                {label}
              </button>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
