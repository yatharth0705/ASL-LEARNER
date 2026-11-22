import { AlertCircle, ArrowRight, BookOpen, Compass, Eye, Info, Lightbulb, Search, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import { ASL_ALPHABET, type AlphabetLetter } from '../lib/alphabet-data';
import type { AppPage } from './SiteHeader';

interface AlphabetPageProps {
  onNavigate: (page: AppPage) => void;
}

export function AlphabetPage({ onNavigate }: AlphabetPageProps) {
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [activeLetter, setActiveLetter] = useState<AlphabetLetter | null>(ASL_ALPHABET[0] ?? null);

  const filteredLetters = useMemo(() => {
    return ASL_ALPHABET.filter((item) => {
      const matchSearch =
        item.letter.toLowerCase().includes(search.toLowerCase()) ||
        item.mnemonic.toLowerCase().includes(search.toLowerCase()) ||
        item.handshapeDescription.toLowerCase().includes(search.toLowerCase());
      const matchCategory =
        selectedCategory === 'all' || item.category === selectedCategory;
      return matchSearch && matchCategory;
    });
  }, [search, selectedCategory]);

  return (
    <div className="space-y-8 pb-16">
      {/* Page Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent mb-1">
            <BookOpen className="size-3.5" />
            ASL Manual Alphabet Reference
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">
            American Sign Language Fingerspelling (A-Z)
          </h1>
          <p className="text-sm text-ink-muted">
            Master the 26 handshapes used to spell names, locations, and words without dedicated signs.
          </p>
        </div>

        <button
          onClick={() => onNavigate('recognition')}
          className="inline-flex items-center gap-1.5 self-start rounded-xl bg-accent px-4 py-2.5 text-xs font-semibold text-accent-ink shadow-sm hover:bg-accent/90 transition"
        >
          <Eye className="size-4" />
          Test in Camera
        </button>
      </div>

      {/* Scope and AI Clarification Banner */}
      <div className="rounded-2xl border border-accent/20 bg-accent-soft/30 p-4 text-xs text-ink-muted leading-relaxed">
        <p className="font-semibold text-ink flex items-center gap-1.5 mb-1">
          <Info className="size-4 text-accent" />
          Educational Reference Scope & Recognition Clarification:
        </p>
        <p>
          This fingerspelling module is an <strong>educational reference guide</strong>. The primary pretrained ONNX AI model is trained on <strong>250 complete ASL vocabulary concepts</strong> (e.g. &ldquo;hello&rdquo;, &ldquo;water&rdquo;, &ldquo;thank you&rdquo;) rather than single isolated alphabet letters. Additionally, letters such as <strong>J</strong> and <strong>Z</strong> inherently require continuous air-tracing motion rather than static poses.
        </p>
      </div>

      {/* Filter & Search Bar */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-ink-muted" />
          <input
            type="text"
            placeholder="Search letter, formation rule, or keyword..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-xl border border-line bg-surface pl-10 pr-4 py-2 text-sm text-ink placeholder:text-ink-muted focus:border-accent focus:outline-none"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink"
            >
              <X className="size-4" />
            </button>
          )}
        </div>

        {/* Category Pills */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          {[
            { id: 'all', label: 'All (26)' },
            { id: 'vowel', label: 'Vowels' },
            { id: 'pointing', label: 'Pointing / Digits' },
            { id: 'fist', label: 'Fist Handshapes' },
            { id: 'open', label: 'Open Palm' },
            { id: 'special', label: 'Motion / Special' },
          ].map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setSelectedCategory(id)}
              className={`rounded-lg px-3 py-1.5 font-medium transition ${
                selectedCategory === id
                  ? 'bg-accent text-accent-ink shadow-xs'
                  : 'bg-surface border border-line text-ink-muted hover:bg-surface-muted hover:text-ink'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Letter Grid & Detail View */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Letter Cards Grid */}
        <div className="lg:col-span-7 grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-4">
          {filteredLetters.map((item) => {
            const isSelected = activeLetter?.letter === item.letter;
            return (
              <button
                key={item.letter}
                onClick={() => setActiveLetter(item)}
                className={`relative flex flex-col items-center justify-center rounded-2xl border p-4 text-center transition ${
                  isSelected
                    ? 'border-accent bg-accent-soft/40 shadow-sm ring-2 ring-accent'
                    : 'border-line bg-surface hover:border-accent/40 hover:bg-surface-muted'
                }`}
              >
                <span className="text-3xl font-black text-ink">{item.letter}</span>
                <span className="mt-1 text-[10px] font-semibold uppercase tracking-wider text-ink-muted">
                  {item.category}
                </span>
                {isSelected && (
                  <span className="absolute top-2 right-2 flex size-2 rounded-full bg-accent" />
                )}
              </button>
            );
          })}
        </div>

        {/* Active Letter Detail Showcase */}
        <div className="lg:col-span-5">
          {activeLetter ? (
            <div className="sticky top-20 rounded-2xl border border-line bg-surface p-6 shadow-sm space-y-6">
              <div className="flex items-center justify-between border-b border-line pb-4">
                <div className="flex items-center gap-3">
                  <div className="flex size-14 items-center justify-center rounded-2xl bg-accent text-accent-ink text-3xl font-black shadow-xs">
                    {activeLetter.letter}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-ink">
                      Letter "{activeLetter.letter}"
                    </h3>
                    <span className="inline-block rounded-md bg-accent-soft px-2 py-0.5 text-xs font-semibold text-accent capitalize">
                      {activeLetter.category} handshape
                    </span>
                  </div>
                </div>
              </div>

              {/* How to Form */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-ink uppercase tracking-wider">
                  <Compass className="size-4 text-accent" />
                  How to form this handshape
                </div>
                <p className="text-sm text-ink-muted leading-relaxed rounded-xl bg-paper p-3.5 border border-line">
                  {activeLetter.handshapeDescription}
                </p>
              </div>

              {/* Mnemonic / Memory Tip */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
                  <Lightbulb className="size-4" />
                  Memory Tip (Mnemonic)
                </div>
                <p className="text-sm text-ink-muted leading-relaxed rounded-xl bg-emerald-500/10 p-3.5 border border-emerald-500/20">
                  {activeLetter.mnemonic}
                </p>
              </div>

              {/* Common Pitfall */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider">
                  <AlertCircle className="size-4" />
                  Common Beginner Mistake
                </div>
                <p className="text-sm text-ink-muted leading-relaxed rounded-xl bg-amber-500/10 p-3.5 border border-amber-500/20">
                  {activeLetter.commonMistake}
                </p>
              </div>

              {/* Palm Orientation */}
              <div className="space-y-1.5">
                <div className="text-xs font-semibold text-ink-muted uppercase tracking-wider">
                  Palm Orientation
                </div>
                <p className="text-xs font-medium text-ink bg-surface-muted px-3 py-2 rounded-lg border border-line">
                  {activeLetter.palmOrientation}
                </p>
              </div>

              {/* Action Button */}
              <div className="pt-2">
                <button
                  onClick={() => onNavigate('recognition')}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-3 text-sm font-semibold text-accent-ink shadow-sm transition hover:bg-accent/90"
                >
                  <Eye className="size-4" />
                  Practice in Live Camera
                  <ArrowRight className="size-4" />
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-line bg-surface p-12 text-center text-ink-muted">
              Select any letter to view formation instructions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
