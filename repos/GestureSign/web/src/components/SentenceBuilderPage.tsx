import { AlertTriangle, Check, Copy, CornerDownLeft, Pause, Play, Trash2, Volume2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import type { LoadedModel } from '../ml/model-client';
import { Stage } from './Stage';
import { Card } from './ui';

interface SentenceBuilderPageProps {
  session: Session;
  snap: SessionSnapshot;
  model: LoadedModel;
}

interface WordToken {
  id: string;
  word: string;
  confidence: number;
  timestamp: number;
}

export function SentenceBuilderPage({ session, snap, model }: SentenceBuilderPageProps) {
  const { labels } = model;
  const [tokens, setTokens] = useState<WordToken[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastAddedSign, setLastAddedSign] = useState<string | null>(null);
  const [holdPrevented, setHoldPrevented] = useState<string | null>(null);

  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;

  const lastAddedSignRef = useRef(lastAddedSign);
  lastAddedSignRef.current = lastAddedSign;

  // Listen to recognized signs
  useEffect(() => {
    return session.onResult((result) => {
      if (isPausedRef.current) return;

      const top = result.decision.top[0];
      if (result.decision.kind === 'sign' && top) {
        const signName = labels[top.index]?.label ?? 'unknown';

        // Duplicate prevention logic:
        // If user is holding the same sign or repeating without a pause, prevent auto-spamming
        if (lastAddedSignRef.current?.toLowerCase() === signName.toLowerCase()) {
          setHoldPrevented(signName);
          setTimeout(() => setHoldPrevented(null), 2500);
          return;
        }

        const newToken: WordToken = {
          id: `${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
          word: signName,
          confidence: top.probability,
          timestamp: Date.now(),
        };

        setTokens((prev) => [...prev, newToken]);
        setLastAddedSign(signName);
        setHoldPrevented(null);
      }
    });
  }, [session, labels]);

  // Construct text sentence
  const constructedSentence = tokens
    .map((t) => t.word)
    .join(' ')
    .trim();

  const handleClear = () => {
    setTokens([]);
    setLastAddedSign(null);
    setHoldPrevented(null);
  };

  const handleBackspace = () => {
    setTokens((prev) => prev.slice(0, -1));
    const remaining = tokens.slice(0, -1);
    setLastAddedSign(remaining.length > 0 ? (remaining[remaining.length - 1]?.word ?? null) : null);
  };

  const handleAddSpace = () => {
    // Reset lastAddedSign so the user can sign the same word again deliberately
    setLastAddedSign(null);
  };

  const handleCopy = () => {
    if (!constructedSentence) return;
    navigator.clipboard.writeText(constructedSentence).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleSpeak = () => {
    if (!constructedSentence || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(constructedSentence);
    utterance.rate = 0.9;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="space-y-6 pb-16">
      {/* Linguistic Disclaimer Banner */}
      <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5 shadow-xs">
        <div className="flex items-start gap-3.5">
          <AlertTriangle className="size-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1.5 text-xs text-ink-muted leading-relaxed">
            <h2 className="font-bold text-ink text-sm">
              Phase 4: Isolated-Sign Sentence Builder (Experimental Feature)
            </h2>
            <p>
              <strong>Important Linguistic Distinction:</strong> This tool concatenates individually articulated, isolated ASL citation signs into a linear English sequence. 
              It is <strong>NOT</strong> genuine continuous ASL translation. Natural ASL has its own distinct grammatical syntax, spatial indexing, non-manual facial markers, and fluid co-articulation that does not translate word-for-word into English sentences.
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid: Camera Stage on Left, Sentence Canvas on Right */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Column: Live Camera Stage */}
        <div className="lg:col-span-7 space-y-4">
          <Stage session={session} snap={snap} />

          {/* Duplicate Hold Status Alert */}
          {holdPrevented && (
            <div className="rounded-xl border border-accent/30 bg-accent-soft p-3 text-xs text-accent font-medium flex items-center justify-between animate-fade-in">
              <span>Holding sign <strong>"{holdPrevented}"</strong> — duplicate addition prevented.</span>
              <button
                onClick={handleAddSpace}
                className="text-[11px] underline font-semibold ml-2 hover:opacity-80"
              >
                Sign again (allow duplicate)
              </button>
            </div>
          )}
        </div>

        {/* Right Column: Sentence Workspace Canvas */}
        <div className="lg:col-span-5 space-y-5">
          <Card className="p-5 space-y-4 shadow-xs">
            <div className="flex items-center justify-between border-b border-line pb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-ink-muted">
                Accumulated Sentence
              </span>
              <div className="flex items-center gap-2">
                <span className="rounded-md bg-surface-muted px-2 py-0.5 text-[11px] font-semibold text-ink-muted">
                  {tokens.length} {tokens.length === 1 ? 'sign' : 'signs'}
                </span>
                <button
                  onClick={() => setIsPaused(!isPaused)}
                  className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                    isPaused
                      ? 'bg-amber-500/10 text-amber-600 border border-amber-500/20'
                      : 'bg-surface-muted text-ink-muted hover:text-ink'
                  }`}
                >
                  {isPaused ? <Play className="size-3" /> : <Pause className="size-3" />}
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
              </div>
            </div>

            {/* Sentence Display Area */}
            <div className="min-h-[140px] rounded-2xl border border-line bg-paper p-4 flex flex-col justify-between">
              {tokens.length === 0 ? (
                <div className="my-auto text-center text-xs text-ink-muted/70 italic py-4">
                  Sign to the camera to add words here. Each completed isolated sign will be appended as a word token.
                </div>
              ) : (
                <div className="flex flex-wrap gap-2 items-center">
                  {tokens.map((token) => (
                    <span
                      key={token.id}
                      className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-surface px-3 py-1.5 text-sm font-semibold text-ink shadow-xs transition hover:border-accent"
                    >
                      <span>{token.word}</span>
                      <span className="text-[10px] text-ink-muted font-normal">
                        {Math.round(token.confidence * 100)}%
                      </span>
                    </span>
                  ))}
                </div>
              )}

              {/* Text Preview Bar */}
              {constructedSentence && (
                <div className="mt-4 pt-3 border-t border-line text-xs font-mono text-ink-muted bg-surface-muted/50 p-2.5 rounded-lg">
                  "{constructedSentence}"
                </div>
              )}
            </div>

            {/* Action Buttons Toolbar */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                onClick={handleSpeak}
                disabled={!constructedSentence}
                className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-xl bg-accent px-3 py-2.5 text-xs font-semibold text-accent-ink transition hover:bg-accent/90 disabled:opacity-40"
              >
                <Volume2 className="size-4" />
                {isSpeaking ? 'Speaking…' : 'Speak Aloud'}
              </button>

              <button
                onClick={handleCopy}
                disabled={!constructedSentence}
                className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-line bg-surface px-3 py-2.5 text-xs font-medium text-ink transition hover:bg-surface-muted disabled:opacity-40"
                title="Copy to clipboard"
              >
                {copied ? <Check className="size-4 text-emerald-600" /> : <Copy className="size-4" />}
                {copied ? 'Copied' : 'Copy'}
              </button>

              <button
                onClick={handleBackspace}
                disabled={tokens.length === 0}
                className="inline-flex items-center justify-center rounded-xl border border-line bg-surface p-2.5 text-xs font-medium text-ink hover:bg-surface-muted transition disabled:opacity-40"
                title="Delete last word"
              >
                <CornerDownLeft className="size-4" />
              </button>

              <button
                onClick={handleClear}
                disabled={tokens.length === 0}
                className="inline-flex items-center justify-center rounded-xl border border-danger/30 bg-danger-soft p-2.5 text-xs font-medium text-danger hover:bg-danger/20 transition disabled:opacity-40"
                title="Clear all words"
              >
                <Trash2 className="size-4" />
              </button>
            </div>
          </Card>

          {/* Sequence Tips */}
          <div className="rounded-xl border border-line bg-surface p-4 text-xs text-ink-muted space-y-2">
            <span className="font-semibold text-ink flex items-center gap-1.5">
              Guidelines:
            </span>
            <p>
              • <strong>Holding signs:</strong> If you keep your hands in the same sign pose, the builder deliberately prevents duplicate additions.
            </p>
            <p>
              • <strong>Repeating a sign:</strong> To sign the same word twice (e.g. "YES YES"), lower your hands briefly between attempts or press the space control.
            </p>
            <p>
              • <strong>Audio Playback:</strong> Click "Speak Aloud" to hear your phrase converted to speech via the Web Speech Synthesis API.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
