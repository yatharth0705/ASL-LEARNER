export interface PracticeRecord {
  id: string;
  sign: string;
  target: string;
  score: number;
  passed: boolean;
  timestamp: number;
}

export interface UserProgress {
  totalAttempts: number;
  passedAttempts: number;
  currentStreak: number;
  bestStreak: number;
  masteredSigns: string[];
  history: PracticeRecord[];
}

const STORAGE_KEY = 'asl_learner_progress_v1';

function getDefaultProgress(): UserProgress {
  return {
    totalAttempts: 0,
    passedAttempts: 0,
    currentStreak: 0,
    bestStreak: 0,
    masteredSigns: [],
    history: [],
  };
}

let memoryStorage: Record<string, string> = {};

function storageGet(key: string): string | null {
  try {
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem(key);
    }
  } catch {
    // fallback
  }
  return memoryStorage[key] ?? null;
}

function storageSet(key: string, value: string): void {
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(key, value);
    }
  } catch {
    // fallback
  }
  memoryStorage[key] = value;
}

export function loadProgress(): UserProgress {
  try {
    const raw = storageGet(STORAGE_KEY);
    if (!raw) return getDefaultProgress();
    const parsed = JSON.parse(raw);
    return {
      ...getDefaultProgress(),
      ...parsed,
    };
  } catch {
    return getDefaultProgress();
  }
}

export function saveProgress(progress: UserProgress): void {
  storageSet(STORAGE_KEY, JSON.stringify(progress));
}

export function recordPracticeAttempt(
  target: string,
  detectedSign: string,
  score: number,
  passed: boolean,
): UserProgress {
  const current = loadProgress();
  const newStreak = passed ? current.currentStreak + 1 : 0;
  const bestStreak = Math.max(current.bestStreak, newStreak);

  const record: PracticeRecord = {
    id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    sign: detectedSign,
    target,
    score,
    passed,
    timestamp: Date.now(),
  };

  const updatedHistory = [record, ...current.history].slice(0, 100);

  // Check if sign is mastered: passed at least 2 times with score >= 0.60
  const targetPassedCount = updatedHistory.filter(
    (h) => h.target.toLowerCase() === target.toLowerCase() && h.passed && h.score >= 0.6,
  ).length;

  const masteredSet = new Set(current.masteredSigns);
  if (targetPassedCount >= 2) {
    masteredSet.add(target);
  }

  const updated: UserProgress = {
    totalAttempts: current.totalAttempts + 1,
    passedAttempts: current.passedAttempts + (passed ? 1 : 0),
    currentStreak: newStreak,
    bestStreak,
    masteredSigns: Array.from(masteredSet),
    history: updatedHistory,
  };

  saveProgress(updated);
  return updated;
}

export function resetProgress(): UserProgress {
  memoryStorage = {};
  const def = getDefaultProgress();
  saveProgress(def);
  return def;
}
