import { describe, expect, it, beforeEach } from 'vitest';
import { loadProgress, recordPracticeAttempt, resetProgress } from './progress';

describe('Progress Service', () => {
  beforeEach(() => {
    resetProgress();
  });

  it('starts with clean default progress state', () => {
    const p = loadProgress();
    expect(p.totalAttempts).toBe(0);
    expect(p.passedAttempts).toBe(0);
    expect(p.currentStreak).toBe(0);
    expect(p.bestStreak).toBe(0);
    expect(p.masteredSigns).toEqual([]);
    expect(p.history).toEqual([]);
  });

  it('records a passed attempt and increases streak', () => {
    const p1 = recordPracticeAttempt('hello', 'hello', 0.85, true);
    expect(p1.totalAttempts).toBe(1);
    expect(p1.passedAttempts).toBe(1);
    expect(p1.currentStreak).toBe(1);
    expect(p1.bestStreak).toBe(1);
    expect(p1.history.length).toBe(1);
    expect(p1.history[0]?.sign).toBe('hello');

    const p2 = recordPracticeAttempt('hello', 'hello', 0.92, true);
    expect(p2.totalAttempts).toBe(2);
    expect(p2.passedAttempts).toBe(2);
    expect(p2.currentStreak).toBe(2);
    expect(p2.bestStreak).toBe(2);
    // Signs passed at least twice with score >= 0.60 are marked as mastered
    expect(p2.masteredSigns).toContain('hello');
  });

  it('resets current streak upon failed attempt while preserving best streak', () => {
    recordPracticeAttempt('water', 'water', 0.8, true);
    recordPracticeAttempt('water', 'water', 0.8, true);
    const pFail = recordPracticeAttempt('water', 'apple', 0.2, false);

    expect(pFail.totalAttempts).toBe(3);
    expect(pFail.passedAttempts).toBe(2);
    expect(pFail.currentStreak).toBe(0);
    expect(pFail.bestStreak).toBe(2);
  });

  it('retains progress on subsequent reloads and clears on resetProgress', () => {
    recordPracticeAttempt('cat', 'cat', 0.88, true);
    recordPracticeAttempt('cat', 'cat', 0.91, true);

    // Simulate page reload by reading storage directly
    const reloaded = loadProgress();
    expect(reloaded.totalAttempts).toBe(2);
    expect(reloaded.passedAttempts).toBe(2);
    expect(reloaded.currentStreak).toBe(2);
    expect(reloaded.masteredSigns).toContain('cat');

    // Test clearing/resetting
    const reset = resetProgress();
    expect(reset.totalAttempts).toBe(0);
    expect(reset.passedAttempts).toBe(0);
    expect(reset.currentStreak).toBe(0);
    expect(reset.masteredSigns).toEqual([]);

    const afterReset = loadProgress();
    expect(afterReset.totalAttempts).toBe(0);
  });
});
