import { describe, expect, it } from 'vitest';
import { ASL_ALPHABET } from './alphabet-data';
import { SIGN_CATEGORIES, getCategoryForSign } from './categories';

describe('ASL Alphabet Data', () => {
  it('covers all 26 letters of the English alphabet', () => {
    expect(ASL_ALPHABET.length).toBe(26);
    const letters = ASL_ALPHABET.map((l) => l.letter);
    const expected = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    expect(letters).toEqual(expected);
  });

  it('provides non-empty descriptions, mnemonics, and mistakes for every letter', () => {
    for (const item of ASL_ALPHABET) {
      expect(item.handshapeDescription.length).toBeGreaterThan(10);
      expect(item.mnemonic.length).toBeGreaterThan(5);
      expect(item.commonMistake.length).toBeGreaterThan(5);
      expect(item.palmOrientation.length).toBeGreaterThan(5);
    }
  });
});

describe('Sign Categories Data', () => {
  it('contains expected core categories', () => {
    expect(SIGN_CATEGORIES.length).toBeGreaterThanOrEqual(6);
    const names = SIGN_CATEGORIES.map((c) => c.id);
    expect(names).toContain('essentials');
    expect(names).toContain('animals');
    expect(names).toContain('food');
    expect(names).toContain('colors');
  });

  it('correctly maps signs to their category', () => {
    expect(getCategoryForSign('hello')?.id).toBe('essentials');
    expect(getCategoryForSign('tiger')?.id).toBe('animals');
    expect(getCategoryForSign('water')?.id).toBe('food');
    expect(getCategoryForSign('blue')?.id).toBe('colors');
  });
});

