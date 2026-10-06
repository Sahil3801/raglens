import { expect, it } from 'vitest';
import { formatShare, matchShares } from './relevance';

const cite = (score?: number | null) => ({ source_file: 'a.pdf', page: 1, text: 't', score });

it('turns raw reranker scores into match shares that add up to 100% in the same order', () => {
  // Real scores from a resume question: all negative, yet the top source clearly wins.
  const shares = matchShares([-8.98, -11.18, -11.33, -12.5].map(cite))!;
  expect(shares.reduce((a, b) => a + b, 0)).toBeCloseTo(100);
  expect(shares.map(Math.round)).toEqual([81, 9, 8, 2]);
  expect([...shares].sort((a, b) => b - a)).toEqual(shares);
});

it('gives a single source 100% and stays finite for extreme scores', () => {
  expect(matchShares([cite(-11)])).toEqual([100]);
  expect(matchShares([1000, -1000].map(cite))).toEqual([100, 0]);
});

it('returns null when scores are missing (older backend)', () => {
  expect(matchShares([])).toBeNull();
  expect(matchShares([cite(2), cite(undefined)])).toBeNull();
  expect(matchShares([cite(null)])).toBeNull();
});

it('formats tiny shares as <1%', () => {
  expect(formatShare(76.6)).toBe('77%');
  expect(formatShare(0.4)).toBe('<1%');
});
