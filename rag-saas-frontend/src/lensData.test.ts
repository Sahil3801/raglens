import { expect, it } from 'vitest';
import { barWidth, cutoffReason, movement, type LensData } from './lensData';

const base: LensData = { candidates: [], threshold: -2.5, min_chunks: 4, top_n: 8, rule: 'threshold' };

it('reports how many places the reranker moved a chunk', () => {
  const c = { source_file: 'a.pdf', page: 1, preview: '', score: 1, selected: true };
  expect(movement({ ...c, mmr_rank: 19, rerank_rank: 1 })).toBe(18);
  expect(movement({ ...c, mmr_rank: 2, rerank_rank: 5 })).toBe(-3);
  expect(movement({ ...c, mmr_rank: 3, rerank_rank: 3 })).toBe(0);
});

it('scales bars between the lowest and highest score, never fully hidden', () => {
  expect(barWidth(5, -10, 5)).toBe(100);
  expect(barWidth(-10, -10, 5)).toBe(4);
  expect(barWidth(-2.5, -10, 5)).toBe(50);
  expect(barWidth(3, 3, 3)).toBe(100);
});

it('explains each cut-off rule in plain language', () => {
  expect(cutoffReason(base)).toBe('Below this line: scored under -2.50 (best score − 8), so not sent to the AI.');
  expect(cutoffReason({ ...base, rule: 'min_chunks' })).toContain('the top 4 were kept anyway');
  expect(cutoffReason({ ...base, rule: 'top_n' })).toContain('at most 8 are sent');
  expect(cutoffReason({ ...base, rule: 'all' })).toContain('Fewer than 4 chunks');
  expect(cutoffReason({ ...base, rule: 'empty' })).toBe('');
});
