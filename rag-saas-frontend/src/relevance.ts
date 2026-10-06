export type Citation = { source_file: string; page: number | null; text: string; score?: number | null };

// Raw CrossEncoder scores are unbounded logits (e.g. -9 or +7) that mean little to
// readers. A softmax turns them into each source's share of the total match, so
// the shares add up to 100% across the sources sent to the LLM.
export function matchShares(citations: Citation[]): number[] | null {
  const scores = citations.map((c) => c.score);
  if (scores.length === 0 || scores.some((s) => typeof s !== "number")) return null;
  const best = Math.max(...(scores as number[]));
  const weights = (scores as number[]).map((s) => Math.exp(s - best));
  const total = weights.reduce((sum, w) => sum + w, 0);
  return weights.map((w) => (100 * w) / total);
}

export function formatShare(share: number): string {
  return share >= 1 ? `${Math.round(share)}%` : "<1%";
}
