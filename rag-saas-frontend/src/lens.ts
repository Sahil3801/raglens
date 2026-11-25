export type LensCandidate = {
  source_file: string;
  page: number | null;
  preview: string;
  mmr_rank: number;
  rerank_rank: number;
  score: number;
  selected: boolean;
};

export type LensData = {
  candidates: LensCandidate[];
  threshold: number | null;
  min_chunks: number;
  top_n: number;
  rule: "threshold" | "min_chunks" | "all" | "top_n" | "empty" | string;
};

/** Positions gained (+) or lost (-) between MMR retrieval and reranking. */
export function movement(c: LensCandidate): number {
  return c.mmr_rank - c.rerank_rank;
}

/** Bar width in percent, scaled between the lowest and highest score in this answer. */
export function barWidth(score: number, min: number, max: number): number {
  if (max === min) return 100;
  return Math.max(4, Math.round(((score - min) / (max - min)) * 100));
}

/** Plain-language explanation of where the cut-off fell and why. */
export function cutoffReason(lens: LensData): string {
  const threshold = lens.threshold?.toFixed(2);
  switch (lens.rule) {
    case "threshold":
      return `Below this line: scored under ${threshold} (best score − 8), so not sent to the AI.`;
    case "min_chunks":
      return `Few chunks scored above ${threshold} (best − 8), so the top ${lens.min_chunks} were kept anyway.`;
    case "top_n":
      return `More chunks passed the threshold, but at most ${lens.top_n} are sent to the AI.`;
    case "all":
      return `Fewer than ${lens.min_chunks} chunks were found, so all of them were sent.`;
    default:
      return "";
  }
}
