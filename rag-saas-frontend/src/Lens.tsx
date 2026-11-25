import { useState } from "react";
import { barWidth, cutoffReason, movement, type LensData } from "./lensData";

const COLLAPSED_ROWS = 10;

function Movement({ delta, from }: { delta: number; from: number }) {
  const color = delta > 0 ? "#047857" : delta < 0 ? "#b91c1c" : "#6b7280";
  const label = delta > 0 ? `▲${delta}` : delta < 0 ? `▼${-delta}` : "=";
  return (
    <span
      title={`MMR rank #${from}`}
      style={{ color, fontSize: "12px", fontWeight: 600, minWidth: "88px", display: "inline-block" }}
    >
      {label} <span style={{ color: "#9ca3af", fontWeight: 400 }}>from #{from}</span>
    </span>
  );
}

export default function Lens({ lens }: { lens: LensData }) {
  const [expanded, setExpanded] = useState(false);
  const rows = lens.candidates;
  if (rows.length === 0) return null;
  const scores = rows.map((c) => c.score);
  const [min, max] = [Math.min(...scores), Math.max(...scores)];
  const visible = expanded ? rows : rows.slice(0, COLLAPSED_ROWS);
  const lastSelected = rows.reduce((last, c, i) => (c.selected ? i : last), -1);
  const sent = rows.filter((c) => c.selected).length;
  const promoted = rows.filter((c) => c.selected && movement(c) > 0).length;

  return (
    <section
      aria-label="Lens"
      style={{
        marginTop: "12px",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        padding: "14px 16px",
        backgroundColor: "#fcfcfd",
      }}
    >
      <p style={{ margin: "0 0 10px", fontSize: "13px", color: "#4b5563" }}>
        Every chunk retrieval found, re-ranked by the CrossEncoder. Bars show the relevance score;
        arrows show how far each chunk moved from its MMR retrieval position.
        {promoted > 0 &&
          ` The reranker moved up ${promoted} of the ${sent} chunks sent to the AI.`}
      </p>
      <ol style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {visible.map((c, i) => (
          <li key={`${c.mmr_rank}`}>
            <div
              data-testid="lens-row"
              data-selected={c.selected}
              style={{
                display: "grid",
                gridTemplateColumns: "32px 1fr 64px 120px",
                alignItems: "center",
                gap: "10px",
                padding: "5px 0",
                opacity: c.selected ? 1 : 0.5,
                fontSize: "13px",
              }}
            >
              <span style={{ color: "#6b7280", fontVariantNumeric: "tabular-nums" }}>#{c.rerank_rank}</span>
              <div style={{ minWidth: 0 }}>
                <div
                  title={c.preview}
                  style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                >
                  {c.selected && <span style={{ color: "#047857", fontWeight: 600 }}>✓ </span>}
                  {c.source_file}
                  {c.page !== null && ` · p.${c.page}`}
                  <span style={{ color: "#6b7280" }}> — {c.preview}</span>
                </div>
                <div style={{ height: "6px", backgroundColor: "#f3f4f6", borderRadius: "3px", marginTop: "3px" }}>
                  <div
                    style={{
                      width: `${barWidth(c.score, min, max)}%`,
                      height: "100%",
                      borderRadius: "3px",
                      backgroundColor: c.selected ? "#3b82f6" : "#9ca3af",
                    }}
                  />
                </div>
              </div>
              <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{c.score.toFixed(2)}</span>
              <Movement delta={movement(c)} from={c.mmr_rank} />
            </div>
            {i === lastSelected && lastSelected < rows.length - 1 && (
              <div
                role="separator"
                style={{
                  borderTop: "2px dashed #f59e0b",
                  margin: "6px 0",
                  paddingTop: "4px",
                  fontSize: "12px",
                  color: "#b45309",
                }}
              >
                ✂ {cutoffReason(lens)}
              </div>
            )}
          </li>
        ))}
      </ol>
      {rows.length > COLLAPSED_ROWS && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          style={{ background: "none", border: "none", color: "#3b82f6", cursor: "pointer", padding: 0, fontSize: "13px" }}
        >
          {expanded ? "Show fewer" : `Show all ${rows.length} chunks`}
        </button>
      )}
    </section>
  );
}
