import React, { useState, useEffect } from "react";
import { apiRequest, errorMessage } from "./api";
import { formatShare, matchShares, type Citation } from "./relevance";

type PipelineStats = {
  retrieved: number;
  reranked: number;
  timings_ms: { retrieval: number; reranking: number; generation: number };
};
type ChatResponse = {
  answer: string;
  sources: string[];
  citations?: Citation[];
  pipeline?: PipelineStats | null;
};

const TOP_SOURCES = 3;

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [pipeline, setPipeline] = useState<PipelineStats | null>(null);
  const [showAllSources, setShowAllSources] = useState(false);
  const shares = matchShares(citations);
  const [documents, setDocuments] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  // NEW: State to track which document is currently selected for filtering
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);

  const fetchDocuments = async () => {
    try {
      const data = await apiRequest<{ documents: string[] }>("/documents");
      setDocuments(data.documents);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
      setError(errorMessage(error));
    }
  };

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const data = await apiRequest<{ documents: string[] }>("/documents");
        setDocuments(data.documents);
      } catch (error) {
        console.error("Failed to load initial documents:", error);
        setError(errorMessage(error));
      }
    };
    loadInitialData();
  }, []);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const fileInput = event.currentTarget;
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      await apiRequest("/upload", {
        method: "POST",
        body: formData,
      });
      await fetchDocuments();

      // NEW: Automatically select the newly uploaded document!
      setSelectedDocument(file.name);
    } catch (error) {
      console.error("Upload failed:", error);
      setError(errorMessage(error));
    } finally {
      setUploading(false);
      fileInput.value = "";
    }
  };

  const handleDelete = async (filename: string) => {
    setError("");
    try {
      await apiRequest(`/documents/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      // If we delete the currently selected document, clear the selection
      if (selectedDocument === filename) {
        setSelectedDocument(null);
      }
      await fetchDocuments();
    } catch (error) {
      console.error("Delete failed:", error);
      setError(errorMessage(error));
    }
  };

  const handleAsk = async () => {
    if (!query.trim() || asking) return;
    setAsking(true);
    setError("");
    setAnswer("Thinking...");
    setCitations([]);
    setPipeline(null);
    setShowAllSources(false);

    // NEW: Dynamically build the payload based on whether a document is selected
    const payload: { query: string; filter_filename?: string } = { query };
    if (selectedDocument) {
      payload.filter_filename = selectedDocument;
    }

    try {
      const data = await apiRequest<ChatResponse>("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setAnswer(data.answer);
      setCitations(data.citations ?? []);
      setPipeline(data.pipeline ?? null);
    } catch (error) {
      console.error("Chat failed:", error);
      setAnswer("");
      setError(errorMessage(error));
    } finally {
      setAsking(false);
    }
  };

  return (
    <div
      style={{ display: "flex", minHeight: "100vh", fontFamily: "sans-serif" }}
    >
      {/* SIDEBAR: Document Management */}
      <div
        style={{
          width: "300px",
          backgroundColor: "#f3f4f6",
          padding: "20px",
          borderRight: "1px solid #e5e7eb",
        }}
      >
        <h2 style={{ fontSize: "18px", marginBottom: "15px" }}>
          Knowledge Base
        </h2>

        <label
          style={{
            display: "block",
            marginBottom: "20px",
            cursor: "pointer",
            backgroundColor: "#3b82f6",
            color: "white",
            padding: "10px",
            textAlign: "center",
            borderRadius: "5px",
          }}
        >
          {uploading ? "Uploading..." : "+ Upload PDF"}
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            style={{ display: "none" }}
            disabled={uploading}
          />
        </label>

        <div>
          <h3
            style={{ fontSize: "14px", color: "#6b7280", marginBottom: "10px" }}
          >
            UPLOADED FILES (Click to filter)
          </h3>
          {documents.length === 0 ? (
            <p style={{ fontSize: "14px", color: "#9ca3af" }}>
              No documents uploaded yet.
            </p>
          ) : (
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {documents.map((doc) => {
                const isSelected = selectedDocument === doc;
                return (
                  <li
                    key={doc}
                    // NEW: Clicking the row selects/deselects the document
                    onClick={() => setSelectedDocument(isSelected ? null : doc)}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      backgroundColor: isSelected ? "#eff6ff" : "white", // Highlight if selected
                      padding: "10px",
                      marginBottom: "8px",
                      borderRadius: "4px",
                      border: isSelected
                        ? "2px solid #3b82f6"
                        : "1px solid #e5e7eb",
                      cursor: "pointer",
                      transition: "all 0.2s",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "14px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        fontWeight: isSelected ? "bold" : "normal",
                        color: isSelected ? "#1d4ed8" : "black",
                      }}
                    >
                      {doc}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation(); // NEW: Prevents row click when clicking delete
                        handleDelete(doc);
                      }}
                      style={{
                        color: "#ef4444",
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        fontSize: "18px",
                      }}
                    >
                      &times;
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      {/* MAIN CONTENT: Chat Interface */}
      <div
        style={{
          flex: 1,
          padding: "40px",
          maxWidth: "800px",
          margin: "0 auto",
        }}
      >
        <h1 style={{ fontSize: "24px", marginBottom: "5px" }}>
          Evaluation-First RAG
        </h1>
        {error && <p role="alert" style={{ color: "#b91c1c" }}>{error}</p>}

        {/* NEW: Visual indicator of current mode */}
        <p style={{ color: "#6b7280", marginBottom: "20px", fontSize: "14px" }}>
          Mode:{" "}
          {selectedDocument ? (
            <strong>Filtering by {selectedDocument}</strong>
          ) : (
            <strong>Global Search (All Documents)</strong>
          )}
        </p>

        <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What is Chinmay's highest education?"
            style={{
              flex: 1,
              padding: "10px",
              border: "1px solid #d1d5db",
              borderRadius: "5px",
            }}
          />
          <button
            onClick={handleAsk}
            disabled={asking}
            style={{
              padding: "10px 20px",
              backgroundColor: "#10b981",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Ask
          </button>
        </div>

        {answer && (
          <div
            style={{
              backgroundColor: "#f9fafb",
              padding: "20px",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
              whiteSpace: "pre-wrap",
            }}
          >
            <strong>Answer:</strong>
            <p style={{ marginTop: "10px" }}>{answer}</p>
          </div>
        )}

        {pipeline && (
          <p
            data-testid="pipeline-stats"
            style={{ marginTop: "12px", fontSize: "13px", color: "#6b7280" }}
          >
            Pipeline: {pipeline.retrieved} chunks retrieved (MMR) →{" "}
            {pipeline.reranked} kept after reranking · retrieval{" "}
            {Math.round(pipeline.timings_ms.retrieval)} ms · reranking{" "}
            {Math.round(pipeline.timings_ms.reranking)} ms · LLM{" "}
            {Math.round(pipeline.timings_ms.generation)} ms
          </p>
        )}

        {citations.length > 0 && (
          <div style={{ marginTop: "20px" }}>
            <h2 style={{ fontSize: "16px", marginBottom: "4px" }}>Sources</h2>
            {shares && (
              <p style={{ fontSize: "12px", color: "#6b7280", marginTop: 0, marginBottom: "10px" }}>
                Match % = how strongly the reranker connected each source to your question,
                compared with all {citations.length} sources given to the AI (they add up to 100%).
                Higher is better.
              </p>
            )}
            <ol style={{ paddingLeft: "20px", margin: 0 }}>
              {(showAllSources ? citations : citations.slice(0, TOP_SOURCES)).map((citation, index) => (
                <li key={index} style={{ marginBottom: "8px", fontSize: "14px" }}>
                  <details>
                    <summary style={{ cursor: "pointer" }}>
                      {citation.source_file}
                      {citation.page !== null && ` · page ${citation.page}`}
                      {shares && typeof citation.score === "number" && (
                        <span
                          style={{ color: "#6b7280" }}
                          title={`Raw reranker score: ${citation.score.toFixed(2)}`}
                        >
                          {` · match ${formatShare(shares[index])}`}
                        </span>
                      )}
                    </summary>
                    <p
                      style={{
                        whiteSpace: "pre-wrap",
                        color: "#4b5563",
                        backgroundColor: "#f9fafb",
                        border: "1px solid #e5e7eb",
                        borderRadius: "5px",
                        padding: "10px",
                        marginTop: "6px",
                      }}
                    >
                      {citation.text}
                    </p>
                  </details>
                </li>
              ))}
            </ol>
            {citations.length > TOP_SOURCES && (
              <button
                type="button"
                onClick={() => setShowAllSources(!showAllSources)}
                style={{
                  marginTop: "6px",
                  background: "none",
                  border: "none",
                  color: "#3b82f6",
                  cursor: "pointer",
                  padding: 0,
                  fontSize: "14px",
                }}
              >
                {showAllSources ? "Show top 3" : `Show all ${citations.length}`}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
