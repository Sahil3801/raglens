import React, { useState, useEffect } from "react";
import { apiRequest, errorMessage } from "./api";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
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

    // NEW: Dynamically build the payload based on whether a document is selected
    const payload: { query: string; filter_filename?: string } = { query };
    if (selectedDocument) {
      payload.filter_filename = selectedDocument;
    }

    try {
      const data = await apiRequest<{ answer: string; sources: string[] }>("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setAnswer(data.answer);
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
      </div>
    </div>
  );
}

export default App;
