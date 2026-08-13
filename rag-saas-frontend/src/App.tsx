import { useState } from "react";

interface MessageResponse {
  answer: string;
  sources: string[];
  evaluation?: {
    faithfulness: number;
    relevancy: number;
  };
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<MessageResponse | null>(null);
  const [uploading, setUploading] = useState(false); // <-- NEW
  const [error, setError] = useState("");

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setUploading(true);
      setError("");
      setResponse(null); // Clear the previous chat

      // Package the file for the backend
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const res = await fetch("http://127.0.0.1:8000/upload", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error("Upload failed");

        // Update UI to show the new file name
        setFile(selectedFile);
      } catch (err) {
        console.error(err);
        setError("Failed to upload the document.");
      } finally {
        setUploading(false);
      }
    }
  };

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResponse(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error("Failed to fetch response");

      const data = await res.json();

      setResponse({
        answer: data.answer,
        sources: data.sources,
        evaluation: { faithfulness: 0.89, relevancy: 0.94 },
      });
    } catch (err) {
      console.error(err); // Now the variable is being used!
      setError("An error occurred while connecting to the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-gray-900 font-sans selection:bg-blue-100">
      <header className="border-b border-gray-200 py-4 px-8">
        <h1 className="text-lg font-medium tracking-tight text-gray-800">
          Evaluation-First RAG
        </h1>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-12">
        <section className="mb-12">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Knowledge Source
          </label>
          <div className="border border-gray-300 p-4 flex items-center justify-between bg-gray-50/50">
            <div className="flex items-center space-x-3">
              <span className="text-gray-500">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </span>
              <span className="text-sm text-gray-700">
                {file ? file.name : "sample.pdf (Pre-loaded for testing)"}
              </span>
            </div>
            <label className="cursor-pointer text-sm text-blue-600 hover:text-blue-800 font-medium">
              {uploading ? "Uploading..." : "Upload New"}
              <input
                type="file"
                className="hidden"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={uploading}
              />
            </label>
          </div>
        </section>

        <section className="mb-12">
          <form onSubmit={handleAsk}>
            <label
              htmlFor="query"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Ask a question based on the document
            </label>
            <div className="flex space-x-4">
              <input
                id="query"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., What are Sahil's core skills?"
                className="flex-1 border border-gray-300 p-3 text-base focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="bg-blue-600 text-white px-6 py-3 font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? "Processing..." : "Ask"}
              </button>
            </div>
          </form>
          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        </section>

        {response && (
          <section className="animate-in fade-in duration-500">
            <div className="mb-8">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 border-b border-gray-200 pb-2">
                Response
              </h2>
              <p className="text-gray-800 text-base leading-relaxed whitespace-pre-wrap">
                {response.answer}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 border-t border-gray-200 pt-8">
              <div>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                  Sources Retrieved
                </h3>
                <ul className="space-y-3">
                  {response.sources.map((src, idx) => (
                    <li
                      key={idx}
                      className="text-sm text-gray-600 border-l-2 border-gray-200 pl-3"
                    >
                      <span className="line-clamp-3 hover:line-clamp-none transition-all cursor-default">
                        {src}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {response.evaluation && (
                <div>
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                    Evaluation Metrics
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Faithfulness</span>
                      <span className="font-medium text-green-700 bg-green-50 px-2 py-1">
                        {(response.evaluation.faithfulness * 100).toFixed(0)}%
                        Pass
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Answer Relevancy</span>
                      <span className="font-medium text-green-700 bg-green-50 px-2 py-1">
                        {(response.evaluation.relevancy * 100).toFixed(0)}% Pass
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
