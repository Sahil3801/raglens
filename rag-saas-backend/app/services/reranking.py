import re
from pathlib import PurePath
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.core.config import settings

class RerankingService:
    """
    Service responsible for re-ordering retrieved document chunks based on their 
    semantic relevance to the user's query using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # The CrossEncoder is loaded on first use, so importing the app (and starting
        # the server) does not download or load the model.
        self.model_name = model_name
        self._encoder = None
        self._encoder_lock = Lock()

    @property
    def encoder(self) -> CrossEncoder:
        if self._encoder is None:
            with self._encoder_lock:
                if self._encoder is None:
                    self._encoder = CrossEncoder(self.model_name)
        return self._encoder

    @encoder.setter
    def encoder(self, value: CrossEncoder) -> None:
        self._encoder = value

    def rerank(self, query: str, documents: List[Document], top_n: int = None,
               source_headers: bool = None) -> List[Document]:
        """Reranked documents without their scores (see rerank_with_scores)."""
        return [doc for doc, _ in self.rerank_with_scores(query, documents, top_n, source_headers)]

    def rerank_with_scores(self, query: str, documents: List[Document], top_n: int = None,
                           source_headers: bool = None) -> List[Tuple[Document, float]]:
        """Selected documents with their CrossEncoder scores, highest first."""
        return self.rerank_trace(query, documents, top_n, source_headers).selected

    def rerank_trace(self, query: str, documents: List[Document], top_n: int = None,
                     source_headers: bool = None) -> "RerankResult":
        """
        Takes a query and a list of documents, scores their relevance, filters out 
        low-quality matches using a dynamic threshold, and returns the top results
        together with every candidate's score and the rule that set the cut-off
        (used by the UI's Lens view).
        """
        # Determine the maximum number of documents to return (fallback to settings if not explicitly passed)
        limit = settings.RERANKER_TOP_N if top_n is None else top_n

        # Return an empty result early if no documents were provided
        if not documents:
            return RerankResult(selected=[], scores=[], threshold=None, rule="empty", top_n=limit)
        
        # Step 1: Create pairs of [query, document text] required by the cross-encoder model
        if source_headers is None:
            source_headers = settings.RERANKER_SOURCE_HEADERS
        pairs = [[query, _with_source_header(doc) if source_headers else doc.page_content] for doc in documents]
        
        # Step 2: Predict relevance scores for all pairs simultaneously
        scores = [float(score) for score in self.encoder.predict(pairs)]
        
        # Step 3: Zip documents with their respective scores and sort them in descending order (highest score first)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # --- THE DYNAMIC THRESHOLD ---
        # Calculate a dynamic cutoff score based on the highest-scoring document found
        best_score = scored_docs[0][1]
        tolerance = 8.0  # Allow documents scoring within 8 points of the best score
        dynamic_threshold = best_score - tolerance
        
        # Filter documents to keep only those meeting or exceeding the dynamic threshold
        filtered_docs = [(doc, score) for doc, score in scored_docs if score >= dynamic_threshold]
        rule = "threshold"
        
        # --- THE FIX: The Minimum Context Safety Net ---
        # If the threshold is too aggressive and starves the LLM, 
        # guarantee it gets at least the top 4 chunks (if they exist).
        if len(filtered_docs) < MIN_CHUNKS and len(scored_docs) >= MIN_CHUNKS:
            # Fall back to taking the top MIN_CHUNKS if the filter was too strict
            filtered_docs = scored_docs[:MIN_CHUNKS]
            rule = "min_chunks"
        elif len(filtered_docs) < MIN_CHUNKS:
            # Just in case the entire document pool has fewer than MIN_CHUNKS total, keep all of them
            filtered_docs = scored_docs
            rule = "all"
        if len(filtered_docs) > limit:
            rule = "top_n"
            
        # Return the final filtered documents capped at the specified limit
        return RerankResult(selected=filtered_docs[:limit], scores=scores,
                            threshold=dynamic_threshold, rule=rule, top_n=limit)

MIN_CHUNKS = 4


@dataclass
class RerankResult:
    selected: List[Tuple[Document, float]]  # documents sent to the LLM, best first
    scores: List[float]                      # every candidate's score, in input order
    threshold: Optional[float]               # best score - 8 (dynamic cut-off)
    rule: str  # what set the cut-off: threshold, min_chunks, all, top_n or empty
    top_n: int


def source_label(filename: str) -> str:
    """'omarPaypalResume.pdf' -> 'omar paypal resume': words the CrossEncoder can match."""
    stem = PurePath(filename).stem
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", stem)
    words = re.sub(r"([A-Za-z])([0-9])|([0-9])([A-Za-z])", lambda m: " ".join(g for g in m.groups() if g), words)
    return " ".join(re.split(r"[\s_\-.()\[\]]+", words)).strip().lower()


def _with_source_header(doc: Document) -> str:
    source = doc.metadata.get("source_file")
    return f"{source_label(source)}: {doc.page_content}" if source else doc.page_content


# Global singleton shared by the API routes; cheap to create because the model loads lazily
reranker_service = RerankingService()