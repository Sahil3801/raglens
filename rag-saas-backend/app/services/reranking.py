from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.core.config import settings

class RerankingService:
    """
    Service responsible for re-ordering retrieved document chunks based on their 
    semantic relevance to the user's query using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Initialize the CrossEncoder model (which evaluates query-document pairs simultaneously for higher accuracy)
        self.encoder = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = None) -> List[Document]:
        """
        Takes a query and a list of documents, scores their relevance, filters out 
        low-quality matches using a dynamic threshold, and returns the top results.
        """
        # Return an empty list early if no documents were provided
        if not documents:
            return []
            
        # Determine the maximum number of documents to return (fallback to settings if not explicitly passed)
        limit = settings.RERANKER_TOP_N if top_n is None else top_n
        
        # Step 1: Create pairs of [query, document text] required by the cross-encoder model
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Step 2: Predict relevance scores for all pairs simultaneously
        scores = self.encoder.predict(pairs)
        
        # Step 3: Zip documents with their respective scores and sort them in descending order (highest score first)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # --- THE DYNAMIC THRESHOLD ---
        # Calculate a dynamic cutoff score based on the highest-scoring document found
        best_score = scored_docs[0][1]
        tolerance = 8.0  # Allow documents scoring within 8 points of the best score
        dynamic_threshold = best_score - tolerance
        
        # Filter documents to keep only those meeting or exceeding the dynamic threshold
        filtered_docs = [doc for doc, score in scored_docs if score >= dynamic_threshold]
        
        # --- THE FIX: The Minimum Context Safety Net ---
        # If the threshold is too aggressive and starves the LLM, 
        # guarantee it gets at least the top 4 chunks (if they exist).
        MIN_CHUNKS = 4
        if len(filtered_docs) < MIN_CHUNKS and len(scored_docs) >= MIN_CHUNKS:
            # Fall back to taking the top MIN_CHUNKS if the filter was too strict
            filtered_docs = [doc for doc, score in scored_docs[:MIN_CHUNKS]]
        elif len(filtered_docs) < MIN_CHUNKS:
            # Just in case the entire document pool has fewer than MIN_CHUNKS total, keep all of them
            filtered_docs = [doc for doc, score in scored_docs]
            
        # Return the final filtered documents capped at the specified limit
        return filtered_docs[:limit]

# Instantiate a global singleton service object to be imported across the application API routes
reranker_service = RerankingService()