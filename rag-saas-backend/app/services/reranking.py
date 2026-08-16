from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.core.config import settings

class RerankingService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.encoder = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = None) -> List[Document]:
        if not documents:
            return []
            
        limit = settings.RERANKER_TOP_N if top_n is None else top_n
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.encoder.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # --- THE DYNAMIC THRESHOLD ---
        best_score = scored_docs[0][1]
        tolerance = 8.0 
        dynamic_threshold = best_score - tolerance
        
        filtered_docs = [doc for doc, score in scored_docs if score >= dynamic_threshold]
        
        # --- THE FIX: The Minimum Context Safety Net ---
        # If the threshold is too aggressive and starves the LLM, 
        # guarantee it gets at least the top 4 chunks (if they exist).
        MIN_CHUNKS = 4
        if len(filtered_docs) < MIN_CHUNKS and len(scored_docs) >= MIN_CHUNKS:
            filtered_docs = [doc for doc, score in scored_docs[:MIN_CHUNKS]]
        elif len(filtered_docs) < MIN_CHUNKS:
             # Just in case the document itself has fewer than 4 chunks total
            filtered_docs = [doc for doc, score in scored_docs]
            
        return filtered_docs[:limit]

reranker_service = RerankingService()