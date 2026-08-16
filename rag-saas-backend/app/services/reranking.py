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
        # 1. Find the score of the absolute best chunk for this specific query
        best_score = scored_docs[0][1]
        
        # 2. Set a tolerance gap (e.g., drop anything that is 4+ points worse than the best)
        # You can tweak this number (3.0 to 5.0) based on your preference
        tolerance = 4.0 
        dynamic_threshold = best_score - tolerance
        
        # 3. Filter out the garbage that falls below the relative threshold
        filtered_docs = [doc for doc, score in scored_docs if score >= dynamic_threshold]
        
        return filtered_docs[:limit]

reranker_service = RerankingService()