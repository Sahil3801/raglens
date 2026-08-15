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
            
        # SAFER CHECK: Treats 0 as a valid limit instead of falsy
        limit = settings.RERANKER_TOP_N if top_n is None else top_n
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.encoder.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:limit]]

reranker_service = RerankingService()