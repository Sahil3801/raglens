from typing import List, Optional
from langchain_core.documents import Document
from app.core.config import settings
from app.repositories.vector_store import VectorStoreRepository

class RetrievalService:
    def __init__(self, repo: VectorStoreRepository):
        self.repo = repo

    def retrieve(self, query: str, filter_filename: Optional[str] = None) -> List[Document]:
        return self.repo.search_mmr(
            query=query.strip(),
            k=settings.RETRIEVER_K,
            fetch_k=settings.RETRIEVER_FETCH_K,
            filter_filename=filter_filename
        )