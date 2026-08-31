from typing import List, Optional
from langchain_core.documents import Document
from app.core.config import settings
from app.repositories.vector_store import VectorStoreRepository

class RetrievalService:
    """
    Service responsible for fetching relevant document chunks from the vector store
    based on the user's query, utilizing Maximal Marginal Relevance (MMR) for diversity.
    """
    def __init__(self, repo: VectorStoreRepository):
        # Initialize the vector store repository used to execute search queries
        self.repo = repo

    def retrieve(self, query: str, filter_filename: Optional[str] = None) -> List[Document]:
        """
        Retrieves top document chunks matching the query using MMR search,
        with optional filtering by a specific source filename.
        """
        # Execute MMR (Maximal Marginal Relevance) search via the vector store repository.
        # MMR balances relevance to the query with diversity among the retrieved chunks to avoid redundancy.
        return self.repo.search_mmr(
            query=query.strip(),  # Clean up any leading/trailing whitespace from the query
            k=settings.RETRIEVER_K,  # The final number of chunks to return
            fetch_k=settings.RETRIEVER_FETCH_K,  # The initial pool size fetched for MMR diversity calculation
            filter_filename=filter_filename  # Optional filter to restrict search to a specific uploaded document
        )