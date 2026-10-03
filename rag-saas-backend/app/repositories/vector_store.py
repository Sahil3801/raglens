from typing import List, Set
from contextlib import nullcontext
from threading import Lock, RLock
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings

class VectorStoreRepository:
    def __init__(self):
        # 1. Initialize the Qdrant connection first
        # Endpoints run in FastAPI's threadpool. Embedded (local) Qdrant is not
        # safe for concurrent use, so its operations are serialized.
        self._lock = RLock() if settings.QDRANT_PATH else nullcontext()
        if settings.QDRANT_PATH:
            self.client = QdrantClient(
                path=settings.QDRANT_PATH,
                force_disable_check_same_thread=True,
            )
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=60.0,
            )

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self._ensure_collection_exists()
        
        # 3. Connect them both to the LangChain VectorStore
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self.embeddings,
        )

    def _ensure_collection_exists(self) -> None:
        if not self.client.collection_exists(settings.QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION, 
                    distance=Distance.COSINE
                ),
            )
            # Create the required keyword index for fast filtering and deletion
            # Embedded Qdrant supports filtering but does not use payload indexes.
            if not settings.QDRANT_PATH:
                self.client.create_payload_index(
                    collection_name=settings.QDRANT_COLLECTION,
                    field_name="metadata.source_file",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def add_documents(self, documents: List[Document]) -> None:
        if documents:
            with self._lock:
                self.store.add_documents(documents)

    def search_mmr(self, query: str, k: int = 8, fetch_k: int = 20, filter_filename: str = None) -> List[Document]:
        # Create a Qdrant filter if a filename is provided
        search_kwargs = {"k": k, "fetch_k": fetch_k}
        
        if filter_filename:
            search_kwargs["filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source_file",
                        match=models.MatchValue(value=filter_filename)
                    )
                ]
            )

        retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        with self._lock:
            return retriever.invoke(query)

    def list_unique_source_files(self) -> List[str]:
        files: Set[str] = set()
        offset = None
        while True:
            with self._lock:
                records, offset = self.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION,
                    with_payload=["metadata"],
                    limit=1000,
                    offset=offset,
                )
            for record in records:
                if (
                    record.payload
                    and "metadata" in record.payload
                    and "source_file" in record.payload["metadata"]
                ):
                    files.add(record.payload["metadata"]["source_file"])
            if offset is None:
                break
        return sorted(list(files))

    def delete_by_source_file(self, filename: str) -> None:
        with self._lock:
            self.client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source_file",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )

# vector_store_repo = VectorStoreRepository()

_vector_store_repo = None
_vector_store_lock = Lock()

def get_vector_store_repo() -> VectorStoreRepository:
    global _vector_store_repo
    if _vector_store_repo is None:
        with _vector_store_lock:
            if _vector_store_repo is None:
                _vector_store_repo = VectorStoreRepository()
    return _vector_store_repo
