from fastapi import HTTPException
from app.core.config import settings
from app.repositories.vector_store import VectorStoreRepository, get_vector_store_repo
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService, get_shared_generation_service
from app.services.ingestion import IngestionService


def get_retrieval_service() -> RetrievalService:
    return RetrievalService(repo=get_vector_store_repo())


def get_generation_service() -> GenerationService:
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")
    return get_shared_generation_service()


def get_ingestion_service() -> IngestionService:
    return IngestionService(repo=get_vector_store_repo())