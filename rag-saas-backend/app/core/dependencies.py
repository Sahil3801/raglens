from app.repositories.vector_store import VectorStoreRepository, vector_store_repo
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService

def get_vector_store_repo() -> VectorStoreRepository:
    return vector_store_repo

def get_retrieval_service() -> RetrievalService:
    return RetrievalService(repo=vector_store_repo)

def get_generation_service() -> GenerationService:
    return GenerationService()

def get_ingestion_service() -> IngestionService:
    return IngestionService(repo=vector_store_repo)