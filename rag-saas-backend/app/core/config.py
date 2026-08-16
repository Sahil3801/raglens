import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Evaluation-First RAG API"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "my_documents"
    
    # Retrieval Configuration
    RETRIEVER_K: int = 40
    RETRIEVER_FETCH_K: int = 60

    # Reranker Configuration
    RERANKER_TOP_N: int = 8

    # Ingestion Defaults
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()