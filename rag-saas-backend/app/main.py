import logging

from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import chat, documents
from app.models.schemas import UploadResponse
from app.services.ingestion import IngestionService
from app.core.dependencies import get_ingestion_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
def health():
    # Liveness only: does not load models or touch Qdrant.
    return {"status": "ok"}

# Compatibility route for Vite frontends pointing directly to /upload
@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
def root_upload_compat(
    file: UploadFile = File(...),
    ingestion: IngestionService = Depends(get_ingestion_service)
):
    filename = ingestion.process_pdf_upload(file)
    return UploadResponse(
        filename=filename,
        message="Document successfully uploaded and indexed!"
    )

# Include Modular Routers
app.include_router(chat.router)
app.include_router(documents.router)