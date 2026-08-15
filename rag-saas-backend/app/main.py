from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import chat, documents
from app.models.schemas import UploadResponse
from app.services.ingestion import IngestionService
from app.core.dependencies import get_ingestion_service

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compatibility route for Vite frontends pointing directly to /upload
@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def root_upload_compat(
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