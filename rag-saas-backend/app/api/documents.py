from fastapi import APIRouter, UploadFile, File, Depends
from app.models.schemas import UploadResponse, DocumentListResponse, GenericResponse
from app.services.ingestion import IngestionService
from app.repositories.vector_store import VectorStoreRepository
from app.core.dependencies import get_ingestion_service, get_vector_store_repo

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion: IngestionService = Depends(get_ingestion_service)
):
    filename = ingestion.process_pdf_upload(file)
    return UploadResponse(
        filename=filename,
        message="Document successfully uploaded and indexed!"
    )

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    repo: VectorStoreRepository = Depends(get_vector_store_repo)
):
    files = repo.list_unique_source_files()
    return DocumentListResponse(documents=files)

@router.delete("/{filename}", response_model=GenericResponse)
async def delete_document(
    filename: str,
    repo: VectorStoreRepository = Depends(get_vector_store_repo)
):
    repo.delete_by_source_file(filename)
    return GenericResponse(message=f"Document {filename} deleted successfully.")