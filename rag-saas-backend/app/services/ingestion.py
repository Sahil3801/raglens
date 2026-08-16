import os
import shutil
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.repositories.vector_store import VectorStoreRepository

class IngestionService:
    def __init__(self, repo: VectorStoreRepository):
        self.repo = repo
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def process_pdf_upload(self, file: UploadFile) -> str:
        temp_file_path = f"./temp_{file.filename}"
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = file.filename

            chunks = self.splitter.split_documents(documents)
            self.repo.add_documents(chunks)
            return file.filename
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)