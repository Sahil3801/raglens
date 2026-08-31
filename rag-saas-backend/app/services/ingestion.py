import os
import shutil
import tempfile
from fastapi import HTTPException, UploadFile
from pypdf.errors import PdfReadError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.repositories.vector_store import VectorStoreRepository

class IngestionService:
    """
    Service responsible for handling file uploads, extracting text from PDFs,
    splitting the text into smaller chunks, and storing them in the vector database.
    """
    def __init__(self, repo: VectorStoreRepository):
        # Initialize the vector store repository used for database operations
        self.repo = repo
        
        # Configure the text splitter using chunk size and overlap values from application settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def process_pdf_upload(self, file: UploadFile) -> str:
        """
        Takes an uploaded PDF file, saves it temporarily, extracts its text,
        splits it into chunks, saves it to the vector store, and cleans up.
        """
        # Unique server-owned paths prevent same-name uploads from overwriting
        # each other or files already present in the working directory.
        descriptor, temp_file_path = tempfile.mkstemp(prefix="raglens-upload-", suffix=".pdf")
        try:
            # Step 1: Save the uploaded file to disk temporarily so the loader can read it
            with os.fdopen(descriptor, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Step 2: Load the PDF document and extract pages into LangChain Document objects
            loader = PyPDFLoader(temp_file_path)
            try:
                documents = loader.load()
            except PdfReadError as exc:
                raise HTTPException(status_code=400, detail="The uploaded PDF could not be read.") from exc

            # Step 3: Enrich document metadata to track the original source file name
            for doc in documents:
                doc.metadata["source_file"] = file.filename

            # Step 4: Split the loaded documents into smaller, manageable chunks for embedding
            chunks = self.splitter.split_documents(documents)
            if not any(chunk.page_content.strip() for chunk in chunks):
                raise HTTPException(status_code=400, detail="The PDF contains no extractable text. OCR is not supported.")
            
            # Step 5: Embed and add the chunks to the vector store repository
            self.repo.add_documents(chunks)
            
            # Return the filename upon successful processing
            return file.filename
            
        finally:
            # Step 6: Ensure cleanup happens (temporary file is deleted) even if an error occurs
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
