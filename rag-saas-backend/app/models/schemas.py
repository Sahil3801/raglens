from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or prompt")
    filter_filename: Optional[str] = Field(None, description="Restrict search to this specific file")

class Citation(BaseModel):
    source_file: str = Field(..., description="Uploaded document the chunk came from")
    page: Optional[int] = Field(None, description="1-based PDF page number, when known")
    text: str = Field(..., description="Chunk text passed to the LLM as context")

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    citations: List[Citation] = Field(default_factory=list, description="Context chunks in the order given to the LLM")

class UploadResponse(BaseModel):
    filename: str
    message: str

class DocumentListResponse(BaseModel):
    documents: List[str]

class GenericResponse(BaseModel):
    message: str