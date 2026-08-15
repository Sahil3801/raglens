from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or prompt")
    filter_filename: Optional[str] = Field(None, description="Restrict search to this specific file")

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class UploadResponse(BaseModel):
    filename: str
    message: str

class DocumentListResponse(BaseModel):
    documents: List[str]

class GenericResponse(BaseModel):
    message: str