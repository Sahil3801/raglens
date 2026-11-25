from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or prompt")
    filter_filename: Optional[str] = Field(None, description="Restrict search to this specific file")

class Citation(BaseModel):
    source_file: str = Field(..., description="Uploaded document the chunk came from")
    page: Optional[int] = Field(None, description="1-based PDF page number, when known")
    text: str = Field(..., description="Chunk text passed to the LLM as context")
    score: Optional[float] = Field(None, description="CrossEncoder relevance score (higher is more relevant)")

class PipelineTimings(BaseModel):
    retrieval: float
    reranking: float
    generation: float

class PipelineStats(BaseModel):
    retrieved: int = Field(..., description="Chunks returned by MMR retrieval")
    reranked: int = Field(..., description="Chunks kept after CrossEncoder reranking and sent to the LLM")
    timings_ms: PipelineTimings

class LensCandidate(BaseModel):
    source_file: str
    page: Optional[int] = Field(None, description="1-based PDF page number, when known")
    preview: str = Field(..., description="Start of the chunk text")
    mmr_rank: int = Field(..., description="1-based position in the MMR retrieval results")
    rerank_rank: int = Field(..., description="1-based position after CrossEncoder scoring")
    score: float = Field(..., description="CrossEncoder relevance score")
    selected: bool = Field(..., description="True if the chunk was sent to the LLM")

class Lens(BaseModel):
    """Every retrieved chunk and how reranking decided which ones reach the LLM."""
    candidates: List[LensCandidate] = Field(..., description="All retrieved chunks, best reranker score first")
    threshold: Optional[float] = Field(None, description="Dynamic cut-off: best score minus 8")
    min_chunks: int
    top_n: int
    rule: str = Field(..., description="What set the cut-off: threshold, min_chunks, all, top_n or empty")

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    citations: List[Citation] = Field(default_factory=list, description="Context chunks in the order given to the LLM")
    pipeline: Optional[PipelineStats] = None
    lens: Optional[Lens] = None

class UploadResponse(BaseModel):
    filename: str
    message: str

class DocumentListResponse(BaseModel):
    documents: List[str]

class GenericResponse(BaseModel):
    message: str