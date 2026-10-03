import logging

from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse, Citation
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService
from app.services.reranking import reranker_service
from app.core.dependencies import get_retrieval_service, get_generation_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    retriever: RetrievalService = Depends(get_retrieval_service),
    generator: GenerationService = Depends(get_generation_service)
):
    # 1. Retrieve Phase
    docs = retriever.retrieve(
        request.query,
        filter_filename=request.filter_filename
    )

    # 2. Rerank Phase
    reranked_docs = reranker_service.rerank(
        request.query,
        docs
    )

    logger.info(
        "RAG pipeline: filter=%s retrieved=%d reranked=%d top_sources=%s",
        request.filter_filename or "global",
        len(docs),
        len(reranked_docs),
        [doc.metadata.get("source_file", "Unknown") for doc in reranked_docs[:3]],
    )

    # 3. Generation Phase
    answer = generator.generate_answer(
        request.query,
        reranked_docs
    )

    return ChatResponse(
        answer=answer,
        sources=[doc.page_content for doc in reranked_docs],
        citations=[_citation(doc) for doc in reranked_docs],
    )


def _citation(doc) -> Citation:
    # PyPDFLoader stores 0-based page indexes; users read 1-based page numbers.
    page = doc.metadata.get("page")
    return Citation(
        source_file=doc.metadata.get("source_file", "Unknown"),
        page=page + 1 if isinstance(page, int) else None,
        text=doc.page_content,
    )