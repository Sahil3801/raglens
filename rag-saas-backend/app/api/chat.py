import logging
from time import perf_counter

from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse, Citation, PipelineStats, PipelineTimings
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
    started = perf_counter()
    docs = retriever.retrieve(
        request.query,
        filter_filename=request.filter_filename
    )
    retrieved_at = perf_counter()

    # 2. Rerank Phase
    reranked = reranker_service.rerank_with_scores(
        request.query,
        docs
    )
    reranked_docs = [doc for doc, _ in reranked]
    reranked_at = perf_counter()

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
    generated_at = perf_counter()

    return ChatResponse(
        answer=answer,
        sources=[doc.page_content for doc in reranked_docs],
        citations=[_citation(doc, score) for doc, score in reranked],
        pipeline=PipelineStats(
            retrieved=len(docs),
            reranked=len(reranked_docs),
            timings_ms=PipelineTimings(
                retrieval=_ms(retrieved_at - started),
                reranking=_ms(reranked_at - retrieved_at),
                generation=_ms(generated_at - reranked_at),
            ),
        ),
    )


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _citation(doc, score: float = None) -> Citation:
    # PyPDFLoader stores 0-based page indexes; users read 1-based page numbers.
    page = doc.metadata.get("page")
    return Citation(
        source_file=doc.metadata.get("source_file", "Unknown"),
        page=page + 1 if isinstance(page, int) else None,
        text=doc.page_content,
        score=round(score, 3) if score is not None else None,
    )