import logging
from time import perf_counter

from fastapi import APIRouter, Depends
from app.models.schemas import (ChatRequest, ChatResponse, Citation, Lens, LensCandidate, PipelineStats,
                                PipelineTimings)
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService
from app.services.reranking import MIN_CHUNKS, reranker_service
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
    rerank_result = reranker_service.rerank_trace(
        request.query,
        docs
    )
    reranked = rerank_result.selected
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
        lens=_lens(docs, rerank_result),
    )


PREVIEW_CHARS = 220


def _lens(docs, result) -> Lens:
    selected_ids = {id(doc) for doc, _ in result.selected}
    by_score = sorted(range(len(docs)), key=lambda i: result.scores[i], reverse=True)
    candidates = []
    for rerank_rank, i in enumerate(by_score, start=1):
        doc, page = docs[i], docs[i].metadata.get("page")
        text = " ".join(doc.page_content.split())
        candidates.append(LensCandidate(
            source_file=doc.metadata.get("source_file", "Unknown"),
            page=page + 1 if isinstance(page, int) else None,
            preview=text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else ""),
            mmr_rank=i + 1,
            rerank_rank=rerank_rank,
            score=round(result.scores[i], 3),
            selected=id(doc) in selected_ids,
        ))
    return Lens(
        candidates=candidates,
        threshold=round(result.threshold, 3) if result.threshold is not None else None,
        min_chunks=MIN_CHUNKS,
        top_n=result.top_n,
        rule=result.rule,
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