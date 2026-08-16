from fastapi import APIRouter, Depends
from app.models.schemas import ChatRequest, ChatResponse
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService
from app.services.reranking import reranker_service
from app.core.dependencies import get_retrieval_service, get_generation_service

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
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

    # --- PIPELINE DEBUG PRINTS ---
    print("\n" + "=" * 50)
    print("RAG PIPELINE")
    print("=" * 50)
    print(f"QUERY: {request.query}")
    print(f"FILTER: {request.filter_filename or 'Global Search'}")
    print(f"RETRIEVED: {len(docs)} chunks")
    print(f"RERANKED: {len(reranked_docs)} chunks")
    print(f"LLM CONTEXT: {len(reranked_docs)} chunks")
    print("-" * 50)

    print("TOP SOURCES:")
    for i, doc in enumerate(reranked_docs[:3]):
        source = doc.metadata.get("source_file", "Unknown")
        print(f"  [{i+1}] {source}")

    print("=" * 50 + "\n")
    # -----------------------------

    # 3. Generation Phase
    answer = generator.generate_answer(
        request.query,
        reranked_docs
    )

    return ChatResponse(
        answer=answer,
        sources=[doc.page_content for doc in reranked_docs]
    )