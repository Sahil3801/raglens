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
    docs = retriever.retrieve(request.query, filter_filename=request.filter_filename)
    
    # 2. Rerank Phase
    reranked_docs = reranker_service.rerank(request.query, docs)
    
    # --- PIPELINE DEBUG PRINTS ---
    print("\n" + "="*50)
    print(f"QUERY: {request.query}")
    print(f"FILTER APPLIED: {request.filter_filename}")
    print(f"RETRIEVED: {len(docs)} chunks from Vector Store")
    print(f"AFTER RERANKING: {len(reranked_docs)} chunks sent to LLM")
    print("-" * 50)
    for i, doc in enumerate(reranked_docs):
        source = doc.metadata.get("source_file", "Unknown")
        print(f"  [{i+1}] Source: {source}")
        print(f"  Content: {doc.page_content}\n")
    print("="*50 + "\n")
    # -----------------------------

    # 3. Generation Phase
    answer = generator.generate_answer(request.query, reranked_docs)

    return ChatResponse(
        answer=answer,
        sources=[doc.page_content for doc in reranked_docs]
    )