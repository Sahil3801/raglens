"""Optional checks with genuine models. No benchmark metrics are generated."""
import os

import numpy as np
import pytest
from langchain_core.documents import Document


@pytest.mark.models
def test_cached_huggingface_embedding_is_really_384_dimensions():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu', 'local_files_only': True,
                      'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'})
    query = np.array(embeddings.embed_query('What is the maintenance access code?'))
    documents = np.array(embeddings.embed_documents(['The access code is AMBER-731.', 'A telescope observes stars.']))
    assert query.shape == (384,) and documents.shape == (2, 384)
    assert np.isfinite(query).all() and np.isfinite(documents).all()
    assert np.linalg.norm(query) > 0 and not np.allclose(documents[0], documents[1])


@pytest.mark.models
def test_cached_crossencoder_promotes_relevant_passage_with_real_scores(monkeypatch):
    from sentence_transformers import CrossEncoder
    from app.services import reranking
    encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', local_files_only=True,
                           revision='233902d25c440f23af6f7d6e94d2946bac0bee0a')
    monkeypatch.setattr(reranking.reranker_service, 'encoder', encoder)
    docs = [Document(page_content=text, metadata={'source_file': 'model-test.pdf'}) for text in [
        'Bananas are yellow fruit.', 'A telescope observes distant stars.',
        'The capital of France is Paris.']]
    selected = reranking.reranker_service.rerank('What is the capital of France?', docs, top_n=1)
    assert selected == [docs[2]]


@pytest.mark.live
@pytest.mark.parametrize('documents', [[], [Document(page_content='Aurora has a blue door.')]], ids=['empty-context', 'irrelevant-context'])
def test_live_groq_refuses_unsupported_question(monkeypatch, documents):
    """Unlike prompt-contract tests, this checks actual hosted-model compliance."""
    key = os.getenv('TEST_GROQ_API_KEY')
    if not key:
        pytest.skip('TEST_GROQ_API_KEY is required; this test never reads the production .env key')
    from app.core.config import settings
    from app.services.generation import GenerationService
    monkeypatch.setattr(settings, 'GROQ_API_KEY', key)
    monkeypatch.setattr(settings, 'GROQ_MODEL', os.getenv('TEST_GROQ_MODEL', 'openai/gpt-oss-20b'))
    service = GenerationService()
    service.llm.request_timeout = 30
    service.llm.max_retries = 0
    response = service.generate_answer('What is the annual operating budget of Aurora station?', documents)
    assert response.strip() == 'I cannot answer this based on the provided document.'


@pytest.mark.models
def test_real_pipeline_ranks_the_answering_page_first_with_ordered_scores(monkeypatch):
    """Upload -> real embeddings -> embedded Qdrant -> MMR -> real CrossEncoder -> /chat.

    Only the LLM is replaced, so this checks retrieval, reranking scores and the
    context order without a Groq key. It is a behavior check, not a benchmark.
    """
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from sentence_transformers import CrossEncoder
    from app.core.dependencies import get_generation_service
    from app.main import app
    from app.repositories import vector_store
    from app.services import generation, reranking
    from tests.pdf_factory import pdf_bytes

    # Cached models only: the suite blocks network access.
    monkeypatch.setattr(reranking.reranker_service, 'encoder',
                        CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', local_files_only=True))
    real_embeddings = vector_store.HuggingFaceEmbeddings
    monkeypatch.setattr(vector_store, 'HuggingFaceEmbeddings', lambda model_name: real_embeddings(
        model_name=model_name, model_kwargs={'device': 'cpu', 'local_files_only': True}))
    repo = vector_store.VectorStoreRepository()  # real all-MiniLM-L6-v2 embeddings
    monkeypatch.setattr(vector_store, '_vector_store_repo', repo)
    contexts = []
    def fake_llm(prompt):
        contexts.append(prompt.to_messages()[0].content)
        return AIMessage(content='stub answer')
    monkeypatch.setattr(generation, 'ChatGroq', lambda **kwargs: RunnableLambda(fake_llm))
    app.dependency_overrides[get_generation_service] = generation.GenerationService

    resumes = {
        'omar_resume.pdf': [
            'Omar Badr. Software Engineer. San Diego, California. omar@example.com',
            'Experience: Software Engineering Intern at PayPal. Built payment APIs in Java and Spring Boot.',
            'Education: San Diego State University. Bachelor of Science in Computer Science, expected May 2025.',
        ],
        'sahil_portfolio.pdf': [
            'Sahil Shinde. Portfolio. Projects: a RAG application with FastAPI, Qdrant and React.',
            'Skills: Python, FastAPI, PostgreSQL, Docker, React and machine learning.',
        ],
    }
    try:
        with TestClient(app) as client:
            for name, pages in resumes.items():
                assert client.post('/upload', files={'file': (name, pdf_bytes(pages), 'application/pdf')}).status_code == 200
            body = client.post('/chat', json={'query': "what is omar's education"}).json()
    finally:
        repo.client.close()

    citations = body['citations']
    scores = [c['score'] for c in citations]
    print('real reranker ranking:', [(c['source_file'], c['page'], c['score']) for c in citations])
    assert scores == sorted(scores, reverse=True)  # shown best first
    # Omar's resume outranks the unrelated portfolio, and the page with the answer
    # reaches the LLM. (The real CrossEncoder ranks the page that repeats the name
    # "Omar" above the Education page, which never mentions him.)
    assert [c['source_file'] for c in citations[:2]] == ['omar_resume.pdf', 'omar_resume.pdf']
    assert ('omar_resume.pdf', 3) in [(c['source_file'], c['page']) for c in citations]
    assert body['pipeline']['retrieved'] == 5 and body['pipeline']['reranked'] == len(citations) >= 4
    files_in_context = [part.split(' ---', 1)[0] for part in contexts[0].split('--- CHUNK FROM ')[1:]]
    first_sahil = files_in_context.index('sahil_portfolio.pdf')
    assert set(files_in_context[first_sahil:]) == {'sahil_portfolio.pdf'}  # Omar's chunks grouped first
