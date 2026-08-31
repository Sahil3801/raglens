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
