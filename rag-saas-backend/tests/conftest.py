"""Default tests isolate external models but exercise real parsing and Qdrant.

Do not persist mock scores into evaluation/runs. Real model/live checks are opt-in.
"""
import os
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

# Set before app import; no production .env is changed or required.
os.environ.update(GROQ_API_KEY='test-only-not-a-real-key', GROQ_MODEL='test-generator',
                  QDRANT_PATH='test-only-must-be-overridden', QDRANT_COLLECTION='test_documents',
                  CHUNK_SIZE='800', CHUNK_OVERLAP='200', RETRIEVER_K='40', RETRIEVER_FETCH_K='60',
                  RERANKER_TOP_N='8', EMBEDDING_DIMENSION='384', RAGAS_DO_NOT_TRACK='true',
                  HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

# Only prevent import-time singleton model construction. Test bodies patch explicit
# boundaries, not whole app modules, and restore the real constructors below.
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
with patch('sentence_transformers.CrossEncoder'), patch('langchain_groq.ChatGroq'):
    from app.main import app
    from app.services import reranking, generation
    reranking.reranker_service.encoder  # Lazily created while CrossEncoder is still patched.
reranking.CrossEncoder = CrossEncoder
generation.ChatGroq = ChatGroq


class FixedEmbeddings(Embeddings):
    """Controlled vectors for behavior tests, NOT semantic quality measurements."""
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name')

    def embed_query(self, text):
        vector = np.zeros(384)
        if text == 'near':
            vector[:2] = [0.98, 0.2]
        elif text == 'duplicate':
            vector[:2] = [0.979, 0.201]
        elif text == 'diverse':
            vector[:2] = [0.8, -0.6]
        elif 'Borealis' in text:
            vector[1] = 1
        else:
            vector[0] = 1
        return vector.tolist()

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def pytest_addoption(parser):
    parser.addoption('--run-models', action='store_true', help='Use real, already cached HF models; no downloads.')
    parser.addoption('--run-live', action='store_true', help='Call real Groq; requires TEST_GROQ_API_KEY and quota.')


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, option in [('models', '--run-models'), ('live', '--run-live')]:
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f'Opt-in: pass {option}'))


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path, request):
    from app.core.config import settings
    monkeypatch.setattr(settings, 'QDRANT_PATH', str(tmp_path / 'qdrant'))
    monkeypatch.setattr(settings, 'QDRANT_COLLECTION', 'test_documents')
    monkeypatch.chdir(tmp_path)
    app.dependency_overrides.clear()
    if 'live' not in request.keywords:
        def no_network(*args, **kwargs):
            raise AssertionError('Unexpected network request in an isolated test')
        original_async_send = httpx.AsyncClient.send
        original_sync_send = httpx.Client.send

        def guarded_sync_send(client, *args, **kwargs):
            if type(client._transport).__module__ == 'starlette.testclient':
                return original_sync_send(client, *args, **kwargs)
            return no_network()

        async def guarded_send(client, *args, **kwargs):
            if isinstance(client._transport, httpx.ASGITransport):
                return await original_async_send(client, *args, **kwargs)
            return no_network()

        monkeypatch.setattr(httpx.Client, 'send', guarded_sync_send)
        monkeypatch.setattr(httpx.AsyncClient, 'send', guarded_send)
        import requests
        monkeypatch.setattr(requests.Session, 'request', no_network)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def repo(monkeypatch):
    from app.repositories import vector_store
    monkeypatch.setattr(vector_store, 'HuggingFaceEmbeddings', FixedEmbeddings)
    repository = vector_store.VectorStoreRepository()
    monkeypatch.setattr(vector_store, '_vector_store_repo', repository)
    yield repository
    repository.client.close()


@pytest.fixture
def client(repo, monkeypatch):
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from app.core.dependencies import get_vector_store_repo, get_generation_service
    from app.services.generation import GenerationService

    # Keep production prompt formatting and parsing; substitute only provider IO.
    def fake_completion(prompt):
        return AIMessage(content='I cannot answer this based on the provided document.')
    with patch('app.services.generation.ChatGroq', return_value=RunnableLambda(fake_completion)):
        generator = GenerationService()
    monkeypatch.setattr(reranking.reranker_service.encoder, 'predict',
                        lambda pairs: np.arange(len(pairs), dtype=float))
    app.dependency_overrides[get_vector_store_repo] = lambda: repo
    app.dependency_overrides[get_generation_service] = lambda: generator
    # TestClient.send is intentionally restored only for its in-process transport.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def corpus_root():
    return Path(__file__).resolve().parents[1] / 'evaluation' / 'corpora' / 'synthetic_stations_v1'
