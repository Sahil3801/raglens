from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from app.main import app
from app.api import chat
from app.core.dependencies import get_generation_service, get_retrieval_service

REFUSAL = 'I cannot answer this based on the provided document.'


def test_chat_endpoint_contract(client):
    response = client.post('/chat', json={'query': 'Unsupported question'})
    assert response.status_code == 200
    assert response.json() == {'answer': REFUSAL, 'sources': [], 'citations': []}


@pytest.mark.parametrize('filename', [None, 'A & notes.pdf'])
def test_only_reranked_context_reaches_generation_in_order(client, monkeypatch, filename):
    candidates = [Document(page_content=f'candidate {i}', metadata={'source_file': 'A & notes.pdf'}) for i in range(40)]
    selected = [candidates[31], candidates[8], candidates[19], candidates[2]]
    events = []
    retriever, generator = MagicMock(), MagicMock()
    retriever.retrieve.side_effect = lambda *args, **kwargs: events.append('retrieve') or candidates
    rerank = MagicMock(side_effect=lambda *args: events.append('rerank') or selected)
    generator.generate_answer.side_effect = lambda *args: events.append('generate') or 'A grounded answer.'
    app.dependency_overrides[get_retrieval_service] = lambda: retriever
    app.dependency_overrides[get_generation_service] = lambda: generator
    monkeypatch.setattr(chat.reranker_service, 'rerank', rerank)
    request = {'query': 'question'}
    if filename:
        request['filter_filename'] = filename
    response = client.post('/chat', json=request)
    assert response.status_code == 200
    assert events == ['retrieve', 'rerank', 'generate']
    retriever.retrieve.assert_called_once_with('question', filter_filename=filename)
    rerank.assert_called_once_with('question', candidates)
    generator.generate_answer.assert_called_once_with('question', selected)
    assert response.json() == {
        'answer': 'A grounded answer.',
        'sources': [d.page_content for d in selected],
        'citations': [{'source_file': 'A & notes.pdf', 'page': None, 'text': d.page_content} for d in selected],
    }


@pytest.mark.parametrize('payload', [{}, {'query': None}, {'query': 123}, {'query': []}])
def test_invalid_payload_is_422_without_retrieval(client, payload):
    retriever = MagicMock()
    app.dependency_overrides[get_retrieval_service] = lambda: retriever
    assert client.post('/chat', json=payload).status_code == 422
    retriever.retrieve.assert_not_called()


def test_irrelevant_question_refusal_is_not_rewritten(client, repo):
    repo.add_documents([Document(page_content='Archive boxes have paper labels.', metadata={'source_file': 'archive.pdf'})])
    response = client.post('/chat', json={'query': 'What is the weather on Mars?'})
    assert response.json()['answer'] == REFUSAL
    assert response.json()['sources'] == ['Archive boxes have paper labels.']
    # Controlled provider response; not proof that a real LLM obeys the prompt.


def test_blocking_endpoints_run_off_the_event_loop():
    # Sync handlers run in FastAPI's threadpool, so model/LLM calls cannot block the loop.
    import inspect
    from app.api import documents
    from app import main
    for handler in [chat.chat_endpoint, documents.upload_document, documents.list_documents,
                    documents.delete_document, main.root_upload_compat]:
        assert not inspect.iscoroutinefunction(handler), handler.__name__


def test_citations_carry_file_and_one_based_page(client, monkeypatch):
    docs = [Document(page_content='Pump spec', metadata={'source_file': 'manual.pdf', 'page': 0}),
            Document(page_content='Valve spec', metadata={'source_file': 'manual.pdf', 'page': 4}),
            Document(page_content='Loose text', metadata={})]
    retriever = MagicMock()
    retriever.retrieve.return_value = docs
    app.dependency_overrides[get_retrieval_service] = lambda: retriever
    monkeypatch.setattr(chat.reranker_service, 'rerank', lambda query, candidates: candidates)
    response = client.post('/chat', json={'query': 'specs'})
    assert response.json()['citations'] == [
        {'source_file': 'manual.pdf', 'page': 1, 'text': 'Pump spec'},
        {'source_file': 'manual.pdf', 'page': 5, 'text': 'Valve spec'},
        {'source_file': 'Unknown', 'page': None, 'text': 'Loose text'},
    ]


def test_chat_without_groq_key_is_a_clear_503(client, monkeypatch):
    from app.core.config import settings
    app.dependency_overrides.pop(get_generation_service)
    retriever = MagicMock()
    app.dependency_overrides[get_retrieval_service] = lambda: retriever
    monkeypatch.setattr(settings, 'GROQ_API_KEY', '')
    response = client.post('/chat', json={'query': 'question'})
    assert response.status_code == 503
    assert response.json()['detail'] == 'GROQ_API_KEY is not configured on the server.'


def test_app_imports_without_groq_key_or_models():
    # Regression: the server must start (health, upload, list, delete) with no Groq key.
    import os, subprocess, sys
    from pathlib import Path
    env = {**os.environ, 'GROQ_API_KEY': '', 'HF_HUB_OFFLINE': '1'}
    code = 'import app.main; print("ok")'
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'
