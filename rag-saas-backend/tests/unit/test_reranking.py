from unittest.mock import MagicMock
import numpy as np
import pytest
from langchain_core.documents import Document
from app.services.reranking import RerankingService


def run(scores, top_n=None):
    service = RerankingService.__new__(RerankingService)
    service.encoder = MagicMock()
    service.encoder.predict.return_value = np.asarray(scores, dtype=np.float32)
    documents = [Document(page_content=f'chunk {i}', metadata={'source_file': f'{i}.pdf', 'page': i}) for i in range(len(scores))]
    original = [d.model_dump() for d in documents]
    result = service.rerank('question', documents, top_n=top_n)
    assert [d.model_dump() for d in documents] == original
    assert all(any(d is original_doc for original_doc in documents) for d in result)
    if documents:
        service.encoder.predict.assert_called_once_with([['question', d.page_content] for d in documents])
    else:
        service.encoder.predict.assert_not_called()
    return [d.metadata['page'] for d in result]


def test_high_scoring_tail_candidate_is_promoted_and_capped_at_eight():
    assert run(list(range(40))) == list(range(39, 31, -1))


def test_threshold_is_inclusive_and_minimum_floor_does_not_hide_boundary():
    assert run([10, 9, 8, 7, 2, 1.99, -10]) == [0, 1, 2, 3, 4]


def test_aggressive_threshold_retains_the_top_four_candidates():
    assert run([20, -30, -10, -20, -40]) == [0, 2, 3, 1]


@pytest.mark.parametrize('size', [0, 1, 2, 3])
def test_small_candidate_pools_never_duplicate_or_invent_chunks(size):
    assert run([i * 20 for i in range(size)]) == list(reversed(range(size)))


def test_ties_preserve_mmr_order():
    assert run([1] * 10) == list(range(8))


@pytest.mark.parametrize('limit', [0, 1, 2, 4, 8, 20])
def test_explicit_top_n_is_respected_even_below_safety_floor(limit):
    assert run([20, -10, -20, -30, -40], top_n=limit) == [0, 1, 2, 3][:limit]


def test_configured_crossencoder_model_is_used(monkeypatch):
    from app.services import reranking
    constructor = MagicMock()
    monkeypatch.setattr(reranking, 'CrossEncoder', constructor)
    RerankingService()
    constructor.assert_called_once_with('cross-encoder/ms-marco-MiniLM-L-6-v2')
