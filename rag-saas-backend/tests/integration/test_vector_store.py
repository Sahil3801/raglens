from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from app.core.config import settings
from app.services.retrieval import RetrievalService


def doc(text, source='a.pdf', page=0):
    return Document(page_content=text, metadata={'source_file': source, 'page': page})


def test_real_qdrant_roundtrip_filter_global_and_delete(repo):
    repo.add_documents([doc('Aurora code', 'A & #.pdf', 2), doc('Borealis code', 'b.pdf', 3)])
    assert repo.client.get_collection(settings.QDRANT_COLLECTION).config.params.vectors.size == 384
    filtered = RetrievalService(repo).retrieve('code', 'b.pdf')
    assert len(filtered) == 1 and filtered[0].page_content == 'Borealis code'
    assert filtered[0].metadata['page'] == 3 and filtered[0].metadata['source_file'] == 'b.pdf'
    assert {d.metadata['source_file'] for d in RetrievalService(repo).retrieve('code')} == {'A & #.pdf', 'b.pdf'}
    assert RetrievalService(repo).retrieve('code', 'missing.pdf') == []
    repo.delete_by_source_file('A & #.pdf')
    assert repo.list_unique_source_files() == ['b.pdf']
    assert [d.page_content for d in RetrievalService(repo).retrieve('code')] == ['Borealis code']
    repo.delete_by_source_file('missing.pdf')
    assert repo.list_unique_source_files() == ['b.pdf']


def test_mmr_selects_diverse_chunk_instead_of_near_duplicate(repo):
    repo.add_documents([doc('near'), doc('duplicate'), doc('diverse')])
    nearest = repo.store.similarity_search('query', k=3)
    assert [d.page_content for d in nearest] == ['near', 'duplicate', 'diverse']
    selected = repo.search_mmr('query', k=2, fetch_k=3)
    assert [d.page_content for d in selected] == ['near', 'diverse']


def test_real_mmr_candidate_and_return_limits_are_60_and_40(repo, monkeypatch):
    repo.add_documents([doc(f'chunk {i}') for i in range(70)])
    collection = repo.client._client.collections[settings.QDRANT_COLLECTION]
    original = collection.search
    observed = []
    def search(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append((kwargs['limit'], len(result)))
        return result
    monkeypatch.setattr(collection, 'search', search)
    assert len(RetrievalService(repo).retrieve('query')) == 40
    assert observed == [(60, 60)]


def test_document_listing_reads_all_scroll_pages(repo):
    names = [f'file-{i:04d}.pdf' for i in range(1003)]
    repo.add_documents([doc('content', name) for name in names])
    assert repo.list_unique_source_files() == names


def test_listing_deduplicates_and_skips_missing_metadata(repo):
    repo.add_documents([doc('one', 'z.pdf'), doc('two', 'z.pdf'), doc('three', 'a.pdf'),
                        Document(page_content='no source')])
    assert repo.list_unique_source_files() == ['a.pdf', 'z.pdf']


def test_empty_add_does_not_call_embedding_or_store(repo, monkeypatch):
    add = MagicMock()
    monkeypatch.setattr(repo.store, 'add_documents', add)
    repo.add_documents([])
    add.assert_not_called()


def test_singleton_is_safe_under_concurrent_initialization(monkeypatch):
    from app.repositories import vector_store
    expected = object()
    factory = MagicMock(return_value=expected)
    monkeypatch.setattr(vector_store, '_vector_store_repo', None)
    monkeypatch.setattr(vector_store, 'VectorStoreRepository', factory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(lambda _: vector_store.get_vector_store_repo(), range(32)))
    assert all(item is expected for item in result)
    factory.assert_called_once()
