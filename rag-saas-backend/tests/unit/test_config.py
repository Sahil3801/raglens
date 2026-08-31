from app.core.config import Settings


def test_defaults_and_environment_overrides_do_not_silently_change_retrieval(monkeypatch):
    fields = ['CHUNK_SIZE', 'CHUNK_OVERLAP', 'EMBEDDING_DIMENSION', 'RETRIEVER_K',
              'RETRIEVER_FETCH_K', 'RERANKER_TOP_N']
    for name in fields:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert [getattr(settings, name) for name in fields] == [800, 200, 384, 40, 60, 8]
    monkeypatch.setenv('RETRIEVER_K', '17')
    monkeypatch.setenv('RETRIEVER_FETCH_K', '29')
    overridden = Settings(_env_file=None)
    assert overridden.RETRIEVER_K == 17 and overridden.RETRIEVER_FETCH_K == 29
