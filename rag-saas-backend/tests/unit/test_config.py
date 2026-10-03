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


def test_cors_origins_default_to_vite_dev_server_and_accept_a_comma_list(monkeypatch):
    monkeypatch.delenv('CORS_ORIGINS', raising=False)
    assert Settings(_env_file=None).cors_origin_list == ['http://localhost:5173', 'http://127.0.0.1:5173']
    monkeypatch.setenv('CORS_ORIGINS', 'https://raglens.example.com, http://localhost:8080 ,')
    assert Settings(_env_file=None).cors_origin_list == ['https://raglens.example.com', 'http://localhost:8080']
