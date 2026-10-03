from urllib.parse import quote
import pytest
from tests.pdf_factory import pdf_bytes


@pytest.mark.parametrize('upload_route', ['/upload', '/documents/upload'])
def test_upload_list_filter_delete_lifecycle(client, upload_route):
    assert client.get('/documents').json() == {'documents': []}
    names = ['Aurora & #1.pdf', 'Borealis.pdf']
    for name in names:
        response = client.post(upload_route, files={'file': (name, pdf_bytes([name + ' evidence']), 'application/pdf')})
        assert response.status_code == 200
        assert response.json()['filename'] == name
    assert client.get('/documents').json()['documents'] == names
    single = client.post('/chat', json={'query': 'evidence', 'filter_filename': names[0]}).json()
    assert single['sources'] == [names[0] + ' evidence']
    assert single['citations'] == [{'source_file': names[0], 'page': 1, 'text': names[0] + ' evidence'}]
    global_result = client.post('/chat', json={'query': 'evidence'}).json()
    assert set(global_result['sources']) == {name + ' evidence' for name in names}
    response = client.delete('/documents/' + quote(names[0], safe=''))
    assert response.status_code == 200
    assert names[0] in response.json()['message']
    assert client.get('/documents').json()['documents'] == [names[1]]
    assert client.post('/chat', json={'query': 'evidence', 'filter_filename': names[0]}).json()['sources'] == []
    assert client.delete('/documents/' + quote(names[0], safe='')).status_code == 200
    assert client.get('/documents').json()['documents'] == [names[1]]


@pytest.mark.parametrize('route', ['/upload', '/documents/upload'])
@pytest.mark.parametrize('content', [b'bad PDF', b'', pdf_bytes([''])], ids=['malformed', 'empty', 'textless'])
def test_invalid_upload_is_400_and_not_listed(client, route, content):
    response = client.post(route, files={'file': ('bad.pdf', content, 'application/pdf')})
    assert response.status_code == 400
    assert isinstance(response.json()['detail'], str)
    assert client.get('/documents').json() == {'documents': []}


@pytest.mark.parametrize('route', ['/upload', '/documents/upload'])
def test_upload_requires_multipart_file(client, route):
    assert client.post(route, json={'filename': 'fake.pdf'}).status_code == 422


@pytest.mark.parametrize('method,path,operation', [
    ('POST', '/upload', 'add_documents'), ('POST', '/documents/upload', 'add_documents'),
    ('GET', '/documents', 'list_unique_source_files'), ('DELETE', '/documents/a.pdf', 'delete_by_source_file')])
def test_storage_failures_do_not_return_false_success(client, repo, monkeypatch, method, path, operation):
    from fastapi.testclient import TestClient
    from app.main import app
    def unavailable(*args, **kwargs):
        raise RuntimeError('controlled storage failure')
    monkeypatch.setattr(repo, operation, unavailable)
    kwargs = {'files': {'file': ('a.pdf', pdf_bytes(['Evidence.']), 'application/pdf')}} if method == 'POST' else {}
    with TestClient(app, raise_server_exceptions=False) as transport:
        response = transport.request(method, path, **kwargs)
    assert response.status_code == 500 and 'successfully' not in response.text


@pytest.mark.parametrize('origin,allowed', [('http://localhost:5173', True), ('http://127.0.0.1:5173', True), ('https://untrusted.example', False)])
def test_cors_allows_only_configured_frontend_origins(client, origin, allowed):
    response = client.options('/chat', headers={'Origin': origin, 'Access-Control-Request-Method': 'POST',
                                               'Access-Control-Request-Headers': 'content-type'})
    assert (response.headers.get('access-control-allow-origin') == origin) is allowed
    assert response.status_code == (200 if allowed else 400)


def test_health_does_not_need_models_or_qdrant():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
