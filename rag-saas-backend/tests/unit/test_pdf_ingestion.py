from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile
from app.services.ingestion import IngestionService
from tests.pdf_factory import pdf_bytes


def upload(data, name='paper.pdf'):
    return UploadFile(filename=name, file=BytesIO(data))


def test_real_pdf_extraction_chunk_overlap_and_page_metadata():
    paragraphs = [' '.join(f'p{page}token{i:03d}' for i in range(230)) for page in range(2)]
    repo = MagicMock()
    name = 'Notes & resume #1.pdf'
    assert IngestionService(repo).process_pdf_upload(upload(pdf_bytes(paragraphs), name)) == name
    chunks = repo.add_documents.call_args.args[0]
    assert len(chunks) > 2
    assert {c.metadata['page'] for c in chunks} == {0, 1}
    assert all(c.metadata['source_file'] == name and c.metadata['total_pages'] == 2 for c in chunks)
    assert all(0 < len(c.page_content) <= 800 for c in chunks)
    for page in range(2):
        parts = [c.page_content for c in chunks if c.metadata['page'] == page]
        reconstructed = parts[0]
        for left, right in zip(parts, parts[1:]):
            overlap = max(i for i in range(201) if left.endswith(right[:i]))
            assert 180 <= overlap <= 200
            reconstructed += right[overlap:]
        assert reconstructed == paragraphs[page]


@pytest.mark.parametrize('data', [b'', b'not a PDF', b'%PDF-1.7\ntruncated', pdf_bytes(['secret'], password='locked')],
                         ids=['empty', 'not-pdf', 'truncated', 'encrypted'])
def test_malformed_empty_or_encrypted_pdf_is_400_and_never_indexed(data, monkeypatch):
    from app.services import ingestion
    paths = []
    original = ingestion.tempfile.mkstemp
    def record_tempfile(**kwargs):
        descriptor, name = original(**kwargs)
        paths.append(Path(name))
        return descriptor, name
    monkeypatch.setattr(ingestion.tempfile, 'mkstemp', record_tempfile)
    repo = MagicMock()
    with pytest.raises(HTTPException) as error:
        IngestionService(repo).process_pdf_upload(upload(data))
    assert error.value.status_code == 400
    repo.add_documents.assert_not_called()
    assert len(paths) == 1 and not paths[0].exists()


def test_blank_pages_do_not_erase_page_numbers():
    repo = MagicMock()
    IngestionService(repo).process_pdf_upload(upload(pdf_bytes(['', 'Evidence on page two.'])))
    chunks = repo.add_documents.call_args.args[0]
    assert len(chunks) == 1
    assert chunks[0].metadata['page'] == 1
    assert chunks[0].page_content == 'Evidence on page two.'


def test_repository_failure_cleans_temporary_pdf(monkeypatch):
    from app.services import ingestion
    paths = []
    loader = ingestion.PyPDFLoader
    def record_loader(path):
        paths.append(Path(path))
        return loader(path)
    monkeypatch.setattr(ingestion, 'PyPDFLoader', record_loader)
    repo = MagicMock()
    repo.add_documents.side_effect = RuntimeError('storage unavailable')
    with pytest.raises(RuntimeError, match='storage unavailable'):
        IngestionService(repo).process_pdf_upload(upload(pdf_bytes(['Evidence.'])))
    assert paths and all(not path.exists() for path in paths)


def test_upload_does_not_overwrite_existing_workspace_file():
    sentinel = Path('temp_paper.pdf')
    sentinel.write_bytes(b'user-owned data')
    IngestionService(MagicMock()).process_pdf_upload(upload(pdf_bytes(['Evidence.'])))
    assert sentinel.exists() and sentinel.read_bytes() == b'user-owned data'


def test_concurrent_same_filename_uploads_keep_their_own_content(monkeypatch):
    from app.services import ingestion
    barrier = Barrier(2)
    original_load = ingestion.PyPDFLoader.load
    def simultaneous_load(loader):
        barrier.wait(timeout=10)
        return original_load(loader)
    monkeypatch.setattr(ingestion.PyPDFLoader, 'load', simultaneous_load)

    def process(text):
        repo = MagicMock()
        IngestionService(repo).process_pdf_upload(upload(pdf_bytes([text]), 'same.pdf'))
        return repo.add_documents.call_args.args[0][0].page_content
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(process, ['Aurora evidence.', 'Borealis evidence.'])) == ['Aurora evidence.', 'Borealis evidence.']
