from unittest.mock import MagicMock
from io import BytesIO
import pytest
from fastapi import HTTPException, UploadFile
from pypdf import PdfWriter
from app.services.ingestion import IngestionService

def test_ingestion_service_init():
    mock_repo = MagicMock()
    service = IngestionService(repo=mock_repo)
    assert service.splitter._chunk_size == 800
    assert service.splitter._chunk_overlap == 200


def test_invalid_pdf_is_rejected_without_indexing_and_temp_file_is_removed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = MagicMock()
    file = UploadFile(filename="invalid.pdf", file=BytesIO(b"not a PDF"))
    with pytest.raises(HTTPException) as error:
        IngestionService(repo).process_pdf_upload(file)
    assert error.value.status_code == 400
    repo.add_documents.assert_not_called()
    assert not (tmp_path / "temp_invalid.pdf").exists()


def test_pdf_without_text_is_rejected_instead_of_reporting_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(stream)
    stream.seek(0)
    repo = MagicMock()
    with pytest.raises(HTTPException) as error:
        IngestionService(repo).process_pdf_upload(UploadFile(filename="blank.pdf", file=stream))
    assert error.value.status_code == 400
    assert "no extractable text" in error.value.detail
    repo.add_documents.assert_not_called()
    assert not (tmp_path / "temp_blank.pdf").exists()
