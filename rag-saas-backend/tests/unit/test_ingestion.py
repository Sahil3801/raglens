from unittest.mock import MagicMock
from app.services.ingestion import IngestionService

def test_ingestion_service_init():
    mock_repo = MagicMock()
    service = IngestionService(repo=mock_repo)
    assert service.splitter._chunk_size == 500
    assert service.splitter._chunk_overlap == 100