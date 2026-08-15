from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.services.retrieval import RetrievalService

def test_retrieval_service_invokes_repo():
    mock_repo = MagicMock()
    mock_repo.search_mmr.return_value = [
        Document(page_content="Sample text", metadata={"source_file": "sample.pdf"})
    ]
    service = RetrievalService(repo=mock_repo)
    results = service.retrieve("test query")
    
    assert len(results) == 1
    assert results[0].page_content == "Sample text"
    mock_repo.search_mmr.assert_called_once()