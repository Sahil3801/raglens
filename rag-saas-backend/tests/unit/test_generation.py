from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from app.core.config import settings
from app.services.generation import GenerationService

REFUSAL = 'I cannot answer this based on the provided document.'


@pytest.mark.parametrize('documents', [[], [Document(page_content='Only archive housekeeping.', metadata={})]])
def test_refusal_and_grounding_instruction_survive_prompt_and_parser(monkeypatch, documents):
    observed = []
    def provider(prompt):
        observed.extend(prompt.to_messages())
        return AIMessage(content=REFUSAL)
    constructor = MagicMock(return_value=RunnableLambda(provider))
    monkeypatch.setattr('app.services.generation.ChatGroq', constructor)
    generator = GenerationService()
    assert generator.generate_answer('Unsupported question', documents) == REFUSAL
    constructor.assert_called_once_with(model_name=settings.GROQ_MODEL, temperature=0, api_key=settings.GROQ_API_KEY)
    assert observed[1].content == 'Unsupported question'
    assert 'Do not use external assumptions or outside knowledge.' in observed[0].content
    assert "you MUST reply ONLY with: '" + REFUSAL + "'" in observed[0].content
    if documents:
        assert '--- CHUNK FROM Unknown Document ---\nOnly archive housekeeping.\n--- END CHUNK ---' in observed[0].content


def test_multi_document_prompt_preserves_source_boundaries_text_and_order(monkeypatch):
    observed = []
    def provider(prompt):
        observed.extend(prompt.to_messages())
        return AIMessage(content='The supplied answer.')
    monkeypatch.setattr('app.services.generation.ChatGroq', lambda **kwargs: RunnableLambda(provider))
    documents = [Document(page_content='Second {literal} text.', metadata={'source_file': 'b.pdf'}),
                 Document(page_content='First text.', metadata={'source_file': 'a.pdf'})]
    assert GenerationService().generate_answer('Use {only} context', documents) == 'The supplied answer.'
    expected = '--- CHUNK FROM b.pdf ---\nSecond {literal} text.\n--- END CHUNK ---\n\n--- CHUNK FROM a.pdf ---\nFirst text.\n--- END CHUNK ---'
    assert observed[0].content.endswith('Context:\n' + expected)
    assert observed[1].content == 'Use {only} context'


def test_provider_failure_is_not_reported_as_a_grounded_answer(monkeypatch):
    def unavailable(prompt):
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr('app.services.generation.ChatGroq', lambda **kwargs: RunnableLambda(unavailable))
    with pytest.raises(RuntimeError, match='provider unavailable'):
        GenerationService().generate_answer('question', [Document(page_content='Evidence')])
