"""Real pinned RAGAS calculations; only structured judge IO/vectors are controlled.

The hand-calculated examples below are test fixtures, never benchmark results.
"""
import math

import pytest
from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, Faithfulness


@pytest.fixture
async def judge_factory(monkeypatch):
    clients = []
    def create(payloads):
        client = AsyncOpenAI(api_key='test-only')
        clients.append(client)
        judge = llm_factory('test-judge', client=client, temperature=0)
        remaining = list(payloads)
        prompts = []
        async def respond(prompt, response_model, **kwargs):
            prompts.append(prompt)
            return response_model.model_validate(remaining.pop(0))
        monkeypatch.setattr(judge, 'agenerate', respond)
        return judge, prompts, remaining
    yield create
    for client in clients:
        await client.close()


async def test_real_faithfulness_scores_supported_claim_fraction(judge_factory):
    judge, prompts, remaining = judge_factory([
        {'statements': ['A', 'B', 'C']},
        {'statements': [{'statement': s, 'reason': 'controlled verdict', 'verdict': v}
                        for s, v in [('A', 1), ('B', 0), ('C', 1)]]}])
    value = (await Faithfulness(llm=judge).ascore(
        user_input='test question', response='A B C', retrieved_contexts=['support for A and C'])).value
    assert value == pytest.approx(2 / 3)
    assert 'support for A and C' in prompts[1] and not remaining


async def test_no_claims_is_undefined_not_perfect_faithfulness(judge_factory):
    judge, _, _ = judge_factory([{'statements': []}])
    assert math.isnan((await Faithfulness(llm=judge).ascore(
        user_input='question', response='I cannot answer.', retrieved_contexts=['irrelevant'])).value)


@pytest.mark.parametrize('verdicts,expected', [([0, 1, 1], 7 / 12), ([1, 1, 0], 1), ([0, 0, 0], 0)])
async def test_context_precision_uses_reference_and_ranked_average_precision(judge_factory, verdicts, expected):
    judge, prompts, remaining = judge_factory([{'reason': 'controlled', 'verdict': v} for v in verdicts])
    contexts = ['first context', 'second context', 'third context']
    result = await ContextPrecisionWithReference(llm=judge).ascore(
        user_input='original question', reference='frozen ground truth', retrieved_contexts=contexts)
    assert result.value == pytest.approx(expected, abs=1e-9)
    assert len(prompts) == 3 and not remaining
    for prompt, context in zip(prompts, contexts):
        assert context in prompt and 'original question' in prompt and 'frozen ground truth' in prompt


class ControlledEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text, **kwargs):
        return [1., 0.] if text == 'original question' else [.6, .8]

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)


@pytest.mark.parametrize('question,noncommittal,expected', [
    ('generated question', 0, .6), ('generated question', 1, 0), ('', 0, 0)])
async def test_real_answer_relevancy_cosine_and_noncommittal_handling(judge_factory, question, noncommittal, expected):
    judge, prompts, remaining = judge_factory([{'question': question, 'noncommittal': noncommittal}])
    metric = AnswerRelevancy(llm=judge, embeddings=ControlledEmbeddings(), strictness=1)
    assert (await metric.ascore(user_input='original question', response='the answer')).value == pytest.approx(expected)
    assert len(prompts) == 1 and not remaining
