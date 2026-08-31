"""Exercise the real comparison runner with real PDFs/Qdrant, isolated model IO.

All artifacts stay in pytest's temporary directory. Fake scores are not evidence
of answer quality and must never be copied into evaluation/runs.
"""
import csv
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from comparison_artifacts import digest, evidence_scores
import run_comparison
from tests.conftest import FixedEmbeddings


@pytest.mark.parametrize('inject_failures', [False, True], ids=['success', 'provider-and-undefined-failures'])
async def test_paired_runner_preserves_inputs_ranks_and_failure_accounting(
        monkeypatch, tmp_path, corpus_root, inject_failures):
    from app.core.config import settings
    from app.repositories import vector_store
    from app.services import generation, reranking
    import openai
    import ragas.embeddings
    import ragas.llms
    import ragas.metrics.collections as metrics

    # This entry point normally runs in a fresh process. Isolate its environment
    # changes here so later tests cannot inherit a run's collection or model.
    for key in ['QDRANT_PATH', 'QDRANT_COLLECTION', 'GROQ_MODEL', 'TOKENIZERS_PARALLELISM', 'RAGAS_DO_NOT_TRACK']:
        import os
        monkeypatch.setenv(key, os.environ.get(key, ''))
    monkeypatch.setattr(run_comparison, 'EVAL', tmp_path / 'evaluation')
    monkeypatch.setattr(settings, 'GROQ_MODEL', 'test-generator')

    class AuditEmbeddings(FixedEmbeddings):
        _client = [SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash='test-embedding')))]
        encode_kwargs = {}
        query_encode_kwargs = {}
    monkeypatch.setattr(vector_store, 'HuggingFaceEmbeddings', AuditEmbeddings)
    encoder = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash='test-crossencoder')),
                              predict=lambda pairs: np.arange(len(pairs), dtype=float))
    monkeypatch.setattr(reranking.reranker_service, 'encoder', encoder)

    manifest = json.loads((corpus_root / 'manifest.json').read_text(encoding='utf-8'))
    frozen_bytes = (corpus_root / 'manifest.json').read_bytes()
    generation_calls = []
    with patch.object(generation, 'ChatGroq', return_value=FakeListChatModel(responses=['controlled test answer'])):
        generator = generation.GenerationService()
    original_generate = generator.generate_answer
    def generate(question, docs):
        generation_calls.append((question, [d.metadata['eval_chunk_id'] for d in docs]))
        if inject_failures and question == manifest['questions'][1]['question']:
            raise RuntimeError('controlled generation failure')
        return original_generate(question, docs)
    monkeypatch.setattr(generator, 'generate_answer', generate)
    monkeypatch.setattr(generation, 'generation_service', generator)

    fake_client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(
        data=[SimpleNamespace(id='test-generator'), SimpleNamespace(id='test-judge')]))), close=AsyncMock())
    monkeypatch.setattr(openai, 'AsyncOpenAI', lambda **kwargs: fake_client)
    fake_judge = SimpleNamespace(agenerate=AsyncMock(), model_args={'temperature': 0})
    monkeypatch.setattr(ragas.llms, 'llm_factory', lambda *args, **kwargs: fake_judge)
    monkeypatch.setattr(ragas.embeddings, 'HuggingFaceEmbeddings', lambda **kwargs: object())
    metric_calls = []
    def metric_factory(name):
        class ControlledMetric:
            def __init__(self, **kwargs):
                assert kwargs['llm'] is fake_judge
                if name == 'answer_relevancy':
                    assert kwargs['strictness'] == 1

            async def ascore(self, **kwargs):
                metric_calls.append((name, kwargs))
                if inject_failures and kwargs['user_input'] == manifest['questions'][0]['question']:
                    if name == 'faithfulness':
                        return SimpleNamespace(value=float('nan'))
                    if name == 'answer_relevancy':
                        raise RuntimeError('429 controlled quota failure')
                return SimpleNamespace(value=.5)
        return ControlledMetric
    for classname, name in [('Faithfulness', 'faithfulness'), ('ContextPrecisionWithReference', 'context_precision'),
                            ('AnswerRelevancy', 'answer_relevancy')]:
        monkeypatch.setattr(metrics, classname, metric_factory(name))

    args = SimpleNamespace(run_id='test_only', manifest=corpus_root / 'manifest.json',
                           generation_model='test-generator', judge_model='test-judge')
    if inject_failures:
        with pytest.raises(SystemExit) as error:
            await run_comparison.main(args)
        assert error.value.code == 1
    else:
        await run_comparison.main(args)
    run = tmp_path / 'evaluation' / 'runs' / 'test_only'
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    metadata, queries, chunks = read('metadata.json'), read('per_query.json'), read('chunks.json')
    assert metadata['status'] == ('completed_with_failures' if inject_failures else 'completed')
    assert metadata['failed_arms'] == (4 if inject_failures else 0)
    assert metadata['production_sources_unchanged'] is True
    assert metadata['embedding']['actual_dimension'] == 384 and len(chunks) == 72
    assert metadata['generation']['model'] == 'test-generator' and metadata['judge']['model'] == 'test-judge'
    assert len(queries) == 16 and len(generation_calls) == 32
    assert (corpus_root / 'manifest.json').read_bytes() == frozen_bytes
    fake_client.close.assert_awaited_once()
    for q, frozen in zip(queries, manifest['questions']):
        assert all(q[key] == value for key, value in frozen.items())
        assert read('queries/' + q['id'] + '.json') == q
        mmr = q['mmr_ids']
        assert q['vector_requested'] == 60 and q['mmr_request']['k'] == 40
        assert len(mmr) == min(40, len(q['rank_trace']))
        a, b = q['arms']['A'], q['arms']['B']
        assert a['context_ids'] == mmr[:8]
        assert b['context_ids'] == list(reversed(mmr))[:8]
        assert [r['chunk_id'] for r in sorted([r for r in q['rank_trace'] if r['mmr_rank']],
                                             key=lambda r: r['mmr_rank'])] == mmr
        gold = set().union(*map(set, q['resolved_evidence_groups'])) if q['answerable'] else set()
        assert set(q['promoted_relevant_into_context']) == (set(b['context_ids']) - set(a['context_ids'])) & gold
        for arm, result in q['arms'].items():
            assert (q['question'], result['context_ids']) in generation_calls
            assert result['evidence'] == evidence_scores(result['context_ids'], q['resolved_evidence_groups'])
            if q['filter_filename']:
                selected = [c for c in chunks if c['chunk_id'] in result['context_ids']]
                assert all(c['metadata']['source_file'] == q['filter_filename'] for c in selected)
            if not q['answerable']:
                assert result['metrics']['context_precision']['status'] == 'not_applicable'
                assert result['metrics']['context_precision']['value'] is None
            if result['generation_status'] == 'ok':
                messages = result['generation_messages']
                assert messages[1] == {'type': 'human', 'content': q['question']}
                for context in result['contexts']:
                    assert context in messages[0]['content']
            if inject_failures and q['id'] == 'q01':
                assert result['metrics']['faithfulness']['status'] == 'undefined'
                assert result['metrics']['faithfulness']['value'] is None
                assert result['metrics']['answer_relevancy']['status'] == 'failed'
            if inject_failures and q['id'] == 'q02':
                assert result['metrics']['faithfulness']['status'] == 'generation_failed'
                assert result['metrics']['answer_relevancy']['status'] == 'generation_failed'
                assert result['metrics']['context_precision']['status'] == 'ok'
    questions = {q['question']: q for q in manifest['questions']}
    cp_calls = [kwargs for name, kwargs in metric_calls if name == 'context_precision']
    assert len(cp_calls) == 24
    for name, kwargs in metric_calls:
        if name == 'context_precision':
            assert kwargs['reference'] == questions[kwargs['user_input']]['reference']
            assert 'response' not in kwargs
        else:
            assert kwargs['response'] == 'controlled test answer' and 'reference' not in kwargs
    with (run / 'per_query.csv').open(encoding='utf-8', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 32
    # Reusing an ID cannot overwrite any artifacts.
    original_metadata = (run / 'metadata.json').read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        await run_comparison.main(args)
    assert (run / 'metadata.json').read_bytes() == original_metadata


async def test_runner_rejects_changed_corpus_before_creating_results(monkeypatch, tmp_path, corpus_root):
    manifest = json.loads((corpus_root / 'manifest.json').read_text())
    for document in manifest['documents']:
        document['path'] = str(corpus_root / document['path'])
    manifest['documents'][0]['sha256'] = 'bad-hash'
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(run_comparison, 'EVAL', tmp_path / 'eval')
    args = SimpleNamespace(run_id='must_not_exist', manifest=path)
    with pytest.raises(ValueError, match='Corpus hash mismatch'):
        await run_comparison.main(args)
    assert not (tmp_path / 'eval').exists()


def test_archived_real_vectors_replay_identical_ranks_without_network(tmp_path, corpus_root):
    from replay_retrieval import main
    source = corpus_root.parents[1] / 'runs' / '20260831_synthetic_stations_v1_01'
    run = tmp_path / 'archive_copy'
    run.mkdir()
    for name in ['metadata.json', 'chunks.json', 'stored_vectors.npz']:
        shutil.copy2(source / name, run / name)
    shutil.copytree(source / 'queries', run / 'queries')
    original = digest((source / 'retrieval_replay.json').read_bytes())
    main(run)
    report = json.loads((run / 'retrieval_replay.json').read_text())
    assert report['status'] == 'passed' and len(report['queries']) == 16
    assert all(q['vector_and_mmr_ranks_identical'] for q in report['queries'])
    assert digest((source / 'retrieval_replay.json').read_bytes()) == original


@pytest.mark.parametrize('problem', ['completed', 'exists', 'no_failures', 'code_failure'])
async def test_recovery_rejects_unsafe_or_unnecessary_rescoring(monkeypatch, tmp_path, problem):
    import recover_failed_metrics
    monkeypatch.setattr(recover_failed_metrics, 'EVAL', tmp_path)
    source = tmp_path / 'source'
    source.mkdir()
    metadata = {'status': 'completed' if problem == 'completed' else 'completed_with_failures',
                'packages': {name: version(name) for name in ['ragas', 'openai', 'instructor',
                    'sentence-transformers', 'torch', 'langchain-community']}}
    (source / 'metadata.json').write_text(json.dumps(metadata))
    (source / 'per_query.json').write_text(json.dumps([{'id': 'q1', 'arms': {'A': {'metrics': {
        'faithfulness': {'value': None, 'status': 'failed' if problem == 'code_failure' else 'undefined',
                         'error': 'ValueError: broken metric'}}}}}]))
    if problem == 'exists':
        (tmp_path / 'runs' / 'new_run').mkdir(parents=True)
    patterns = {'completed': 'finished run', 'exists': 'Destination exists',
                'no_failures': 'No failed metrics', 'code_failure': 'Non-rate-limit failure'}
    with pytest.raises(ValueError, match=patterns[problem]):
        await recover_failed_metrics.main(SimpleNamespace(source=source, run_id='new_run'))
    assert not (tmp_path / 'runs' / 'new_run' / 'metadata.json').exists()
