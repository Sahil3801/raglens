import csv
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from comparison_artifacts import digest, evidence_scores, export_run, resolve_evidence, summarize, write_json
from run_comparison import RecordedJudge, safe_error


def test_ranking_metrics_distinguish_equivalent_chunks_from_required_facts():
    # a/b are alternative evidence for fact one; c is required for fact two.
    scores = evidence_scores(['noise', 'b', 'c'], [['a', 'b'], ['c']])
    assert scores['hit'] == 1
    assert scores['chunk_precision'] == pytest.approx(2 / 3)
    assert scores['chunk_recall'] == pytest.approx(2 / 3)
    assert scores['evidence_group_recall'] == 1
    assert scores['all_evidence'] == 1
    assert scores['reciprocal_rank'] == .5
    assert scores['ndcg'] == pytest.approx((1 / math.log2(3) + .5) / (1 + 1 / math.log2(3) + .5))
    incomplete = evidence_scores(['a', 'b'], [['a', 'b'], ['c']])
    assert incomplete['evidence_group_recall'] == .5
    assert incomplete['all_evidence'] == 0


def test_missing_retrieval_is_zero_but_unlabeled_query_is_not_scored():
    assert set(evidence_scores([], [['gold']]).values()) == {0.0}
    assert set(evidence_scores(['noise'], [['gold']]).values()) == {0.0}
    assert evidence_scores(['anything'], []) == {}


@pytest.mark.parametrize('hits', [0, 2])
def test_evidence_resolution_rejects_missing_and_ambiguous_labels(hits):
    question = {'id': 'q', 'answerable': True, 'evidence_groups': [[
        {'source_file': 'a.pdf', 'page': 2, 'contains': 'code 731'}]]}
    chunk = {'chunk_id': 'a', 'metadata': {'source_file': 'a.pdf', 'page': 2}, 'text': 'code\n731'}
    with pytest.raises(ValueError, match=f'resolved to {hits} chunks'):
        resolve_evidence(question, [chunk] * hits)
    assert resolve_evidence(question, [chunk]) == [['a']]
    question['answerable'] = False
    with pytest.raises(ValueError, match='Answerability'):
        resolve_evidence(question, [chunk])


def query(qid, a, b, answerable=True):
    def arm(value):
        return {'context_ids': ['x'], 'generation_status': 'ok', 'answer': 'Answer, with\nUnicode: café',
                'exact_refusal': False, 'evidence': {}, 'matched_count_evidence': {},
                'metrics': {'faithfulness': {'value': value, 'status': 'ok' if value is not None else 'failed'}}}
    return {'id': qid, 'answerable': answerable, 'filter_filename': None, 'question': 'Question?',
            'reference': 'Reference', 'rank_trace': [{'chunk_id': 'x', 'mmr_rank': 1, 'rerank_rank': 2}],
            'arms': {'A': arm(a), 'B': arm(b)}}


def test_aggregates_pair_by_query_and_never_impute_failed_scores():
    queries = [query('q1', 0, 1), query('q2', 1, None), query('q3', None, .5),
               query('q4', .25, .25), query('q5', None, None, False)]
    queries[-1]['arms']['A']['metrics']['faithfulness']['status'] = 'undefined'
    queries[-1]['arms']['B']['metrics']['faithfulness']['status'] = 'generation_failed'
    _, aggregates = summarize(queries)
    row = next(r for r in aggregates if r['population'] == 'answerable' and r['metric'] == 'faithfulness')
    assert row['queries'] == 4 and row['A_valid'] == row['B_valid'] == 3
    assert row['A_mean'] == pytest.approx(1.25 / 3)
    assert row['B_mean'] == pytest.approx(1.75 / 3)
    assert row['paired_valid'] == 2
    assert row['paired_A_mean'] == .125 and row['paired_B_mean'] == .625
    assert row['paired_mean_delta_B_minus_A'] == .5
    assert row['improved_pairs'] == 1 and row['tied_pairs'] == 1 and row['worsened_pairs'] == 0
    assert row['A_failed'] == row['B_failed'] == 1
    missing = next(r for r in aggregates if r['population'] == 'unanswerable' and r['metric'] == 'faithfulness')
    assert missing['paired_valid'] == 0 and missing['A_mean'] is None and missing['B_mean'] is None
    assert missing['A_undefined'] == 1 and missing['B_generation_failed'] == 1


def test_export_roundtrips_newlines_unicode_ranks_and_nulls(tmp_path):
    queries = [query('q1', 0, None)]
    export_run(tmp_path, queries)
    assert json.loads((tmp_path / 'per_query.json').read_text(encoding='utf-8')) == queries
    with (tmp_path / 'per_query.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['answer'] == 'Answer, with\nUnicode: café'
    assert rows[0]['faithfulness'] == '0' and rows[1]['faithfulness'] == ''
    assert rows[1]['faithfulness_status'] == 'failed'
    with (tmp_path / 'candidate_ranks.csv').open(encoding='utf-8', newline='') as stream:
        assert list(csv.DictReader(stream)) == [{'query_id': 'q1', 'chunk_id': 'x', 'mmr_rank': '1', 'rerank_rank': '2'}]


def test_json_rejects_nan_without_destroying_previous_artifact(tmp_path):
    path = tmp_path / 'artifact.json'
    write_json(path, {'value': .5})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {'value': math.nan})
    assert path.read_bytes() == original
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1})


class Verdict(BaseModel):
    verdict: int


async def test_judge_cache_reuses_only_identical_inputs_and_validates_cache(tmp_path):
    original = AsyncMock(return_value=Verdict(verdict=1))
    llm = SimpleNamespace(agenerate=original)
    recorder = RecordedJudge(llm, tmp_path, {'model': 'judge', 'seed': 7})
    assert (await llm.agenerate('same prompt', Verdict)).verdict == 1
    assert (await llm.agenerate('same prompt', Verdict)).verdict == 1
    original.assert_awaited_once()
    await llm.agenerate('different prompt', Verdict)
    await llm.agenerate('same prompt', Verdict, temperature=.1)
    recorder.configuration = {'model': 'other-judge', 'seed': 7}
    await llm.agenerate('same prompt', Verdict)
    assert original.await_count == 4 and recorder.cache_hits == 1
    path = tmp_path / (recorder.active_keys[-1] + '.json')
    record = json.loads(path.read_text())
    record['response'] = {'verdict': 'not-an-integer'}
    write_json(path, record)
    with pytest.raises(ValidationError):
        await llm.agenerate('same prompt', Verdict)
    assert original.await_count == 4


async def test_judge_failure_is_never_cached_as_a_success(tmp_path, monkeypatch):
    llm = SimpleNamespace(agenerate=AsyncMock(side_effect=RuntimeError('429 unavailable')))
    recorder = RecordedJudge(llm, tmp_path, {'model': 'judge'})
    for _ in range(2):
        with pytest.raises(RuntimeError, match='429'):
            await llm.agenerate('prompt', Verdict)
    assert recorder.network_calls == 2 and recorder.cache_hits == 0
    assert list(tmp_path.glob('*.json')) == []
    monkeypatch.setenv('GROQ_API_KEY', 'secret-test-token')
    assert safe_error(ValueError('rejected secret-test-token')) == 'ValueError: rejected [REDACTED]'
