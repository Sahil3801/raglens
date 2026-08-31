"""Verify archived judge outputs and reject tampered copies without any API calls."""
import json
import shutil
from types import SimpleNamespace

import pytest

from comparison_artifacts import digest, write_json
import verify_comparison


@pytest.fixture
def verifier_fixture(tmp_path, corpus_root, monkeypatch):
    source = corpus_root.parents[1] / 'runs' / '20260831_synthetic_stations_v1_01'
    run = tmp_path / 'run-copy'
    shutil.copytree(source, run)
    # Isolate the source-file hash check from checkout history, just as other
    # tests isolate provider IO. The fixture still exercises the hash guard.
    # Corpus, prompts, answers, judge calls and metric values remain byte-for-byte
    # copies of genuine Phase 3 artifacts. Nothing in evaluation/runs is modified.
    source_file = tmp_path / 'backend' / 'app' / 'fixture_source.py'
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b'# controlled source-hash fixture\n')
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    metadata['source_sha256'] = {'app/fixture_source.py': digest(source_file.read_bytes())}
    write_json(run / 'metadata.json', metadata)
    monkeypatch.setattr(verify_comparison, '__file__', str(tmp_path / 'backend' / 'evaluation' / 'verify_comparison.py'))
    return run, source_file


def test_verifier_recomputes_archived_scores_and_exports(verifier_fixture):
    run, _ = verifier_fixture
    verify_comparison.main(SimpleNamespace(run=run, check_embeddings=False))
    report = json.loads((run / 'verification.json').read_text())
    assert report['status'] == 'passed'
    assert report['queries'] == 16 and report['arms'] == 32 and report['chunks'] == 72
    assert report['context_precision_scores_recomputed'] == 24
    assert report['faithfulness_scores_recomputed'] > 0
    assert report['answer_relevancy_scores_recomputed'] == 0  # Separate cached-model option.


@pytest.mark.parametrize('target', ['source', 'manifest', 'context_order', 'judge_request', 'aggregate', 'csv'])
def test_verifier_rejects_tampered_artifacts(verifier_fixture, target):
    run, source_file = verifier_fixture
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    if target == 'source':
        source_file.write_bytes(b'# modified source\n')
    elif target == 'manifest':
        manifest = read('manifest.json')
        manifest['seed'] += 1
        write_json(run / 'manifest.json', manifest)
    elif target == 'context_order':
        queries = read('per_query.json')
        queries[0]['arms']['B']['context_ids'].reverse()
        write_json(run / 'per_query.json', queries)
        write_json(run / 'queries' / (queries[0]['id'] + '.json'), queries[0])
    elif target == 'judge_request':
        query = read('per_query.json')[0]
        score = query['arms']['A']['metrics']['context_precision']
        path = run / 'judge_calls' / (score['judge_call_keys'][0] + '.json')
        record = json.loads(path.read_text(encoding='utf-8'))
        record['request']['judge']['model'] = 'different-judge'
        write_json(path, record)
    elif target == 'aggregate':
        aggregate = read('aggregate.json')
        aggregate[0]['A_mean'] = 999
        write_json(run / 'aggregate.json', aggregate)
    elif target == 'csv':
        (run / 'per_query.csv').write_text('query_id,arm\nwrong,A\n')
    with pytest.raises(AssertionError):
        verify_comparison.main(SimpleNamespace(run=run, check_embeddings=False))
