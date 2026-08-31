"""Verify archived A/B evidence and recompute aggregates without API calls.

Optional --check-embeddings also replays Answer Relevancy using the saved judge
questions and the recorded HuggingFace revision (requires cached model/deps).
This checks evaluation artifacts, not the deferred Phase 4 application test gaps.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from comparison_artifacts import digest, evidence_scores, resolve_evidence, summarize, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b, tolerance=1e-8):
    assert math.isclose(a, b, abs_tol=tolerance), (a, b)


def check_csv(path, expected_rows):
    with path.open(encoding='utf-8', newline='') as stream:
        actual_rows = list(csv.DictReader(stream))
    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows):
        for key, value in expected.items():
            assert actual[key] == ('' if value is None else str(value)), (path.name, key, actual[key], value)
        assert all(value == '' for key, value in actual.items() if key not in expected)


def main(args):
    run = args.run.resolve()
    metadata = read(run / 'metadata.json')
    manifest = read(run / 'manifest.json')
    chunks = read(run / 'chunks.json')
    queries = read(run / 'per_query.json')
    chunk_map = {c['chunk_id']: c for c in chunks}
    assert metadata['status'] in ['completed', 'completed_with_failures']
    assert digest((run / 'manifest.json').read_bytes()) == metadata['manifest_sha256']
    assert metadata['benchmark'] == manifest
    assert len(queries) == len(manifest['questions'])
    assert len(chunks) == metadata['corpus_chunks'] == len(chunk_map)
    assert metadata['embedding']['actual_dimension'] == 384
    for c in chunks:
        assert c['identity']['text_sha256'] == digest(c['text'].encode('utf-8'))
        assert c['chunk_id'] == digest(c['identity'])
    backend = Path(__file__).resolve().parent.parent
    source_checks = []
    for relative, expected in metadata['source_sha256'].items():
        if relative.startswith('app/'):
            assert digest((backend / relative).read_bytes()) == expected, relative
            source_checks.append(relative)
    checked_cp = checked_f = checked_ar = 0
    judged_chunks = {}
    ar_model = None
    if args.check_embeddings:
        from sentence_transformers import SentenceTransformer
        ar_model = SentenceTransformer(metadata['embedding']['model'],
                                       revision=metadata['embedding']['revision'], device='cpu',
                                       local_files_only=True)
        import numpy as np
        vectors = np.load(run / 'stored_vectors.npz')
        assert vectors['vectors'].shape == (len(chunks), 384)
        assert vectors['chunk_ids'].tolist() == list(chunk_map)
    for q, frozen in zip(queries, manifest['questions']):
        assert all(q[k] == v for k, v in frozen.items())
        assert read(run / 'queries' / (q['id'] + '.json')) == q
        groups = resolve_evidence(q, chunks)
        assert groups == q['resolved_evidence_groups']
        gold = set().union(*map(set, groups)) if groups else set()
        trace = q['rank_trace']
        pool = [t['chunk_id'] for t in trace]
        assert len(pool) == len(set(pool)) and set(pool) <= set(chunk_map)
        assert [t['vector_rank'] for t in trace] == list(range(1, len(trace) + 1))
        assert all(trace[i]['vector_score'] >= trace[i + 1]['vector_score'] for i in range(len(trace) - 1))
        mmr = [t['chunk_id'] for t in sorted([t for t in trace if t['mmr_rank']], key=lambda t:t['mmr_rank'])]
        assert mmr == q['mmr_ids'] and len(mmr) == min(40, len(pool))
        assert q['vector_requested'] == 60 and q['mmr_request']['k'] == 40
        assert q['mmr_request']['query']['mmr']['candidates_limit'] == 60
        close(q['mmr_request']['query']['mmr']['diversity'], 0.5)
        scores = {t['chunk_id']: t['crossencoder_score'] for t in trace if t['mmr_rank']}
        ordered = sorted(mmr, key=lambda cid:scores[cid], reverse=True)
        kept = [cid for cid in ordered if scores[cid] >= scores[ordered[0]] - 8.0]
        if len(kept) < 4:
            kept = ordered[:4]
        assert q['arms']['A']['context_ids'] == mmr[:8]
        assert q['arms']['B']['context_ids'] == kept[:8]
        assert len(set(q['arms']['B']['context_ids'])) == len(q['arms']['B']['context_ids'])
        assert q['mmr_evidence'] == evidence_scores(mmr, groups)
        assert set(q['gold_missing_from_mmr']) == gold - set(mmr)
        assert set(q['gold_missing_from_vector_pool']) == gold - set(pool)
        for t in trace:
            cid = t['chunk_id']
            assert t['gold_relevant'] == (cid in gold if q['answerable'] else None)
            assert t['rerank_rank'] == (ordered.index(cid) + 1 if cid in scores else None)
            if q['filter_filename']:
                assert chunk_map[cid]['metadata']['source_file'] == q['filter_filename']
            for arm in ['A', 'B']:
                ids = q['arms'][arm]['context_ids']
                assert t[arm + '_context_rank'] == (ids.index(cid) + 1 if cid in ids else None)
        a_ids, b_ids = (q['arms'][arm]['context_ids'] for arm in ['A', 'B'])
        assert set(q['promoted_relevant_into_context']) == (set(b_ids) - set(a_ids)) & gold
        assert set(q['lost_relevant_from_context']) == (set(a_ids) - set(b_ids)) & gold
        for arm, result in q['arms'].items():
            ids = result['context_ids']
            assert result['contexts'] == [chunk_map[cid]['text'] for cid in ids]
            assert result['evidence'] == evidence_scores(ids, groups)
            matched_ids = a_ids[:len(b_ids)] if arm == 'A' else b_ids
            assert result['matched_count_context_ids'] == matched_ids
            assert result['matched_count_evidence'] == evidence_scores(matched_ids, groups)
            if result['generation_status'] == 'ok':
                messages = result['generation_messages']
                context = '\n\n'.join(f"--- CHUNK FROM {chunk_map[cid]['metadata']['source_file']} ---\n{chunk_map[cid]['text']}\n--- END CHUNK ---" for cid in ids)
                assert messages[0]['content'].endswith('Context:\n' + context)
                assert messages[1] == {'type': 'human', 'content': q['question']}
                assert result['generation_metadata']['model_name'] == metadata['generation']['model']
            for metric, score in result['metrics'].items():
                records = []
                for key in score['judge_call_keys']:
                    path = run / 'judge_calls' / (key + '.json')
                    if not path.exists():
                        assert score['status'] == 'failed'
                        continue
                    record = read(path)
                    assert digest(record['request']) == key
                    assert record['request']['judge'] == metadata['judge']
                    records.append(record['response'])
                if score['status'] == 'ok':
                    assert math.isfinite(score['value'])
                    if metric == 'context_precision':
                        assert q['answerable'] and len(records) == len(ids)
                        verdicts = [r['verdict'] for r in records]
                        for rank, (cid, record) in enumerate(zip(ids, records), start=1):
                            key = (q['id'], cid)
                            evidence = judged_chunks.setdefault(key, {
                                'query_id': q['id'], 'chunk_id': cid, 'question': q['question'],
                                'source_file': chunk_map[cid]['metadata']['source_file'],
                                'page': chunk_map[cid]['metadata']['page'],
                                'gold_relevant': cid in gold, 'judge_verdict': record['verdict'],
                                'judge_reason': record['reason'], 'arm_context_ranks': {}})
                            assert evidence['judge_verdict'] == record['verdict']
                            evidence['arm_context_ranks'][arm] = rank
                        value = sum(sum(verdicts[:i + 1]) / (i + 1) * v for i, v in enumerate(verdicts)) / (sum(verdicts) + 1e-10)
                        close(value, score['value'])
                        checked_cp += 1
                    elif metric == 'faithfulness':
                        assert len(records) == 2
                        verdicts = records[1]['statements']
                        close(sum(bool(s['verdict']) for s in verdicts) / len(verdicts), score['value'])
                        checked_f += 1
                    elif metric == 'answer_relevancy' and ar_model is not None:
                        assert len(records) == 1  # Frozen strictness = 1.
                        generated = records[0]
                        if not generated['question'] or generated['noncommittal']:
                            value = 0.0
                        else:
                            original = np.asarray(ar_model.encode(q['question'], normalize_embeddings=True).tolist())
                            generated_vector = np.asarray(ar_model.encode([generated['question']], normalize_embeddings=True).tolist()).reshape(-1)
                            value = float(np.dot(original, generated_vector) / (np.linalg.norm(original) * np.linalg.norm(generated_vector)))
                        close(value, score['value'], tolerance=1e-5)
                        checked_ar += 1
                elif score['status'] == 'undefined' and metric == 'faithfulness':
                    assert score['value'] is None and not records[-1]['statements']
                elif score['status'] == 'not_applicable':
                    assert metric == 'context_precision' and not q['answerable'] and score['value'] is None
    rows, aggregates = summarize(queries)
    assert aggregates == read(run / 'aggregate.json')
    check_csv(run / 'per_query.csv', rows)
    check_csv(run / 'aggregate.csv', aggregates)
    check_csv(run / 'candidate_ranks.csv', [dict(query_id=q['id'], **r) for q in queries for r in q['rank_trace']])
    report = {'status': 'passed', 'queries': len(queries), 'arms': len(rows), 'chunks': len(chunks),
              'production_source_files_unchanged': len(source_checks),
              'context_precision_scores_recomputed': checked_cp, 'faithfulness_scores_recomputed': checked_f,
              'answer_relevancy_scores_recomputed': checked_ar,
              'checks': ['frozen inputs and evidence', 'candidate provenance and filtering', 'MMR ranks',
                         'production threshold/top-N replay', 'promotions and losses', 'actual generation prompts/model',
                         'real judge outputs and input hashes', 'per-query JSON/CSV and macro/paired aggregates']}
    disagreements = [v for v in judged_chunks.values() if v['gold_relevant'] != bool(v['judge_verdict'])]
    write_json(run / 'judge_disagreements.json', {
        'method': 'Compare actual Context Precision usefulness verdicts against frozen contributing-evidence labels; deduplicate identical query/chunk pairs across arms. Scores are NOT corrected.',
        'unique_query_chunk_judgments': len(judged_chunks), 'disagreement_count': len(disagreements),
        'disagreements': disagreements})
    report['context_precision_gold_label_disagreements'] = len(disagreements)
    write_json(run / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--check-embeddings', action='store_true')
    main(parser.parse_args())
