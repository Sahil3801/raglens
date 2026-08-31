"""Serialization, evidence scoring, and aggregation; no network/model calls."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_evidence(question, chunks):
    """Reject missing/ambiguous labels instead of guessing ground truth."""
    groups = []
    for alternatives in question['evidence_groups']:
        matches = set()
        for label in alternatives:
            hits = [c['chunk_id'] for c in chunks
                    if c['metadata']['source_file'] == label['source_file']
                    and c['metadata']['page'] == label['page']
                    and label['contains'] in ' '.join(c['text'].split())]
            if len(hits) != 1:
                raise ValueError(f"{question['id']}: evidence label resolved to {len(hits)} chunks: {label}")
            matches.update(hits)
        groups.append(sorted(matches))
    if bool(groups) != question['answerable']:
        raise ValueError('Answerability and evidence labels disagree')
    return groups


def evidence_scores(ids, groups):
    """Binary exhaustive chunk labels plus AND-of-ORs required evidence groups.

    Chunk recall measures coverage of ALL relevant chunks, even equivalent ones;
    evidence-group recall instead measures whether each needed fact is present.
    """
    if not groups:
        return {}
    gold = set().union(*map(set, groups))
    found = set(ids) & gold
    gains = [int(i in gold) for i in ids]
    dcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(gold), len(ids))))
    return {
        'hit': float(bool(found)),
        'chunk_precision': len(found) / len(ids) if ids else 0.0,
        'chunk_recall': len(found) / len(gold),
        'evidence_group_recall': sum(bool(set(g) & set(ids)) for g in groups) / len(groups),
        'all_evidence': float(all(set(g) & set(ids) for g in groups)),
        'reciprocal_rank': next((1 / (i + 1) for i, g in enumerate(gains) if g), 0.0),
        'ndcg': dcg / ideal if ideal else 0.0,
    }


def summarize(queries):
    rows = []
    for q in queries:
        for arm, result in q['arms'].items():
            row = {'query_id': q['id'], 'answerable': q['answerable'], 'arm': arm,
                   'filter_filename': q['filter_filename'], 'question': q['question'],
                   'reference': q['reference'], 'answer': result.get('answer'),
                   'context_count': len(result['context_ids']),
                   'generation_status': result.get('generation_status', 'pending'),
                   'exact_refusal': result.get('exact_refusal')}
            for metric, score in result.get('metrics', {}).items():
                row[metric] = score['value']
                row[metric + '_status'] = score['status']
            row.update({'gold_' + k: v for k, v in result['evidence'].items()})
            row.update({'matched_count_gold_' + k: v for k, v in result['matched_count_evidence'].items()})
            rows.append(row)
    metric_names = ['faithfulness', 'context_precision', 'answer_relevancy']
    numeric = metric_names + ['context_count', 'exact_refusal'] + [
        prefix + k for prefix in ['gold_', 'matched_count_gold_']
        for k in ['hit', 'chunk_precision', 'chunk_recall', 'evidence_group_recall', 'all_evidence', 'reciprocal_rank', 'ndcg']]
    aggregates = []
    for population in ['answerable', 'unanswerable', 'all']:
        subset = [r for r in rows if population == 'all' or r['answerable'] == (population == 'answerable')]
        for metric in numeric:
            by_arm = {a: {r['query_id']: r for r in subset if r['arm'] == a} for a in ['A', 'B']}
            valid = {a: {k: float(v[metric]) for k, v in data.items()
                         if isinstance(v.get(metric), (float, int)) and math.isfinite(v[metric])}
                     for a, data in by_arm.items()}
            paired_ids = sorted(set(valid['A']) & set(valid['B']))
            delta = [valid['B'][i] - valid['A'][i] for i in paired_ids]
            aggregate = {'population': population, 'metric': metric, 'queries': len(subset) // 2,
                         'A_mean': mean(valid['A'].values()) if valid['A'] else None,
                         'B_mean': mean(valid['B'].values()) if valid['B'] else None,
                         'A_valid': len(valid['A']), 'B_valid': len(valid['B']),
                         'paired_valid': len(delta), 'paired_mean_delta_B_minus_A': mean(delta) if delta else None,
                         'paired_A_mean': mean(valid['A'][i] for i in paired_ids) if delta else None,
                         'paired_B_mean': mean(valid['B'][i] for i in paired_ids) if delta else None,
                         'improved_pairs': sum(v > 1e-8 for v in delta),
                         'worsened_pairs': sum(v < -1e-8 for v in delta),
                         'tied_pairs': sum(abs(v) <= 1e-8 for v in delta)}
            if metric in metric_names:
                for arm in ['A', 'B']:
                    for status in ['failed', 'undefined', 'not_applicable', 'generation_failed']:
                        aggregate[f'{arm}_{status}'] = sum(r.get(metric + '_status') == status for r in by_arm[arm].values())
            aggregates.append(aggregate)
    return rows, aggregates


def export_run(run_dir, queries):
    rows, aggregates = summarize(queries)
    write_json(run_dir / 'per_query.json', queries)
    write_csv(run_dir / 'per_query.csv', rows)
    write_json(run_dir / 'aggregate.json', aggregates)
    write_csv(run_dir / 'aggregate.csv', aggregates)
    ranks = [dict(query_id=q['id'], **r) for q in queries for r in q['rank_trace']]
    write_csv(run_dir / 'candidate_ranks.csv', ranks)
    return aggregates
