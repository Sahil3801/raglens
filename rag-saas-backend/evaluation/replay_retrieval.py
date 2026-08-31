"""Replay vector search and native Qdrant MMR from archived vectors, without APIs.

This is an artifact replay, not a new retrieval experiment or model re-embedding.
The stored vectors are restored exactly, avoiding a second cosine normalization.
Private storage access is intentionally guarded by the recorded Qdrant version.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from comparison_artifacts import write_json


def main(run):
    import numpy as np
    from qdrant_client import QdrantClient, models
    run = run.resolve()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    metadata = read(run / 'metadata.json')
    assert version('qdrant-client') == metadata['packages']['qdrant-client']
    chunks = read(run / 'chunks.json')
    queries = [read(p) for p in sorted((run / 'queries').glob('*.json'))]
    assert len(queries) == len(metadata['benchmark']['questions'])
    archive = np.load(run / 'stored_vectors.npz')
    assert archive['chunk_ids'].tolist() == [c['chunk_id'] for c in chunks]
    client = QdrantClient(':memory:')
    client.create_collection('replay', vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    try:
        client.upsert('replay', points=[models.PointStruct(id=c['point_id'], vector=v.tolist(),
            payload={'page_content': c['text'], 'metadata': c['metadata']})
            for c, v in zip(chunks, archive['vectors'])])
        collection = client._client.collections['replay']
        for chunk, vector in zip(chunks, archive['vectors']):
            collection.vectors[''][collection.ids[chunk['point_id']]] = vector
        vector_calls = []
        original_search = collection.search

        def observed_search(*args, **kwargs):
            result = original_search(*args, **kwargs)
            vector_calls.append(result)
            return result

        collection.search = observed_search
        evidence = []
        for q in queries:
            vector_calls.clear()
            request = q['mmr_request']
            response = client.query_points('replay', query=models.NearestQuery.model_validate(request['query']),
                query_filter=models.Filter.model_validate(request['filter']) if request['filter'] else None,
                limit=request['k'], with_payload=True, with_vectors=True)
            assert len(vector_calls) == 1
            pool = vector_calls[0]
            assert [p.payload['metadata']['eval_chunk_id'] for p in pool] == [r['chunk_id'] for r in q['rank_trace']], q['id']
            np.testing.assert_allclose([p.score for p in pool], [r['vector_score'] for r in q['rank_trace']], atol=1e-8)
            assert [p.payload['metadata']['eval_chunk_id'] for p in response.points] == q['mmr_ids'], q['id']
            evidence.append({'query_id': q['id'], 'vector_pool_count': len(pool),
                             'mmr_count': len(response.points), 'vector_and_mmr_ranks_identical': True})
        report = {'status': 'passed', 'method': 'Exact stored-vector replay through pinned embedded Qdrant; no re-embedding or APIs.',
                  'queries': evidence}
        write_json(run / 'retrieval_replay.json', report)
        print(json.dumps(report, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
