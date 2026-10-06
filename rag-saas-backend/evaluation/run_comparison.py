"""Paired, isolated A/B evaluation of the unchanged production services.

No FastAPI server is needed: both arms call the same ingestion/retrieval/generation
services used by the API. A takes the first N MMR results; B invokes the actual
production reranker. Private Qdrant hooks observe, never replace, search results.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import distributions, version
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import uuid

from comparison_artifacts import digest, evidence_scores, export_run, resolve_evidence, write_json

EVAL = Path(__file__).resolve().parent
BACKEND = EVAL.parent
sys.path.insert(0, str(BACKEND))


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc):
    # Provider errors may contain request text; raw keys are never persisted.
    value = f'{type(exc).__name__}: {exc}'
    key = os.getenv('GROQ_API_KEY')
    return value.replace(key, '[REDACTED]') if key else value


def source_hashes():
    return {p.relative_to(BACKEND).as_posix(): digest(p.read_bytes())
            for folder in [BACKEND / 'app', EVAL]
            for p in sorted(folder.rglob('*.py')) if '.work' not in p.parts}


def checkpoint(run_dir, query):
    write_json(run_dir / 'queries' / (query['id'] + '.json'), query)


class RecordedJudge:
    """Share identical judge inputs across arms; archive genuine structured outputs.

    This wraps the real InstructorLLM.agenerate method, retaining RAGAS validation
    and metric implementations. Keys contain the entire prompt, schema and judge
    settings. No success is invented when a provider or schema validation fails.
    """
    def __init__(self, llm, directory, configuration):
        self.original = llm.agenerate
        self.directory = directory
        self.configuration = configuration
        self.active_keys = []
        self.network_calls = 0
        self.cache_hits = 0
        llm.agenerate = self.generate

    async def generate(self, prompt, response_model, **kwargs):
        request = {'prompt': prompt, 'response_schema': response_model.model_json_schema(),
                   'judge': self.configuration, 'kwargs': kwargs}
        key = digest(request)
        path = self.directory / (key + '.json')
        self.active_keys.append(key)
        if path.exists():
            self.cache_hits += 1
            return response_model.model_validate(json.loads(path.read_text(encoding='utf-8'))['response'])
        self.network_calls += 1
        start = time.monotonic()
        result = await self.original(prompt, response_model, **kwargs)
        completion = getattr(result, '_raw_response', None)
        metadata = {}
        if completion is not None:
            metadata = {k: getattr(completion, k, None) for k in ['id', 'model', 'created', 'system_fingerprint']}
            usage = getattr(completion, 'usage', None)
            metadata['usage'] = usage.model_dump() if usage else None
        write_json(path, {'request': request, 'response': result.model_dump(),
                          'provider_metadata': metadata, 'completed_at': now(),
                          'elapsed_seconds': time.monotonic() - start})
        return result


async def main(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND / '.env')
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.chdir(BACKEND)
    if not os.getenv('GROQ_API_KEY'):
        raise RuntimeError('GROQ_API_KEY is required; no mock or placeholder results will be produced.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('run-id must contain only letters, digits, underscores, or hyphens')
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if len({q['id'] for q in manifest['questions']}) != len(manifest['questions']):
        raise ValueError('Duplicate query IDs')
    run_dir = EVAL / 'runs' / args.run_id
    work_dir = EVAL / '.work' / args.run_id
    if run_dir.exists() or work_dir.exists():
        raise ValueError('Run/work directory already exists; choose a new run-id. Existing results are immutable.')
    for entry in manifest['documents']:
        path = manifest_path.parent / entry['path']
        if digest(path.read_bytes()) != entry['sha256']:
            raise ValueError(f'Corpus hash mismatch: {path.name}')
    # Environment is set before any app import, never mutating production .env.
    os.environ['QDRANT_PATH'] = str(work_dir / 'qdrant')
    os.environ['QDRANT_COLLECTION'] = 'raglens_eval_' + args.run_id
    os.environ['GROQ_MODEL'] = args.generation_model
    run_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    seed = manifest['seed']
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    from app.core.config import settings
    from app.repositories.vector_store import VectorStoreRepository
    from app.services.ingestion import IngestionService
    from app.services.retrieval import RetrievalService
    from app.services.reranking import reranker_service
    from app.services.generation import generation_service
    from fastapi import UploadFile
    from langchain_core.callbacks import BaseCallbackHandler
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.metrics.collections import Faithfulness, ContextPrecisionWithReference, AnswerRelevancy

    configuration = {k: getattr(settings, k) for k in ['CHUNK_SIZE', 'CHUNK_OVERLAP', 'EMBEDDING_DIMENSION',
                         'RETRIEVER_K', 'RETRIEVER_FETCH_K', 'RERANKER_TOP_N', 'GROQ_MODEL']}
    # Refuse accidental environmental overrides of this frozen benchmark design.
    assert [configuration[k] for k in ['CHUNK_SIZE', 'CHUNK_OVERLAP', 'RETRIEVER_K', 'RETRIEVER_FETCH_K', 'RERANKER_TOP_N']] == [800, 200, 40, 60, 8]
    judge_config = {'model': args.judge_model, 'provider': 'groq_openai_compatible',
                    'temperature': 0, 'max_tokens': 4096, 'seed': seed,
                    'transport_max_retries': 2, 'timeout_seconds': 60}
    initial_hashes = source_hashes()
    rng = random.Random(seed)
    schedule = []
    for q in manifest['questions']:
        order = ['A', 'B']
        rng.shuffle(order)
        schedule.append({'query_id': q['id'], 'arm_order': order})
    packages = {d.metadata['Name']: d.version for d in distributions() if d.metadata.get('Name')}
    metadata = {'schema_version': 1, 'run_id': args.run_id, 'started_at': now(), 'status': 'running',
                'manifest_sha256': digest(manifest_path.read_bytes()), 'benchmark': manifest,
                'production_config': configuration, 'judge': judge_config,
                'generation': {'model': args.generation_model, 'temperature': 0, 'seed': 'not set in production',
                               'prompt_source_sha256': initial_hashes['app/services/generation.py']},
                'metrics': {'faithfulness': 'ragas.metrics.collections.Faithfulness',
                            'context_precision': 'ragas.metrics.collections.ContextPrecisionWithReference',
                            'answer_relevancy': 'ragas.metrics.collections.AnswerRelevancy',
                            'answer_relevancy_strictness': 1, 'undefined_values': 'null; never imputed as zero'},
                'qdrant': {'mode': 'embedded_local', 'collection': settings.QDRANT_COLLECTION,
                           'distance': 'Cosine', 'mmr_lambda_default': 0.5, 'point_ids': 'deterministic UUID5 from chunk identities'},
                'packages': packages, 'python': sys.version, 'platform': platform.platform(),
                'device': 'cpu', 'torch_threads': 4, 'seed': seed, 'schedule': schedule,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': initial_hashes,
                'limitations': manifest['limitations'] + ['Groq hosted model weights/backends cannot be pinned; temperature zero is not a determinism guarantee.',
                    'B includes the production score threshold and context-count floor; this is the reranker policy effect, not isolated sorting alone.',
                    'Embedded Qdrant exact search does not measure Qdrant Cloud ANN approximation behavior.']}
    write_json(run_dir / 'metadata.json', metadata)
    write_json(run_dir / 'manifest.json', manifest)
    (run_dir / 'environment.freeze.txt').write_text('\n'.join(f'{k}=={v}' for k, v in sorted(packages.items(), key=lambda t:t[0].lower())) + '\n', encoding='utf-8')
    repo = None
    client = None
    queries = []
    try:
        client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'], base_url='https://api.groq.com/openai/v1', timeout=60, max_retries=2)
        available = {m.id for m in (await client.models.list()).data}
        for model in [args.generation_model, args.judge_model]:
            if model not in available:
                raise ValueError(f'Model unavailable to this account: {model}; specify an available model explicitly.')
        repo = VectorStoreRepository()
        chunks = []
        original_add = repo.add_documents

        def record_chunks(documents):
            for ordinal, doc in enumerate(documents):
                identity = {'file': doc.metadata['source_file'], 'page': doc.metadata['page'],
                            'ordinal_in_file': ordinal, 'text_sha256': digest(doc.page_content.encode('utf-8'))}
                chunk_id = digest(identity)
                doc.id = str(uuid.uuid5(uuid.NAMESPACE_URL, 'raglens:' + chunk_id))
                doc.metadata['eval_chunk_id'] = chunk_id
                chunks.append({'chunk_id': chunk_id, 'point_id': doc.id, 'identity': identity,
                               'text': doc.page_content, 'metadata': dict(doc.metadata)})
            # Same production add_documents and embeddings; only IDs/audit metadata added.
            original_add(documents)

        repo.add_documents = record_chunks
        ingestion = IngestionService(repo)
        for entry in manifest['documents']:
            path = manifest_path.parent / entry['path']
            with path.open('rb') as stream:
                ingestion.process_pdf_upload(UploadFile(filename=path.name, file=stream))
        assert len(chunks) == 72 and len({c['chunk_id'] for c in chunks}) == 72
        write_json(run_dir / 'chunks.json', chunks)
        vector = repo.embeddings.embed_query('RagLens actual embedding dimension check')
        assert len(vector) == 384
        points, offset = repo.client.scroll(settings.QDRANT_COLLECTION, limit=1000, with_vectors=True)
        assert offset is None and len(points) == len(chunks) and all(len(p.vector) == 384 for p in points)
        assert {str(p.id) for p in points} == {c['point_id'] for c in chunks}
        by_point = {str(p.id): p for p in points}
        np.savez_compressed(run_dir / 'stored_vectors.npz', chunk_ids=np.array([c['chunk_id'] for c in chunks]),
                            vectors=np.array([by_point[c['point_id']].vector for c in chunks]), probe_vector=np.array(vector))
        metadata['embedding'] = {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'actual_dimension': len(vector),
                                 'revision': repo.embeddings._client[0].auto_model.config._commit_hash,
                                 'document_encode_kwargs': repo.embeddings.encode_kwargs,
                                 'query_encode_kwargs': repo.embeddings.query_encode_kwargs}
        metadata['crossencoder'] = {'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                                    'revision': reranker_service.encoder.model.config._commit_hash,
                                    'threshold': 'best_score - 8.0', 'minimum_chunks': 4, 'maximum_chunks': 8}
        metadata['corpus_chunks'] = len(chunks)
        write_json(run_dir / 'metadata.json', metadata)
        print(f'INGESTED {len(chunks)} chunks; actual vectors {len(vector)} dimensions', flush=True)

        # Observe exact vector pool returned INSIDE production Qdrant MMR.
        local_collection = repo.client._client.collections[settings.QDRANT_COLLECTION]
        original_search = local_collection.search
        original_query = repo.client.query_points
        original_predict = reranker_service.encoder.predict
        observation = {}

        def observe_search(*positional, **kwargs):
            result = original_search(*positional, **kwargs)
            observation['vector_pool'] = [{'chunk_id': p.payload['metadata']['eval_chunk_id'],
                                            'vector_rank': i + 1, 'vector_score': float(p.score)} for i, p in enumerate(result)]
            observation['vector_requested'] = kwargs.get('limit')
            return result

        def observe_query(*positional, **kwargs):
            result = original_query(*positional, **kwargs)
            observation['mmr_request'] = {'k': kwargs['limit'], 'query': kwargs['query'].model_dump(mode='json'),
                                           'filter': kwargs['query_filter'].model_dump(mode='json') if kwargs.get('query_filter') else None}
            return result

        def observe_predict(pairs, **kwargs):
            scores = original_predict(pairs, **kwargs)
            observation['crossencoder_scores'] = [float(s) for s in scores]
            observation['crossencoder_pairs_sha256'] = digest(pairs)
            return scores

        local_collection.search = observe_search
        repo.client.query_points = observe_query
        reranker_service.encoder.predict = observe_predict
        documents_by_query = {}
        for question in manifest['questions']:
            observation.clear()
            docs = RetrievalService(repo).retrieve(question['question'], question['filter_filename'])
            a = docs[:settings.RERANKER_TOP_N]
            # Arm B reproduces the archived 2026-08-31 reranker input (plain chunk text),
            # not the later file-label headers enabled in production.
            b = reranker_service.rerank(question['question'], docs, source_headers=False)
            ids = lambda documents: [d.metadata['eval_chunk_id'] for d in documents]
            mmr_ids, a_ids, b_ids = ids(docs), ids(a), ids(b)
            scores = observation['crossencoder_scores']
            assert len(scores) == len(mmr_ids) and len(set(mmr_ids)) == len(mmr_ids)
            assert set(mmr_ids) <= {p['chunk_id'] for p in observation['vector_pool']}
            assert set(b_ids) <= set(mmr_ids)
            assert observation['crossencoder_pairs_sha256'] == digest([[question['question'], d.page_content] for d in docs])
            groups = resolve_evidence(question, chunks)
            gold = set().union(*map(set, groups)) if groups else set()
            sorted_ids = [cid for cid, score in sorted(zip(mmr_ids, scores), key=lambda t:t[1], reverse=True)]
            trace = []
            for candidate in observation['vector_pool']:
                cid = candidate['chunk_id']
                trace.append(dict(candidate, mmr_rank=mmr_ids.index(cid) + 1 if cid in mmr_ids else None,
                                  crossencoder_score=scores[mmr_ids.index(cid)] if cid in mmr_ids else None,
                                  rerank_rank=sorted_ids.index(cid) + 1 if cid in sorted_ids else None,
                                  A_context_rank=a_ids.index(cid) + 1 if cid in a_ids else None,
                                  B_context_rank=b_ids.index(cid) + 1 if cid in b_ids else None,
                                  gold_relevant=cid in gold if question['answerable'] else None))
            q = dict(question, resolved_evidence_groups=groups, rank_trace=trace,
                     mmr_ids=mmr_ids, mmr_request=observation['mmr_request'],
                     vector_requested=observation['vector_requested'],
                     mmr_evidence=evidence_scores(mmr_ids, groups),
                     gold_missing_from_vector_pool=sorted(gold - {p['chunk_id'] for p in observation['vector_pool']}),
                     gold_missing_from_mmr=sorted(gold - set(mmr_ids)),
                     promoted_relevant_into_context=sorted((set(b_ids) - set(a_ids)) & gold),
                     lost_relevant_from_context=sorted((set(a_ids) - set(b_ids)) & gold), arms={})
            for arm, selected in [('A', a), ('B', b)]:
                context_ids = ids(selected)
                matched_ids = a_ids[:len(b_ids)] if arm == 'A' else b_ids
                q['arms'][arm] = {'context_ids': context_ids, 'contexts': [d.page_content for d in selected],
                                 'evidence': evidence_scores(context_ids, groups),
                                 'matched_count_context_ids': matched_ids,
                                 'matched_count_evidence': evidence_scores(matched_ids, groups)}
            documents_by_query[q['id']] = {'A': a, 'B': b}
            queries.append(q)
            checkpoint(run_dir, q)
            print(f"RETRIEVAL {q['id']} pool={len(trace)} MMR={len(docs)} A={len(a)} B={len(b)} promoted={len(q['promoted_relevant_into_context'])}", flush=True)
        local_collection.search = original_search
        repo.client.query_points = original_query
        reranker_service.encoder.predict = original_predict

        class GenerationObserver(BaseCallbackHandler):
            messages = []
            metadata = {}

            def on_chat_model_start(self, serialized, messages, **kwargs):
                self.messages = [{'type': m.type, 'content': m.content} for m in messages[0]]

            def on_llm_end(self, response, **kwargs):
                self.metadata = response.llm_output

        observer = GenerationObserver()
        generation_service.llm.callbacks = [observer]
        # Generate every answer before starting judging; references never enter generation.
        for q, item in zip(queries, schedule):
            for arm in item['arm_order']:
                result = q['arms'][arm]
                start = time.monotonic()
                try:
                    result['answer'] = generation_service.generate_answer(q['question'], documents_by_query[q['id']][arm])
                    result['generation_status'] = 'ok'
                    result['generation_messages'] = observer.messages
                    result['generation_metadata'] = observer.metadata
                    result['exact_refusal'] = result['answer'].strip() == 'I cannot answer this based on the provided document.'
                except Exception as exc:
                    result.update(generation_status='failed', generation_error=safe_error(exc))
                result['generation_seconds'] = time.monotonic() - start
                checkpoint(run_dir, q)
                print(f"GENERATE {q['id']} {arm}: {result['generation_status']}", flush=True)

        judge = llm_factory(args.judge_model, client=client, temperature=0, max_tokens=4096, seed=seed)
        recorder = RecordedJudge(judge, run_dir / 'judge_calls', judge_config)
        ar_embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
        metrics = {'faithfulness': Faithfulness(llm=judge),
                   'context_precision': ContextPrecisionWithReference(llm=judge),
                   'answer_relevancy': AnswerRelevancy(llm=judge, embeddings=ar_embeddings, strictness=1)}
        metadata['judge']['resolved_model_args'] = judge.model_args
        metadata['judge']['adapter'] = type(judge).__name__
        write_json(run_dir / 'metadata.json', metadata)
        for q, item in zip(queries, schedule):
            for arm in item['arm_order']:
                result = q['arms'][arm]
                result['metrics'] = {}
                for name, metric in metrics.items():
                    score = {'value': None, 'status': 'pending', 'judge_call_keys': []}
                    recorder.active_keys = []
                    start = time.monotonic()
                    if name == 'context_precision' and not q['answerable']:
                        score.update(status='not_applicable', reason='No relevant evidence exists for an unanswerable question.')
                    elif result['generation_status'] != 'ok' and name != 'context_precision':
                        score.update(status='generation_failed')
                    else:
                        try:
                            kwargs = {'user_input': q['question']}
                            if name == 'context_precision':
                                kwargs.update(reference=q['reference'], retrieved_contexts=result['contexts'])
                            else:
                                kwargs['response'] = result['answer']
                                if name == 'faithfulness':
                                    kwargs['retrieved_contexts'] = result['contexts']
                            value = float((await metric.ascore(**kwargs)).value)
                            score.update(value=value if math.isfinite(value) else None,
                                         status='ok' if math.isfinite(value) else 'undefined')
                        except Exception as exc:
                            score.update(status='failed', error=safe_error(exc))
                    score['elapsed_seconds'] = time.monotonic() - start
                    score['judge_call_keys'] = list(recorder.active_keys)
                    result['metrics'][name] = score
                    checkpoint(run_dir, q)
                    print(f"SCORE {q['id']} {arm} {name}: {score['value']} ({score['status']})", flush=True)
        metadata['judge_calls'] = {'network_calls': recorder.network_calls, 'cache_hits': recorder.cache_hits}
        metadata['production_sources_unchanged'] = all(initial_hashes[p] == source_hashes()[p] for p in initial_hashes if p.startswith('app/'))
        failed = sum(r['generation_status'] != 'ok' or any(m['status'] == 'failed' for m in r['metrics'].values())
                     for q in queries for r in q['arms'].values())
        metadata['status'] = 'completed' if not failed else 'completed_with_failures'
        metadata['failed_arms'] = failed
        aggregates = export_run(run_dir, queries)
        for row in aggregates:
            if row['population'] == 'answerable' and row['metric'] in metrics:
                print(json.dumps(row), flush=True)
    except BaseException as exc:
        metadata.update(status='failed', error=safe_error(exc))
        raise
    finally:
        metadata['finished_at'] = now()
        write_json(run_dir / 'metadata.json', metadata)
        if queries:
            export_run(run_dir, queries)
        if repo:
            repo.client.close()
        if client:
            await client.close()
        print(f'Artifacts: {run_dir}', flush=True)
    if metadata['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=EVAL / 'corpora' / 'synthetic_stations_v1' / 'manifest.json')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--generation-model', required=True)
    parser.add_argument('--judge-model', required=True)
    asyncio.run(main(parser.parse_args()))
