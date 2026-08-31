"""Retry failed provider metrics into a NEW run version; never rescore successes.

Candidates, answers, configurations and successful judge cache entries are copied
byte-for-byte from the source run. This is a recovery of the same experiment, NOT
an independent repetition. The original run and every failed attempt are retained.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import math
import time
from importlib.metadata import version

from comparison_artifacts import digest, export_run, write_json
from run_comparison import BACKEND, EVAL, RecordedJudge, checkpoint, now, safe_error


async def main(args):
    from dotenv import load_dotenv
    load_dotenv(BACKEND / '.env')
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    source = args.source.resolve()
    original = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    if original['status'] != 'completed_with_failures':
        raise ValueError('Only a finished run with actual failures can be recovered.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('Invalid run ID')
    destination = EVAL / 'runs' / args.run_id
    if destination.exists():
        raise ValueError('Destination exists; original results cannot be overwritten.')
    if not os.getenv('GROQ_API_KEY'):
        raise ValueError('GROQ_API_KEY is required')
    for package in ['ragas', 'openai', 'instructor', 'sentence-transformers', 'torch', 'langchain-community']:
        assert version(package) == original['packages'][package], package
    queries = json.loads((source / 'per_query.json').read_text(encoding='utf-8'))
    failures = [(q, arm, metric) for q in queries for arm, result in q['arms'].items()
                for metric, score in result['metrics'].items() if score['status'] == 'failed']
    if not failures:
        raise ValueError('No failed metrics to retry. Successful/undefined scores are never retried.')
    # Recovery currently handles only explicit throttling; do not hide code errors.
    for q, arm, name in failures:
        error = q['arms'][arm]['metrics'][name].get('error', '')
        if '429' not in error and 'RateLimit' not in error:
            raise ValueError(f'Non-rate-limit failure requires investigation: {q["id"]} {arm} {name}')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        'verification.json', 'judge_disagreements.json', 'REPORT.md'))
    metadata = dict(original)
    metadata.update(run_id=args.run_id, status='running', recovery_started_at=now(),
                    recovery={'parent_run_id': original['run_id'],
                              'parent_metadata_sha256': digest((source / 'metadata.json').read_bytes()),
                              'kind': 'Retry failed rate-limited metrics only; same stored answers/candidates, not an independent run.',
                              'script_sha256': digest(Path(__file__).read_bytes()), 'retry_limit': 4})
    write_json(destination / 'metadata.json', metadata)
    import torch
    torch.set_num_threads(original['torch_threads'])
    torch.manual_seed(original['seed'])
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, ContextPrecisionWithReference, AnswerRelevancy
    from ragas.embeddings import HuggingFaceEmbeddings
    judge_config = metadata['judge']
    client = AsyncOpenAI(api_key=os.environ['GROQ_API_KEY'], base_url='https://api.groq.com/openai/v1',
                         timeout=judge_config['timeout_seconds'], max_retries=judge_config['transport_max_retries'])
    recorder = None
    try:
        judge = llm_factory(judge_config['model'], client=client, **judge_config['resolved_model_args'])
        assert judge.model_args == judge_config['resolved_model_args']
        recorder = RecordedJudge(judge, destination / 'judge_calls', judge_config)
        metrics = {'faithfulness': Faithfulness(llm=judge),
                   'context_precision': ContextPrecisionWithReference(llm=judge)}
        if any(name == 'answer_relevancy' for _, _, name in failures):
            embedding = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2', revision=metadata['embedding']['revision'])
            metrics['answer_relevancy'] = AnswerRelevancy(llm=judge, embeddings=embedding,
                                                         strictness=metadata['metrics']['answer_relevancy_strictness'])
        daily_quota_exhausted = False
        attempted = []
        for q, arm, name in failures:
            attempted.append({'query_id': q['id'], 'arm': arm, 'metric': name})
            result = q['arms'][arm]
            history = [dict(result['metrics'][name])]
            for attempt in range(1, 5):
                recorder.active_keys = []
                start = time.monotonic()
                score = {'value': None, 'status': 'pending'}
                try:
                    kwargs = {'user_input': q['question']}
                    if name == 'context_precision':
                        kwargs.update(reference=q['reference'], retrieved_contexts=result['contexts'])
                    else:
                        kwargs['response'] = result['answer']
                        if name == 'faithfulness':
                            kwargs['retrieved_contexts'] = result['contexts']
                    value = float((await metrics[name].ascore(**kwargs)).value)
                    score.update(value=value if math.isfinite(value) else None,
                                 status='ok' if math.isfinite(value) else 'undefined')
                except Exception as exc:
                    score.update(status='failed', error=safe_error(exc))
                score.update(elapsed_seconds=time.monotonic() - start, judge_call_keys=list(recorder.active_keys))
                result['metrics'][name] = dict(score, prior_attempts=list(history))
                checkpoint(destination, q)
                print(f'RECOVERY {q["id"]} {arm} {name} attempt={attempt}: {score["status"]} {score["value"]}', flush=True)
                if score['status'] != 'failed':
                    break
                history.append(score)
                if 'tokens per day' in score.get('error', '') or '(TPD)' in score.get('error', ''):
                    daily_quota_exhausted = True
                    print('Daily provider quota is exhausted; stopping recovery without changing judge or successful scores.', flush=True)
                    break
                if attempt < 4:
                    await asyncio.sleep(min(15 * attempt, 45))
            if daily_quota_exhausted:
                break
        remaining = sum(r.get('generation_status') != 'ok' or any(s['status'] == 'failed' for s in r['metrics'].values())
                        for q in queries for r in q['arms'].values())
        metadata['status'] = 'completed' if remaining == 0 else 'completed_with_failures'
        metadata['failed_arms'] = remaining
        metadata['recovery']['targeted_metrics'] = [{'query_id': q['id'], 'arm': arm, 'metric': name} for q, arm, name in failures]
        metadata['recovery']['attempted_metrics'] = attempted
        metadata['recovery']['daily_quota_exhausted'] = daily_quota_exhausted
    except BaseException as exc:
        metadata.update(status='failed', error=safe_error(exc))
        raise
    finally:
        if recorder:
            metadata['recovery']['judge_calls'] = {'network_calls': recorder.network_calls, 'cache_hits': recorder.cache_hits}
        metadata['recovery_finished_at'] = now()
        export_run(destination, queries)
        write_json(destination / 'metadata.json', metadata)
        await client.close()
        print(f'Recovery artifacts: {destination}', flush=True)
    if metadata['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--run-id', required=True)
    asyncio.run(main(parser.parse_args()))
