# Paired baseline versus reranker evaluation

This runner evaluates the existing production services; it does not tune them or
modify the live collection. Phase 4 application test expansion is out of scope.
The historical resume benchmark PDFs are unavailable. The included
`corpora/synthetic_stations_v1` corpus contains the **unchanged two synthetic PDFs
from Phase 2**, newly frozen with 16 questions, references, and exhaustive
source/page/quote labels. Its results must be described as a small synthetic
benchmark, not as verification of the old resume numbers or general PDF quality.

See the [measured results, quota limitations and resume assessment](REPORT_2026-08-31.md).

## Fixed experiment

1. Verify each versioned PDF's SHA-256 against the frozen manifest.
2. Ingest through production `IngestionService`: PyPDFLoader, recursive character
   splitting at size 800 / overlap 200, MiniLM embeddings, embedded Qdrant cosine
   storage. Add only deterministic point IDs and evaluation chunk metadata.
3. Call production MMR **once per question** (`fetch_k=60`, `k=40`, default lambda
   0.5), including the question's filename filter. Preserve this shared candidate
   list for both arms. Smaller filtered corpora legitimately yield fewer results.
4. **A:** first eight MMR results, no CrossEncoder. **B:** invoke the unmodified
   production reranker on that same MMR list: MS MARCO MiniLM L-6-v2, cutoff
   `best_score - 8`, minimum four when available, maximum eight. No score tuning.
5. Generate both answers through the exact production `GenerationService`, with
   the same Groq model, temperature zero and prompt. References never enter
   generation. The seeded arm order is saved before execution. Both arms make
   real generation calls, including when their context lists are identical.
6. Judge both arms with the same RAGAS classes, Groq model and settings. Cache
   identical judge prompt/schema/config inputs across arms, preserving the real
   structured responses. This controls avoidable judge variation for equal inputs.

The fixed generation model for the recorded run is `openai/gpt-oss-20b`; the judge
is `openai/gpt-oss-120b`, temperature 0, seed 20260831, max_tokens 4096. The legacy
default `llama-3.1-8b-instant` was unavailable to this account in Phase 2. Both arms
use the explicitly selected replacement. These are distinct models in the same
family, not independent human judges. Hosted model weights cannot be pinned, and
temperature zero does not guarantee identical answers on future runs.
RAGAS's resolved judge arguments also include its default `top_p=0.1`; this is
recorded in metadata and is identical across arms. The observed Groq judge limit
was 8,000 tokens/minute, so the live scoring pass can take tens of minutes.

## Metrics and denominators

- **Faithfulness:** actual `ragas.metrics.collections.Faithfulness` statement
  extraction and entailment. No extracted statements produces an undefined score,
  saved as JSON null, not 0 or 1. A refusal can also receive a judge-generated
  statement; retain and inspect the judge's actual result rather than overriding it.
- **Context Precision:** actual `ContextPrecisionWithReference`, with the frozen
  reference and the ordered final contexts. This is RAGAS average precision of
  judge usefulness verdicts, **not** the fraction of all contexts that are relevant.
  It cannot on its own measure recall. No relevant evidence exists for unanswerable
  questions, so their Context Precision is explicitly not applicable.
- **Answer Relevancy:** actual `AnswerRelevancy`, strictness 1, local
  `all-MiniLM-L6-v2` embeddings normalized for cosine similarity. Keep it secondary:
  semantically similar generated questions do not establish answer correctness.
- **Gold retrieval evidence:** hit, chunk precision, exhaustive chunk recall,
  reciprocal rank and binary nDCG over the final context. Evidence-group recall
  measures coverage of the required facts (each group is required; any labeled
  alternative within it suffices). All-evidence success requires every group.
  This distinction matters for the many equivalent housekeeping pages.
- **Context-count sensitivity:** compare the first `len(B)` MMR chunks to B for
  gold retrieval metrics as well. This is a retrieval-only analysis, **not** a
  third generation/RAGAS arm. Main A/B scores include the production threshold's
  context-count effect and cannot isolate CrossEncoder sorting alone.

Primary means use the 12 answerable questions. Separate aggregates include the
four unanswerable questions and all 16 questions, with exact-refusal rates.
Every metric reports each arm's valid/failed/undefined counts. Paired means and
deltas use only questions valid in both arms and disclose that count. No zero
imputation, hidden row dropping, target scores or retrospective query selection.
The questions share facts in a tiny repetitive synthetic corpus; treating them as
independent samples for significance claims would overstate the evidence.

## Run locally (PowerShell, from repository root)

```powershell
cd rag-saas-backend
# Existing Phase 2 venv may be reused. For a fresh environment:
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt -r evaluation\requirements.txt
.\venv\Scripts\python.exe -m pip check

# Set locally or put this in rag-saas-backend/.env. Never commit the key.
$env:GROQ_API_KEY = 'your-groq-key'
.\venv\Scripts\python.exe evaluation\run_comparison.py --run-id my_unique_run --generation-model openai/gpt-oss-20b --judge-model openai/gpt-oss-120b
.\venv\Scripts\python.exe evaluation\verify_comparison.py evaluation\runs\my_unique_run --check-embeddings
.\venv\Scripts\python.exe evaluation\replay_retrieval.py evaluation\runs\my_unique_run
```

For the exact recorded Python package versions, install the run's complete lock
snapshot instead of the unpinned application requirements:

```powershell
.\venv\Scripts\python.exe -m pip install -r evaluation\runs\20260831_synthetic_stations_v1_01\environment.freeze.txt
```

Use Python 3.13 on Windows to match the recorded environment. The evaluator pins
`ragas==0.4.3` and `langchain-community==0.4.1` because Community 0.4.2 removes the
VertexAI module that RAGAS imports. It does **not** import the legacy evaluator's
fake compatibility modules. The archived freeze includes transitive versions.
This machine does not have the `py` launcher on PATH. Its verified base interpreter
can instead create the environment with:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv venv
```

Required credential: **GROQ_API_KEY**. Both named Groq models must be accessible
to that key; the runner fails rather than silently changing models. No OpenAI key,
Qdrant Cloud credential, running FastAPI server, or frontend server is needed.
HuggingFace network access is required for the first model download; `HF_TOKEN`
is optional for authenticated download limits. On later runs the local cache can
be used. The resolved HuggingFace revisions are recorded; preserve those snapshots
for reproduction and check that a fresh run resolves the same revisions.

The runner sets, only in its process, `GROQ_MODEL` from the explicit argument,
`QDRANT_PATH=evaluation/.work/<run-id>/qdrant`, a unique `QDRANT_COLLECTION`,
`RAGAS_DO_NOT_TRACK=true`, and `TOKENIZERS_PARALLELISM=false`. It never edits `.env`
or a production collection. Override any locally customized `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `RETRIEVER_K`, `RETRIEVER_FETCH_K`, and `RERANKER_TOP_N` to
800/200/40/60/8 before running; the frozen benchmark rejects other values.

Each run ID must be new. Existing results cannot be overwritten or silently
resumed. A new live run makes new provider calls and records its own outputs;
an API/model failure is saved explicitly and the command exits nonzero.

For a finished run with rate-limit failures, `recover_failed_metrics.py` copies
that run into a new version and retries only failed metrics. It preserves all
answers, candidates, successful judgments and prior failures. It stops promptly
on a daily quota error; it does not change models or rescore successful values.
This is recovery of the same experiment, not an independent repetition:

```powershell
.\venv\Scripts\python.exe evaluation\recover_failed_metrics.py evaluation\runs\my_unique_run --run-id my_unique_run_recovery01
.\venv\Scripts\python.exe evaluation\verify_comparison.py evaluation\runs\my_unique_run_recovery01 --check-embeddings
```

## Saved evidence

`evaluation/runs/<run-id>/` is versionable and is **not** the ignored legacy
`evaluation/results/` directory. It contains:

- `manifest.json`, `metadata.json`, `environment.freeze.txt`: corpus/configuration,
  model revisions, platform, package versions, source hashes, seed and arm order.
- `chunks.json`, `stored_vectors.npz`: exact extracted chunks/metadata, stable
  identities, stored 384-dimensional vectors and an actual query-vector probe.
- `queries/*.json`, `per_query.json`, `per_query.csv`: questions, references,
  candidates, contexts, answers, generation prompts/provider metadata, metric
  values/statuses and gold retrieval evidence.
- `candidate_ranks.csv`: vector rank/score, MMR rank, full reranker rank/score,
  each final-context rank, and relevance labels, including dropped candidates.
- `judge_calls/*.json`: actual judge prompts, structured output schemas,
  responses, metadata and content-addressed cache keys.
- `aggregate.json`, `aggregate.csv`: macro means, paired deltas and denominators.
- `verification.json`: offline evidence/metric/aggregate replay checks.
- `retrieval_replay.json`: exact vector candidate and MMR rank replay through
  pinned embedded Qdrant using the saved vectors and query vectors, with no APIs.

`verify_comparison.py` requires no provider API calls. The optional embedding check
uses the cached recorded model revision to recompute Answer Relevancy; without it,
the validator still replays Context Precision, Faithfulness, ranking, evidence,
prompts, source hashes, JSON/CSV agreement, and aggregates. Successful verification
checks arithmetic and provenance, not whether the LLM judge's verdicts are correct.
The verifier also writes `judge_disagreements.json`, comparing Context Precision
verdicts with the frozen contributing-evidence labels. These disagreements do not
rewrite the RAGAS scores. The metadata's `judge_calls.network_calls` counts logical
cache-miss judge invocations; transport/schema retries are not separately counted.

Only a chunk present in the shared MMR candidates can be promoted. A defensible
claim names movement from an omitted baseline final-context position into the
reranked final context. It does not say the reranker recovered a chunk missing
from vector retrieval or from its input candidate set.
