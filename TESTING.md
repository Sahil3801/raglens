# RagLens automated testing

The default suites require no API credentials, running FastAPI server, external
Qdrant server, or downloaded model weights. They use real PDF parsing, real
chunking, embedded Qdrant and the production service/API code, with controlled
embedding vectors and provider responses. HTTPX/Requests network calls are
blocked in isolated tests. Test credentials and collection paths are set inside
the test process; production `.env` files are not changed.

The evaluation tests exercise the paired runner and real RAGAS calculations with
controlled judge responses. They also replay genuine archived vectors and
validate copies of archived judge artifacts. Generated test results stay in
pytest temporary directories, never `evaluation/runs`. These are correctness
tests, **not new measurements of RAG quality**.

## Backend (PowerShell, from repository root)

Tested with Python 3.13.14. Use the existing virtual environment, or create it if
missing. The dependency installation needs network access; the default test run
does not.

```powershell
cd rag-saas-backend
# Only when the virtual environment does not exist:
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\venv\Scripts\python.exe -m pytest tests -q
```

`requirements-test.txt` includes the production dependencies, the existing
RAGAS/LC-community compatibility pins, coverage support, and the Qdrant version
required by the archived native-MMR replay. It does not change metric definitions.
Direct dependencies in `requirements.txt` are pinned to the versions the suite
passes with. GitHub Actions (`.github/workflows/ci.yml`) runs the backend and
frontend checks on every push and pull request.

Run with coverage, including namespace-package files that have no tests:

```powershell
.\venv\Scripts\python.exe -m pytest tests --cov=app --cov=evaluation --cov-branch --cov-report=term-missing --cov-report=json:tests/.artifacts/coverage.json --cov-report=xml:tests/.artifacts/coverage.xml --junitxml=tests/.artifacts/junit.xml -q
```

Two optional tests use real cached HuggingFace weights. They require neither
Groq nor Qdrant credentials, and explicitly disallow downloads. Missing cached
weights cause a failure when these tests are explicitly enabled.

```powershell
.\venv\Scripts\python.exe -m pytest tests --run-models -q
```

Required cached revisions:

| Model | Revision |
| --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |

Two optional live refusal tests use the real Groq generator. These require an
explicitly supplied **test** key and available model/quota. They do not silently
use the production `.env` key, and are skipped by default.

```powershell
# Set TEST_GROQ_API_KEY securely in this shell before running.
$env:TEST_GROQ_MODEL = 'openai/gpt-oss-20b'
.\venv\Scripts\python.exe -m pytest tests --run-live -m live -q
```

`TEST_GROQ_API_KEY` is required for live tests; `TEST_GROQ_MODEL` is optional and
defaults to the model above. Prompt-contract tests with controlled responses do
not prove live LLM refusal compliance. Production generation is unchanged and
still relies on prompt instructions rather than an enforced refusal guard.

## Frontend (PowerShell, from repository root)

Tested with Node 22.17.1. Use `npm ci` to reproduce the committed lockfile.

```powershell
cd rag-saas-frontend
npm.cmd ci
npm.cmd test
npm.cmd run test:coverage
npm.cmd run lint
npm.cmd run build
```

Vitest uses jsdom and React Testing Library. Fetch is controlled at the network
boundary; component tests exercise the real API helper, request construction,
selection state, upload retry, deletion and error rendering. Unit tests cover
both the default API URL and `VITE_API_BASE_URL`. No environment variables are
required for tests or the default build. Set `VITE_API_BASE_URL` before a build
only when targeting a different backend URL.

Coverage measures `src/App.tsx` and `src/api.ts`; it does not imply browser-engine,
CSS, accessibility, or full browser-to-server coverage. Vite/TypeScript production
builds verify bundling separately. Generated coverage/build files are ignored by
Git and ESLint.

See [the Phase 4 report](docs/testing/phase4-2026-08-31.md) for observed results,
regressions fixed, test boundaries, and remaining gaps.
