# Local runtime setup

Verified on Windows on 2026-08-31 with Python 3.13.14 and Node.js 22.17.1.
This guide covers application operation only. Do not interpret its smoke checks as a retrieval-quality benchmark.

## Environment variables

Backend settings load from `rag-saas-backend/.env`; shell variables take precedence.
Keep credentials in that file or the shell, never in frontend variables or committed reports.

| Variable | Requirement and behavior |
| --- | --- |
| `GROQ_API_KEY` | Required for generation and the existing live integration test. |
| `GROQ_MODEL` | Set explicitly to a model available to your account. `openai/gpt-oss-20b` completed the verification. The historical code default `llama-3.1-8b-instant` returned HTTP 404 for this account. |
| `QDRANT_PATH` | Set to a local directory for embedded Qdrant. A nonempty value selects local mode and takes precedence over cloud URL/key. Default is empty, preserving cloud mode. |
| `QDRANT_URL` | Required in cloud mode; ignored in local mode. Include the actual scheme and endpoint supplied by your cluster. |
| `QDRANT_API_KEY` | Required for authenticated cloud clusters; ignored in local mode. |
| `QDRANT_COLLECTION` | Collection name; default `my_documents`. Use an isolated `raglens_runtime_...` name for verification. |
| `HF_TOKEN` | Optional authentication for downloading the public HuggingFace models. No token is required once models are cached. The old `HUGGINGFACEHUB_API_TOKEN` entry is not explicitly passed to either local model constructor. |
| `HF_HUB_OFFLINE` | Optional shell variable: `1` uses cached HuggingFace assets without contacting the Hub. Do not enable before both models have been cached. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Defaults `800` / `200`, measured in characters. No tuning performed. |
| `EMBEDDING_DIMENSION` | Default `384`, which matches measured vectors. Do not change it independently of the embedding model/collection. |
| `EMBEDDING_MODEL` | Declared setting, but currently unused by the repository; the model remains hardcoded to `all-MiniLM-L6-v2`. |
| `RETRIEVER_K` / `RETRIEVER_FETCH_K` | Defaults `40` / `60`: return up to 40 MMR selections from up to 60 candidates. |
| `RERANKER_TOP_N` | Default `8`. Existing threshold and minimum-four fallback remain unchanged. |
| `CORS_ORIGINS` | Comma-separated browser origins allowed to call the API. Default `http://localhost:5173,http://127.0.0.1:5173`. |
| `PROJECT_NAME` | Optional API display title. |
| `VITE_API_BASE_URL` | Optional frontend build-time setting, default `http://127.0.0.1:8000`. Never put secrets in any `VITE_` variable. |

The CrossEncoder model remains `cross-encoder/ms-marco-MiniLM-L-6-v2`. Generation temperature remains zero.
No OpenAI API key is required: the verified model is served by Groq using `GROQ_API_KEY`.

## Backend: existing checkout

Run in PowerShell from the repository root:

```powershell
Set-Location rag-saas-backend
# Keep the existing .env with your credentials. Do not overwrite it.
$env:QDRANT_PATH = './qdrant_db/runtime-verification'
$env:QDRANT_COLLECTION = 'raglens_runtime_20260831_01a0568f'
$env:GROQ_MODEL = 'openai/gpt-oss-20b'
# Optional, only after the embedding and CrossEncoder models are cached:
$env:HF_HUB_OFFLINE = '1'
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

This is the command used for browser verification. API documentation: http://127.0.0.1:8000/docs.
Use one backend worker for embedded Qdrant. Do not open the same local storage directory from a second process, including pytest or a probe. Use separate paths as shown below.

## Backend: new environment

From the repository root, with Python 3.13 available:

```powershell
Set-Location rag-saas-backend
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env: set GROQ_API_KEY and choose local or cloud settings.
# Ensure GROQ_MODEL names an accessible model.
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If `py` is unavailable, use the absolute path of your Python 3.13 executable. First-time startup needs network access to download both public models. Direct dependencies are pinned in requirements.txt to the versions the test suite passes with; transitive dependencies are not locked.

## Cloud mode

Set `QDRANT_PATH=` in `.env` and remove a shell override if present:

```powershell
Remove-Item Env:QDRANT_PATH -ErrorAction SilentlyContinue
# .env must contain QDRANT_PATH=, a working QDRANT_URL, QDRANT_API_KEY,
# QDRANT_COLLECTION, GROQ_API_KEY and an accessible GROQ_MODEL.
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The supplied cloud endpoint reset TLS connections on both 6333 and 443 during verification. A successful authenticated cloud connection was not established. Cloud ingestion, filtering, server-side MMR, and deletion therefore still require verification against a reachable cluster. TLS verification was not disabled.

## Frontend

In another PowerShell terminal, from the repository root:

```powershell
Set-Location rag-saas-frontend
npm.cmd ci
# Optional; default already points to this backend:
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8000'
node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort
```

Open http://127.0.0.1:5173. `npm.cmd run dev` is also provided but automatically opens a browser.
The backend allows development origins `http://localhost:5173` and `http://127.0.0.1:5173`.

```powershell
npm.cmd run build
npm.cmd run lint
# Stop the development server before using the same port for production preview:
npm.cmd run preview -- --host 127.0.0.1 --port 5173 --strictPort
```

Preview deliberately uses port 5173, matching the existing CORS configuration. A deployed frontend on another origin must be added to `CORS_ORIGINS`. Build-time URL changes require rebuilding.

## Existing tests and ingestion regressions

Run in the backend directory, using storage distinct from the running server:

```powershell
$env:QDRANT_PATH = './qdrant_db/runtime-tests'
$env:QDRANT_COLLECTION = 'raglens_runtime_tests_20260831'
$env:GROQ_MODEL = 'openai/gpt-oss-20b'
$env:HF_HUB_OFFLINE = '1' # cached models only
.\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Result after the fixes: 5 passed (the 3 existing tests and 2 PDF-error regressions). The integration test makes a real Groq request. For unit-only execution, use `tests/unit`; these do not call Groq or Qdrant services.

## Optional observable runtime probe

This is not RAGAS and contains no baseline/reranker comparison, accuracy score, or aggregate quality metric. It uses real application services, observes embedded Qdrant's real candidate search, records dimensions/counts/metadata/model scores, and generates one answer. Its internal observation point is version-dependent and does not modify returned search results.

Run in the backend directory:

```powershell
# Only needed to generate the synthetic PDFs; not an application dependency:
.\venv\Scripts\python.exe -m pip install reportlab==4.4.9
.\venv\Scripts\python.exe scripts/create_runtime_fixtures.py --output ../tmp/runtime-verification
$env:QDRANT_PATH = './qdrant_db/runtime-probe'
$env:QDRANT_COLLECTION = 'raglens_runtime_probe_20260831'
$env:GROQ_MODEL = 'openai/gpt-oss-20b'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
.\venv\Scripts\python.exe scripts/runtime_probe.py `
  --pdf ../tmp/runtime-verification/raglens_runtime_20260831_aurora.pdf `
  --pdf ../tmp/runtime-verification/raglens_runtime_20260831_borealis.pdf `
  --query 'List the maintenance access code and inspection interval for both Aurora station and Borealis station.' `
  --output ../tmp/runtime-verification/runtime-trace.json
```

The probe refuses nonlocal storage, collection names outside its test namespace, and nonempty collections. It deletes the documents it inserts and records the remaining point count. Use synthetic inputs only because source names and the generated answer are saved. The PDF generator uses invariant PDF metadata for repeatable fixture bytes within the same ReportLab version.

For the recorded run, fixture generation used a separate Python installation with ReportLab 4.4.9, not an installation into the application virtual environment. The probe itself used the application's Python 3.13.14 and installed application dependencies.
