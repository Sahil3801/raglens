# Phase 2 runtime verification - 2026-08-31

## Outcome

The complete application ran through React -> FastAPI -> PDF extraction -> chunking -> local HuggingFace embeddings -> embedded Qdrant -> MMR -> CrossEncoder -> Groq -> displayed answer.

This was verified with **embedded local Qdrant** and **Groq `openai/gpt-oss-20b`**, not the unavailable historical Groq model or the unreachable cloud cluster. These distinctions matter: this report does not certify Qdrant Cloud operation and does not support any resume quality-improvement numbers.

No RAGAS evaluation, baseline comparison, retrieval tuning, reranker threshold change, or generation prompt change was performed. The existing `.env` credentials were left unchanged and are not included in evidence files.

Setup and exact commands: [LOCAL_RUNTIME.md](../../LOCAL_RUNTIME.md).

## Initial failures

| Check | Observed result |
| --- | --- |
| Configured Qdrant Cloud | TLS connections reset with WinError 10054 on ports 6333 and 443. No authenticated response was obtained. Credential validity and cluster availability remain unverified. |
| Initial document listing | HTTP 500 due to the cloud connection failure. React previously showed an empty list and only logged the failure. |
| Historical Groq model | `llama-3.1-8b-instant` completion returned HTTP 404: model does not exist or is not accessible. It was absent from this account's `/models` response. |
| Initial existing test suite in cloud mode | 2 passed, 1 failed. The integration test failed at the Qdrant connection, before generation. |
| Deliberately malformed PDF before fix | Unhandled PyPDF error caused HTTP 500. In the browser this surfaced as a fetch failure because the unhandled error lacked CORS response headers. |

Groq credentials were accepted and a listed model, `openai/gpt-oss-20b`, successfully generated the real application answers below. The model was selected to restore runtime availability, not to optimize metrics.

## Verified runtime behavior

| Check | Result and evidence |
| --- | --- |
| FastAPI startup / OpenAPI | HTTP 200 at `/openapi.json`. |
| React listing against initially empty local collection | Empty list returned HTTP 200 and rendered correctly. |
| React PDF upload | Both synthetic PDFs uploaded via the actual file chooser to `POST /upload`, appeared in the list, and became selected after successful upload. |
| Modular upload route | `POST /documents/upload` also returned HTTP 200, including a filename containing spaces, `#`, and `&`. See [upload-checks.json](upload-checks.json). |
| Extraction and indexing | Two 36-page synthetic PDFs produced 36 chunks each: 72 stored points. Filename and page metadata survived; temporary upload files were removed. |
| Chunk configuration | 800-character size and 200-character overlap remained unchanged. These short-page fixtures produced chunks of 417-507 characters, so they do not prove an exact 200-character overlap at a long-page boundary. |
| Actual embeddings | A real query embedding had length **384**. All 72 inspected stored vectors had length **384**. |
| Actual MMR sequence | Real embedded Qdrant candidate search returned **60** candidates; MMR returned **40** with diversity 0.5. |
| CrossEncoder | Real `cross-encoder/ms-marco-MiniLM-L-6-v2` inference scored all **40** query/chunk pairs; **8** selected chunks reached generation. Scores and selected filenames/pages are saved in [runtime-trace.json](runtime-trace.json). |
| Single-document UI question | Aurora answer contained **AMBER-731** and **17 days**. Backend logs recorded 36 retrieved / 8 reranked / 8 LLM-context chunks. |
| Document filter exclusion | Asking for Borealis's code with Aurora selected returned the fixed refusal, even though Borealis was indexed. |
| Global / multi-document UI question | With no selected document, the answer correctly included Aurora **AMBER-731 / 17 days** and Borealis **COBALT-482 / 29 days**. Both first-page sources survived reranking. Backend logs recorded 40 / 8 / 8. |
| Irrelevant question | A France-capital question filtered to Aurora returned HTTP 200 with the fixed refusal and 8 source chunks. |
| Empty and whitespace-only API questions | Both returned HTTP 200, the fixed refusal, and 8 source chunks. This existing behavior was preserved; empty input is not rejected with HTTP 422. See [api-edge-cases.json](api-edge-cases.json). |
| Missing document filter | Returned HTTP 200, the fixed refusal, and 0 source chunks. Generation was still invoked with empty context. |
| Malformed PDF after fix | HTTP 400 with `The uploaded PDF could not be read.` and the correct CORS header. The UI displayed the error, retained the existing list, and did not select the failed filename. |
| Textless PDF after fix | HTTP 400 explaining that no extractable text was found and OCR is unsupported. No indexing was performed. |
| Persistence | Both indexed documents remained listed after stopping and restarting the backend with the same local storage directory. |
| Special-character deletion | Deleting `raglens_runtime_20260831_special # &.pdf` through React succeeded. The selected filter cleared and the two original fixtures remained. |
| Production frontend build | `npm.cmd run build` passed: 17 modules transformed; generated HTML, CSS and JavaScript bundles. |
| Production preview integration | Built assets were served using Vite preview on port 5173. A Borealis-specific question returned **COBALT-482 / 29 days** through the built frontend. |
| Deletion through production frontend | Deleting Borealis cleared selection and retained Aurora. A subsequent query filtered to the deleted document returned 0 sources and refusal. |
| Final cleanup | The remaining Aurora fixture was deleted through React. `/documents` returned `[]`; a global question returned 0 sources and refusal. The separate trace probe also finished with 0 stored points. See [deletion-checks.json](deletion-checks.json). |

The synthetic codes above are fictional fixture values, not credentials. These few successful answers and refusals are smoke checks, not evidence of general accuracy or reliable abstention across a benchmark.

## Changes made

1. Added optional `QDRANT_PATH` configuration for embedded development storage. The default remains empty/cloud mode. Local mode uses real QdrantClient storage, filters, and MMR; it is not a mock. Payload-index creation remains enabled for new cloud collections and is skipped in local mode, where indexes are unsupported/unneeded.
2. Guarded lazy repository initialization with a lock so concurrent initial React requests cannot open the same embedded store twice.
3. Added the missing `pytest-asyncio` requirement. It was already installed locally, but was not declared by the project.
4. Added a frontend API helper that checks HTTP status, displays failures, supports optional `VITE_API_BASE_URL`, and URL-encodes deletion filenames.
5. Converted unreadable-PDF errors and textless-PDF uploads to HTTP 400, with two focused regression tests. Successful-document extraction, metadata, chunking, and indexing behavior are unchanged.
6. Added example configuration, setup instructions, and opt-in synthetic-fixture/runtime-probe scripts. No existing evaluation files were edited by this phase.

The five service files that were already modified before Phase 1 were preserved. This phase adds only the PDF-error handling changes to ingestion among those files. Existing comments and whitespace in the other modified service files were not rewritten.

## Tests

| Test run | Result |
| --- | --- |
| Existing unit tests before fixes | **2 passed**. |
| Existing suite using original cloud settings | **2 passed, 1 failed** because Qdrant TLS connections were reset. |
| Existing suite using local Qdrant and accessible Groq model | **3 passed**. No service mocks or test skips were introduced into the existing integration test. |
| Final suite with two PDF-error regressions | **5 passed**, one upstream `langchain-community` deprecation warning. |
| Final frontend build | **Passed**. |
| Final frontend ESLint | **Passed**. |
| Backend `python -m pip check` | **Passed**: no broken requirements found. |

Full-working-tree `git diff --check` reports trailing whitespace in the user's pre-existing service comments/blank lines. Those unrelated edits were preserved, not silently reformatted.

## Remaining limitations

- Cloud connectivity still needs a reachable Qdrant endpoint. No existing cloud documents were uploaded, deleted, or altered. The TLS failure occurred before collection operations completed.
- The successful run used a different generation model from the historical default. It cannot reproduce historical Llama-based metrics or establish a reranking improvement.
- Embedded Qdrant is a local development option. Cloud/server performance, payload indexing, multiworker use, and concurrent production load were not verified by the embedded run.
- Listing still reads only the first 1,000 points; pagination was outside this small runtime fixture. Duplicate-upload deduplication and arbitrary multi-file selection remain absent.
- Abstention is prompt-based. Empty/whitespace input remains accepted, and even empty-context requests invoke Groq. These behaviors were observed and left unchanged.
- Upload size limits, filename/temp-file collision hardening, OCR, authentication/tenant isolation, and production-origin CORS configuration remain separate work.
- There is no controlled RAGAS experiment and no newly measured Faithfulness or Context Precision score in this report.

The backend and production preview were stopped after verification. All synthetic indexed documents were removed; existing cloud data was untouched.
