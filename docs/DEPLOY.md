# Deploying RagLens (free)

| Part | Host | URL |
| --- | --- | --- |
| Frontend (React build) | Cloudflare Pages, project `raglens` | https://raglens.pages.dev |
| Backend (FastAPI, models, Qdrant) | Hugging Face Docker Space `Sahil3801/raglens-api` | https://sahil3801-raglens-api.hf.space |

Cloudflare Pages only serves static files, and Workers are far too small for
PyTorch and the two models (~1.5–2 GB RAM), so the backend runs on a free
Hugging Face Space instead.

## Backend: Hugging Face Space (deployed by GitHub Actions)

The workflow `.github/workflows/deploy-backend.yml` runs on every push to `main`
that changes the backend (or manually from the **Actions** tab). It builds the
image, smoke-tests it (health, API title, CORS for the Cloudflare site), then
uploads it to the Space, creating the Space on the first run. Hugging Face then
builds and starts it (first build ≈ 10 minutes).

One-time setup:

1. Hugging Face → **Settings → Access Tokens → Create new token**, type **Write**.
2. GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:
   - `HF_TOKEN` = the Hugging Face token
   - `GROQ_API_KEY` = your Groq key (copied into the Space as a secret)
3. GitHub → **Actions → Deploy backend to Hugging Face → Run workflow**.

The workflow also sets the Space variables `CORS_ORIGINS=https://raglens.pages.dev`
and `GROQ_MODEL=openai/gpt-oss-20b`. Override the Space name or allowed origin
with the repository *variables* `HF_SPACE` and `CORS_ORIGINS`.

Check: https://sahil3801-raglens-api.hf.space/health returns `{"status":"ok"}`.

## Frontend: Cloudflare Pages

Project `raglens`, connected to this repository, rebuilt on every push to `main`:

| Setting | Value |
| --- | --- |
| Root directory | `rag-saas-frontend` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| `NODE_VERSION` | `22` |
| `VITE_API_BASE_URL` | `https://sahil3801-raglens-api.hf.space` |

`VITE_API_BASE_URL` is baked in at build time: after changing it, redeploy.

## Free-tier limits

- **The Space sleeps** after about 48 hours without visitors; the next visit takes
  1–2 minutes while it wakes.
- **Uploaded PDFs are lost** when the Space restarts or sleeps, because its disk is
  not persistent. To keep them, create a free Qdrant Cloud cluster and add the
  Space secrets `QDRANT_URL` and `QDRANT_API_KEY`.
- **Everyone shares one document collection**, and anyone with the link can upload
  or delete. Don't upload private documents.
- Groq usage counts against your key's free-tier rate limits.
