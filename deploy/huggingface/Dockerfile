# Backend image for a Hugging Face Docker Space (the frontend is on Cloudflare Pages).
# The deploy workflow uploads this file with rag-saas-backend/requirements.txt and
# rag-saas-backend/app/ as the Space's root, so the paths below are relative to that.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

# CPU-only torch keeps the image around 1 GB instead of several GB of CUDA libraries.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Hugging Face Spaces run containers as user 1000.
RUN useradd --create-home --uid 1000 user && mkdir -p /app/data "$HF_HOME" && chown -R user:user /app
USER user

# Bake both models into the image so the first question does not download ~200 MB.
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user:user app ./app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

# Without QDRANT_URL, documents live in embedded Qdrant inside the container and
# are lost when the Space restarts or sleeps. Set QDRANT_URL/QDRANT_API_KEY to keep them.
CMD ["sh", "-c", "if [ -z \"$QDRANT_URL\" ]; then export QDRANT_PATH=\"${QDRANT_PATH:-/app/data/qdrant}\"; fi; exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\""]
