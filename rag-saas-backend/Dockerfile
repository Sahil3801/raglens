FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch keeps the image around 1 GB instead of several GB of CUDA libraries.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch==2.14.1 --index-url "$TORCH_INDEX_URL"
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Optionally bake the embedding and reranker models into the image so the
# container starts without downloading them: --build-arg PRELOAD_MODELS=1
ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; fi

RUN useradd --create-home --uid 1000 app && mkdir -p "$HF_HOME" && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One worker: the reranker and embedding models are loaded once per process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
