# Serves the Khotwa web app (FastAPI + static UI over the retriever and Groq generation) in a
# reproducible environment. Build context must already have data/rag/ and kb/ populated (dvc pull)
# and the CPU torch wheel in wheels/ (see below).
#
#   docker build -t khotwa .
#   docker run --rm -p 8000:8000 --env-file .env \
#     -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" \
#     -v khotwa-state:/app/state khotwa
#
# --env-file: GROQ_API_KEY (+ SESSION_SECRET) -- never baked into the image.
# HF cache mount: reuses the models already downloaded on the host -- the container then starts
#   in seconds and loads them offline; without it, the first start downloads BGE-M3 (~2GB).
# khotwa-state volume: accounts and conversations survive container restarts.
# The command-line assistant is still available: docker run -it ... khotwa python scripts/generate.py --chat
FROM python:3.11-slim

WORKDIR /app

# System libs sentence-transformers/torch need to load models at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# torch's default PyPI wheel drags in ~1.9GB of CUDA libs (cudnn, cublas, nccl...)
# even though this image only ever runs the embedding/reranker models on CPU.
# The CPU-only wheel (~196MB) is pre-downloaded into wheels/ instead of fetched here --
# on an unstable connection, pip restarts a failed download from scratch, while a
# browser/download manager can resume. Bind-mounted for this one step only (a COPY would keep
# the 196MB wheel in the image as its own layer). Its own layer, so editing the requirements
# below never reinstalls it; installed first, it satisfies the "torch==2.14.1" pin afterwards.
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    pip install --no-cache-dir /tmp/wheels/torch-2.14.1+cpu-*.whl

# Only what the running assistant imports (requirements-serve.txt) -- not the crawler's
# scraping libraries nor the evaluation stack (ragas, langchain...), which the server never uses.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir --default-timeout=120 -r requirements-serve.txt

COPY scripts/ scripts/
COPY static/ static/
COPY kb/ kb/
COPY data/rag/ data/rag/

ENV PYTHONUNBUFFERED=1 \
    KHOTWA_STATE_DIR=/app/state
VOLUME /app/state
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
