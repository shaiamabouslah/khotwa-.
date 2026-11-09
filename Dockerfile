# Serves the Khotwa assistant (retriever + Groq generation) in a reproducible
# environment -- decoupled from whatever happens to be installed on a given
# machine. Build context must already have data/rag/ and kb/ populated
# (dvc pull, or a local prepare_kb run) since this image doesn't fetch them.
FROM python:3.11-slim

WORKDIR /app

# System libs sentence-transformers/torch need to load models at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-rag.txt ./
# torch's default PyPI wheel drags in ~1.9GB of CUDA libs (cudnn, cublas, nccl...)
# even though this image only ever runs the embedding/reranker models on CPU.
# The CPU-only wheel (~196MB) is pre-downloaded into wheels/ (see README note
# below) instead of fetched here -- on an unstable connection, pip restarts a
# failed download from scratch, while a browser/download manager can resume.
# Installing the local wheel first satisfies every other package's
# "torch==2.14.1" requirement without pip trying to replace it afterwards.
COPY wheels/ wheels/
RUN pip install --no-cache-dir wheels/torch-2.14.1+cpu-*.whl && \
    rm -rf wheels && \
    pip install --no-cache-dir --default-timeout=120 -r requirements.txt -r requirements-rag.txt

COPY crawler/ crawler/
COPY scripts/ scripts/
COPY kb/ kb/
COPY data/rag/ data/rag/

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "scripts/generate.py"]
CMD ["--chat"]
