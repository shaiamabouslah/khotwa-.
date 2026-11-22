# Khotwa

Khotwa is a bilingual French/Arabic retrieval-augmented generation (RAG) assistant for Tunisian administrative procedures and services. The project collects official and institutional sources, normalizes them into a reusable corpus, builds a curated knowledge base, and answers user questions with evidence-backed citations.

The repository combines:

- a crawler for discovering and syncing source pages,
- a cleaning pipeline for deduplicating and classifying corpus documents,
- a knowledge-base preparation step for RAG-ready chunks,
- retrieval and answer-generation scripts,
- MLflow-based evaluation tracking for retrieval and generation quality.

## What the project does

Khotwa is designed to help users find trustworthy information on Tunisian procedures such as:

- studies and academic mobility,
- scholarships and equivalence,
- registration and administrative formalities,
- employment and social security procedures,
- business and professional life requirements.

The system aims to:

- answer from official or institutional source material,
- cite the underlying source pages,
- refuse to answer confidently when evidence is weak or absent,
- keep a transparent trace of the documents that informed a response.

## Repository layout

```text
khotwa-repo/
├── README.md
├── CLEANING_AND_MLFLOW_TRACKING.md
├── PROJECT_PIPELINE_AND_MODEL_CHOICES.md
├── config.yaml
├── sources.yaml
├── requirements.txt
├── requirements-evaluation.txt
├── requirements-js.txt
├── crawler/
│   ├── README.md
│   ├── __main__.py
│   ├── pipeline.py
│   └── ...
├── scripts/
│   ├── clean_pipeline.py
│   ├── generate.py
│   ├── rag_prepare.py
│   ├── retriever.py
│   └── ...
├── kb/
│   ├── taxonomy.yaml
│   ├── studies/
│   └── employment/
├── data/
│   ├── corpus/
│   ├── rag/
│   └── state/
├── tests/
├── mlruns/
├── mlflow.db
└── .env
```

## High-level workflow

```text
sources.yaml + kb/taxonomy.yaml
        |
        v
 crawler discovers and syncs pages
        |
        v
 extracted corpus under data/corpus/
        |
        v
 cleaning and classification pipeline
        |
        v
 curated knowledge base under kb/
        |
        v
 RAG chunk preparation under data/rag/
        |
        v
 retrieval + answer generation with citations
        |
        v
 MLflow evaluation / tracking
```

## Core components

### 1. Crawler and corpus sync

The crawler reads site definitions from `sources.yaml` and builds a unified corpus from official sources. It discovers pages, fetches HTML/PDF/JavaScript content, extracts usable text, and records source state. The crawler also keeps a synchronization state to detect unchanged, changed, stale, missing, or removed documents.

Useful commands:

```bash
python -m crawler validate
python -m crawler discover --site cnss
python -m crawler crawl --site cnss
python -m crawler crawl
python -m crawler status
```

See `crawler/README.md` for site setup and sync behavior.

### 2. Cleaning and knowledge-base preparation

The cleaning workflow focuses on:

- fixing corpus documents,
- removing navigation stubs and low-quality entries,
- deduplicating near-identical content,
- reclassifying documents using the taxonomy,
- generating KB entries under `kb/studies/` and `kb/employment/`,
- validating and chunking the accepted content for retrieval.

Main commands:

```bash
python scripts/clean_pipeline.py
python scripts/clean_pipeline.py --apply
```

This repository includes detailed notes in:

- `CLEANING_AND_MLFLOW_TRACKING.md`
- `PROJECT_PIPELINE_AND_MODEL_CHOICES.md`

### 3. Retrieval, generation, and citation

The RAG layer loads prepared chunks from `data/rag/`, embeds the user query, retrieves relevant chunks, and reranks candidates before passing the top evidence to a generation model.

The generation flow:

- retrieves the best evidence,
- abstains when the evidence is too weak,
- answers only from the retrieved excerpts,
- returns citations to the source excerpts and source pages,
- asks for clarification when ambiguous procedure variants are present.

### 4. MLflow evaluation

The project uses MLflow to track experiments around chunking, retrieval quality, and optional generation evaluation. The tracking database is stored at `mlflow.db` and the default experiment is `khotwa-chunking`.

Evaluation focuses on retrieval quality and, optionally, generation/answer quality using RAGAS-style metrics.

## Quick start

### Prerequisites

- Python 3.10+
- a virtual environment is recommended
- access to the project dependencies in `requirements.txt`

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For JavaScript-rendered sources, also install the extra dependencies:

```bash
pip install -r requirements-js.txt
playwright install chromium
```

### Validate the source configuration

```bash
python -m crawler validate
```

### Crawl a specific site

```bash
python -m crawler discover --site cnss
python -m crawler crawl --site cnss
```

### Prepare the cleaned KB and RAG chunks

```bash
python scripts/clean_pipeline.py --apply
```

## Documentation

This repository currently includes the following references:

- `PROJECT_PIPELINE_AND_MODEL_CHOICES.md` — project purpose, pipeline stages, model choice notes
- `CLEANING_AND_MLFLOW_TRACKING.md` — cleaning workflow and MLflow evaluation process
- `crawler/README.md` — crawler setup, commands, and source management
- `kb/README.md` — knowledge-base documentation

## Notes

- The repository stores source-specific rules and site configuration in `sources.yaml`.
- The taxonomy and classification vocabulary live in `kb/taxonomy.yaml`.
- Generation requires a valid API key environment variable such as `GROQ_API_KEY`.
- Keep secrets out of tracked configuration files; use a local `.env` file when needed.

## Status

This project is a research and production-oriented knowledge pipeline for Tunisian administrative information retrieval. It is designed to keep data traceable, source-grounded, and evaluable across crawl, cleaning, retrieval, and generation stages.
