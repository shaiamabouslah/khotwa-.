# Khotwa

Khotwa is a French/Arabic/English retrieval-augmented generation (RAG) assistant for Tunisian administrative procedures and services. The project collects official and institutional sources, normalizes them into a reusable corpus, builds a curated knowledge base, and answers user questions with evidence-backed citations, through a web app with accounts and conversation history.

The repository combines:

- a crawler for discovering and syncing source pages,
- a cleaning pipeline for deduplicating and classifying corpus documents,
- a knowledge-base preparation step for RAG-ready chunks and embeddings,
- retrieval and answer generation (BGE-M3 dense retrieval + a Groq-hosted LLM),
- a web app (FastAPI + a single static page) with usage monitoring,
- MLOps tooling: DVC for data and pipeline versioning, Docker for the served app, GitHub Actions for tests, RAGAS for answer-quality evaluation.

## What the project does

Khotwa is designed to help users find trustworthy information on Tunisian procedures such as:

- studies and academic mobility,
- scholarships and equivalence,
- registration and administrative formalities,
- employment and social security procedures (CNSS),
- business and professional life requirements.

The system aims to:

- answer only from official or institutional source material,
- cite the underlying source pages (clickable source buttons in the web app),
- refuse to answer when evidence is weak or absent, instead of guessing,
- ask a clarification question when a procedure has several variants (e.g. CNSS sector or employer/employee),
- keep a transparent trace of the documents that informed a response.

## Repository layout

```text
.
├── README.md
├── sources.yaml              crawler site definitions
├── requirements.txt          crawler + cleaning
├── requirements-js.txt       extra, for JavaScript-rendered sources (Playwright)
├── requirements-serve.txt    what the running assistant needs (web app, CLI chat) -- the Docker image
├── requirements-rag.txt      everything above + evaluation (RAGAS)
├── Dockerfile                image of the web app
├── dvc.yaml / dvc.lock       data pipeline (DVC)
├── crawler/                  discovery, fetching, extraction, sync state
├── scripts/
│   ├── api.py                web app backend (FastAPI)
│   ├── auth.py               local accounts (salted PBKDF2 hashes)
│   ├── monitoring.py         usage events + metrics
│   ├── grounding.py          checks every figure, link and citation of an answer against its sources
│   ├── generate.py           answer generation, clarification, conversation history
│   ├── retriever.py          BGE-M3 retrieval, embeddings cache + fingerprint
│   ├── rag_prepare.py        kb/ -> chunks
│   ├── clean_pipeline.py     corpus -> kb/
│   ├── validate_kb.py        kb/ checks against kb/taxonomy.yaml
│   ├── eval_ragas.py         answer-quality evaluation
│   ├── compare_chunking.py   chunking-strategy comparison
│   ├── benchmark_retrieval.py retrieval benchmark (DVC metrics)
│   ├── check_metrics.py      quality gate on the benchmark (used by CI)
│   ├── eval/                 BM25 / dense scorers used by compare_chunking, score-floor calibration
│   └── fixes/                document-level cleaning fixes run by clean_pipeline.py
├── static/                   index.html (chat UI), metrics.html (usage statistics)
├── kb/                       curated knowledge base (taxonomy.yaml, studies/, employment/)
├── data/
│   ├── corpus/  raw/  state/ crawler outputs (versioned with DVC)
│   └── rag/                  chunks, embeddings, benchmark questions, evaluation results
├── tests/                    unit tests (no model is loaded)
└── .env                      local secrets, never committed
```

## High-level workflow

```text
sources.yaml + kb/taxonomy.yaml
        |
        v
 crawler discovers and syncs pages          python -m crawler crawl
        |
        v
 extracted corpus under data/corpus/
        |
        v
 cleaning and classification pipeline       python scripts/clean_pipeline.py --apply
        |
        v
 curated knowledge base under kb/
        |
        v
 chunks + BGE-M3 embeddings in data/rag/    dvc repro prepare_kb embed_kb
        |
        v
 retrieval + answer generation with citations (web app / CLI)
        |
        v
 evaluation: retrieval benchmark, RAGAS, usage monitoring
```

## Core components

### 1. Crawler and corpus sync

The crawler reads site definitions from `sources.yaml` and builds a unified corpus from official sources. It discovers pages, fetches HTML/PDF/JavaScript content, extracts usable text, and records source state. The crawler also keeps a synchronization state to detect unchanged, changed, stale, missing, or removed documents.

```bash
python -m crawler validate
python -m crawler discover --site cnss
python -m crawler crawl --site cnss
python -m crawler crawl
python -m crawler status
```

See `crawler/README.md` for site setup and sync behavior.

### 2. Cleaning and knowledge-base preparation

The cleaning workflow fixes corpus documents, removes navigation stubs and low-quality entries, deduplicates near-identical content, reclassifies documents using the taxonomy, and writes KB entries under `kb/studies/` and `kb/employment/`.

```bash
python scripts/clean_pipeline.py            # dry run: report what would change
python scripts/clean_pipeline.py --apply
python scripts/validate_kb.py
```

See `kb/README.md` for the knowledge-base format.

### 3. Retrieval, generation, and citation

`scripts/retriever.py` embeds the question with BGE-M3 and retrieves the closest chunks from `data/rag/`. Results under a similarity floor are dropped, so a question with no relevant source gets an honest "no information" reply instead of an invented one. `scripts/generate.py` then:

- answers in the language of the question (French, Arabic or English), only from the retrieved sources,
- cites them as [1], [2] and lists the official links,
- asks which variant applies when the sources describe several (CNSS sector, employer / employee / self-employed),
- keeps short-term conversation history so follow-up questions work.

Generation uses the Groq API and needs `GROQ_API_KEY` in `.env`.

### 4. Web app and monitoring

`scripts/api.py` serves the chat page with accounts, a conversation sidebar and feedback buttons (useful / not useful). Every question is logged (`data/monitoring/events.jsonl`: outcome, language, latency, no password or raw user name) and summarized on `/metrics.html`: answer rate, unanswered rate, latency p50/p95, satisfaction, and the most recent unanswered or down-voted questions.

Every generated answer also goes through `scripts/grounding.py`, a check that needs no model or API call: each figure (amount, date, age, phone number), each link and each `[n]` citation must appear in the sources the LLM was given. `/metrics.html` shows the share of answers that fail it ("Non vérifiées") and what was missing from the sources -- a live, explainable hallucination signal for the facts that matter most in an administrative procedure. It cannot judge a paraphrased claim; that is what RAGAS faithfulness measures.

### 5. Data versioning and pipeline (DVC)

Large data (`data/raw/`, `data/corpus/`, the chunks and the embeddings) is versioned with DVC, not git. `dvc.yaml` declares the pipeline stages:

- `prepare_kb` -- kb/ to chunks,
- `embed_kb` -- chunks to BGE-M3 embeddings, plus a fingerprint file the retriever checks at load time (stale embeddings are detected and rebuilt, never silently used),
- `compare_chunking` -- compares the heading / fixed / semantic / adaptive chunking strategies,
- `benchmark_retrieval` -- retrieval quality as the app does it, written as DVC metrics,
- `eval_ragas` -- answer-quality evaluation (uses Groq quota, run on demand).

Stages call `python`, so run DVC with the virtual environment activated -- otherwise another
Python installation, with other package versions, may run them.

The default DVC remote is DagsHub (`dagshub`, https://dagshub.com/shaimabouslah/khotwa-.); the earlier local folder remote (`localstorage`) is kept as a backup. Each user stores their own DagsHub token, outside git:

```bash
dvc remote modify dagshub --local user <dagshub-user>
dvc remote modify dagshub --local password <dagshub-token>
dvc pull                     # fetch the data for the current commit
dvc repro prepare_kb embed_kb
dvc push                     # after changing data
```

`kb/overrides.yaml` holds curated additions to the generated KB files (Arabic titles, caveat notes), kept apart because `clean_pipeline.py --apply` rewrites those files; `validate_kb.py` checks every entry points to an existing document.

### 6. Evaluation

- **Retrieval benchmark** -- `scripts/benchmark_retrieval.py` on `data/rag/questions.jsonl`: 94 questions (32 French, 27 Arabic, 22 English, 13 Tunisian Derja in Arabic and Latin script) with the document that answers each one, plus `data/rag/hard_questions.jsonl` (12 typo, keyword-style and slang variants, reported as "robustness"). Measures hit@1/3/5 and MRR per group, with a 95% bootstrap interval on the overall scores, for dense retrieval (used by the app) and the hybrid dense+BM25 mode. The run stops if a label points to a document that is no longer in the index (two stale labels had been counted as retrieval misses). Results go to `data/rag/metrics/retrieval.json`, tracked as DVC metrics.
- **Refusal check** -- `data/rag/negative_questions.jsonl`: 30 questions the assistant must decline (out of domain, or related procedures it has no source for), in French, Arabic, English and Derja. The benchmark reports how many the score floor stops before any LLM call; the rest are declined by the generation prompt.
- **Quality gate** -- `scripts/check_metrics.py` fails when dense hit@5, hit@1 or MRR fall under fixed floors, or when any group's hit@5 drops under 0.70. The floors sit below the current scores by more than the 95% interval, so noise does not fail a build.

```bash
dvc repro benchmark_retrieval
dvc metrics show
dvc metrics diff main        # compare the working copy against main
```

Current results (dense, as used by the app):

| Group | n | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| All | 94 | 0.85 (95% CI 0.78-0.93) | 0.99 (0.97-1.00) | 0.90 |
| French | 32 | 0.81 | 1.00 | 0.89 |
| Arabic | 27 | 0.89 | 1.00 | 0.92 |
| English | 22 | 0.91 | 1.00 | 0.94 |
| Derja | 13 | 0.77 | 0.92 | 0.81 |
| Robustness | 12 | 0.75 | 0.92 | 0.81 |

The Arabic titles of French-only documents (`kb/overrides.yaml`, see Known limitations) raised hit@1 from 0.77 to 0.85 and Arabic hit@5 from 0.93 to 1.00, without changing French. The LLM receives the top 5 documents, so hit@5 says whether a correct answer is possible; hit@1 says how cleanly the right source comes first. Hybrid dense+BM25 scores lower overall (hit@1 0.64, hit@5 0.89), so dense stays the default. The score floor stops 13% of the refusal questions before any LLM call: in-scope Arabic and Derja questions score as low as out-of-scope ones, so the floor only removes the clearly absurd and the generation prompt does the rest. The four chunking strategies tie on the 94 questions (dense hit@1 0.84-0.85, hit@5 0.99 for all, `data/rag/chunking_comparison_report.txt`), so the app keeps heading-based chunking.
- **Answer quality** -- `scripts/eval_ragas.py` runs the full pipeline exactly as the web app does and scores each answer with RAGAS faithfulness (is every claim supported by the sources -- the hallucination check) and context relevance (did retrieval bring the right material), per language.

```bash
python scripts/eval_ragas.py --sleep 25 --resume   # paced for the Groq free tier; spans several days of its daily quota
```

## Quick start

### Install

```bash
python -m venv .venv
.venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements-rag.txt
dvc pull
```

Create a `.env` file at the repository root:

```text
GROQ_API_KEY=...
SESSION_SECRET=...           # any long random string; keeps users logged in across restarts
```

For JavaScript-rendered sources, also install:

```bash
pip install -r requirements-js.txt
playwright install chromium
```

### Run the web app

```bash
python -m uvicorn scripts.api:app --port 8000
```

Then open http://localhost:8000. The first start downloads the BGE-M3 model (about 2 GB) into the Hugging Face cache.

Command-line chat instead: `python scripts/generate.py --chat`

### Run with Docker

```bash
docker build -t khotwa .
docker run --rm -p 8000:8000 --env-file .env -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" -v khotwa-state:/app/state khotwa
```

The image contains only the serving dependencies. The Hugging Face cache mount reuses models already downloaded on the host; the `khotwa-state` volume keeps accounts, conversations and usage logs across restarts. Secrets are passed at run time, never baked into the image. The CPU-only torch wheel must be in `wheels/` before building (see the Dockerfile).

### Run the tests

```bash
python -m unittest discover -s tests -t .
```

The same tests run on every push and pull request to `main` (GitHub Actions, `.github/workflows/tests.yml`), together with `python -m crawler validate` and `python scripts/validate_kb.py`. They test the deterministic logic only -- thresholds, caches, history, clarification, role ordering, cleaning, grounding check, quality gate, API behaviour -- and never load the embedding model, so they take seconds.

A second workflow, `.github/workflows/pipeline.yml`, needs the data: it pulls the chunks and embeddings from the DagsHub DVC remote, runs the retrieval benchmark, applies the quality gate and builds the Docker image. It needs two repository secrets, `DAGSHUB_USER` and `DAGSHUB_TOKEN`; without them it reports a notice and skips.

## Notes

- Source-specific rules and site configuration live in `sources.yaml`.
- The taxonomy and classification vocabulary live in `kb/taxonomy.yaml`.
- Keep secrets out of tracked files: `.env`, `data/users.json` and `data/monitoring/` are git-ignored and excluded from the Docker image.
- Set `KHOTWA_HTTPS=1` when the app is served behind HTTPS: the session cookie is then never sent over plain HTTP.
- Groq free tier: about 200,000 generation tokens per day, and an answer uses about 4,000 -- roughly 50 answered questions per day across all users. Past that, the app replies that the daily quota is reached instead of answering.

## Known limitations

- **Arabic titles are translations to proofread.** 164 of the 268 documents exist only in French. Arabic questions with specialised terms ("المعرف الجبائي", "التأشير") did not reach them, so each French-only document got an Arabic title in `kb/overrides.yaml`, translated from its French title and added to its chunks. Both such questions now rank first. The titles were written by an AI assistant and should be reviewed by a native speaker.
- **Derja in Latin script (arabizi) is the weakest group.** One question ("ena weld prof, famma minha...") scores under the refusal floor and gets "no information"; the robustness set has the same pattern.
- **Refusals rely on the LLM.** The score floor stops 13% of out-of-scope questions; in-scope Derja questions score as low as out-of-scope ones, so the floor cannot go higher without refusing real questions.
- **Answer faithfulness is only partly measured offline.** The grounding check covers figures, links and citations on every live answer; full RAGAS faithfulness needs an LLM judge and the Groq quota.
- **Two services.gov pages carry another service's description** ("Obtention du certificat annuel de l'impôt sur le revenu", "Obtention du certificat de déclaration des montants versés au Trésor public"): the error is on the source portal itself. A note in `kb/overrides.yaml` tells the model which parts concern the certificate.
- **The reranker is off in the web app.** bge-reranker-v2-m3 takes 5 s for 5 candidates and 40 s for 20 on this CPU.
