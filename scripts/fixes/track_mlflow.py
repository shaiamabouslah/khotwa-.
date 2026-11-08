#!/usr/bin/env python3
"""Log one RAG evaluation run to MLflow: retrieval quality, latency, RAGAS.
  python scripts/fixes/track_mlflow.py --name heading-baseline
  python scripts/fixes/track_mlflow.py --build --ragas-n 0
  python scripts/fixes/track_mlflow.py --ragas-n 0      # skip RAGAS (no LLM calls)
  python scripts/fixes/track_mlflow.py --name rerank-5 --candidate-k 5 --ragas-n 0
  python scripts/fixes/track_mlflow.py --name rerank-10 --candidate-k 10 --ragas-n 0
  mlflow ui                                       # browse/compare runs
"""
import argparse, hashlib, json, math, numbers, os, platform, random, re, subprocess, sys, time
from pathlib import Path

import mlflow
import torch
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sources.yaml").exists())
sys.path.insert(0, str(ROOT / "scripts"))
import rag_prepare  # noqa: E402
from retriever import Retriever, EMBED_MODEL, RERANK_MODEL, RERANKER_MAX_LENGTH  # noqa: E402
from generate import generate_text, DEFAULT_MODEL           # noqa: E402
from tracing import span  # noqa: E402

RETRIEVAL_K = 5
GENERATION_K = 3

config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
evaluation_config = config.get("evaluation", {})
mlflow_config = config.get("observability", {}).get("mlflow", {})
os.environ["KHOTWA_TRACE"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--name")
ap.add_argument("--questions", default="data/rag/questions.jsonl")
ap.add_argument("--strategy", choices=["heading", "fixed", "semantic"],
                help="chunking strategy used to build chunks.jsonl")
ap.add_argument("--build", action="store_true",
                help="rebuild chunks for the current branch into data/rag/exp_<branch> before evaluating")
ap.add_argument("--ragas-n", type=int, help="questions sampled for generation + RAGAS (0 = skip)")
ap.add_argument("--llm", help="LLM used to generate evaluation answers")
ap.add_argument("--judge", help="LLM used by RAGAS as judge")
ap.add_argument("--candidate-k", type=int, default=config.get("retrieval", {}).get("candidate_k", 5),
                help="number of dense-retrieval candidates passed to the reranker")
a = ap.parse_args()
a.strategy = a.strategy or evaluation_config.get("chunking_strategy", "heading")
a.ragas_n = a.ragas_n if a.ragas_n is not None else evaluation_config.get("ragas_questions", 10)
a.llm = a.llm or evaluation_config.get("model", DEFAULT_MODEL)
a.judge = a.judge or evaluation_config.get("judge_model", a.llm)
if a.candidate_k < RETRIEVAL_K:
    ap.error(f"--candidate-k must be at least {RETRIEVAL_K} for top-{RETRIEVAL_K} evaluation")
if a.ragas_n < 0:
    ap.error("--ragas-n must be zero or greater")

try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except ImportError:
    pass
if a.ragas_n and not os.environ.get("GROQ_API_KEY"):
    ap.error("GROQ_API_KEY is required when --ragas-n is greater than zero")

p95 = lambda x: sorted(x)[max(0, math.ceil(len(x) * .95) - 1)]
mean = lambda x: sum(x) / len(x)

def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def expected_doc_ids(question):
    expected = question.get("doc_ids", question.get("doc_id"))
    if isinstance(expected, str):
        expected = [expected]
    if not isinstance(expected, list) or not expected or any(not isinstance(doc_id, str) or not doc_id for doc_id in expected):
        raise ValueError("Each question must define a non-empty doc_id or doc_ids list")
    return expected


def fail_run(stage, exc):
    if not mlflow.active_run():
        return
    mlflow.set_tags({
        "evaluation.run_status": "failed",
        "evaluation.failure_stage": stage,
        "evaluation.failure_type": type(exc).__name__,
    })
    mlflow.log_dict({
        "stage": stage, "error_type": type(exc).__name__, "error": str(exc),
    }, "evaluation_error.json")
    mlflow.end_run(status="FAILED")


branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "--short", "HEAD")
dirty = bool(git("status", "--porcelain", "--", "scripts", "kb"))
if dirty:
    print("WARNING: uncommitted changes in scripts/ or kb/, run is tagged git_dirty=True")

rag_dir, build_s = ROOT / "data" / "rag", None
if a.build:
    rag_dir /= "exp_" + re.sub(r"[^\w.-]+", "-", branch)
    t = time.perf_counter()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rag_prepare.py"), "--kb", str(ROOT / "kb"),
         "--out", str(rag_dir), "--strategy", a.strategy],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    build_s = time.perf_counter() - t

# query cache OFF: otherwise repeated questions give fake ~0 ms latencies
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_config.get("tracking_uri", "sqlite:///mlflow.db"))
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", mlflow_config.get("experiment", "khotwa-chunking"))
if tracking_uri.startswith("sqlite:///") and not Path(tracking_uri.removeprefix("sqlite:///")).is_absolute():
    tracking_db = (ROOT / tracking_uri.removeprefix("sqlite:///")).resolve().as_posix()
    tracking_uri = f"sqlite:///{tracking_db}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)
questions_path = Path(a.questions)
if not questions_path.is_absolute():
    questions_path = ROOT / questions_path
qs = [json.loads(line) for line in questions_path.open(encoding="utf-8") if line.strip()]
if not qs:
    ap.error(f"No evaluation questions found in {questions_path}")
for question_index, question in enumerate(qs):
    if not question.get("q"):
        ap.error(f"Question {question_index} has no 'q' text")
    try:
        expected_doc_ids(question)
    except ValueError as exc:
        ap.error(f"Question {question_index}: {exc}")
dataset_sha256 = hashlib.sha256(questions_path.read_bytes()).hexdigest()

mlflow.start_run(run_name=a.name or branch)


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    fail_run("uncaught", exc_value)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_uncaught_exception
mlflow.set_tags({
    "git_branch": branch, "git_commit": commit, "git_dirty": str(dirty),
    "evaluation.task": "rag_retrieval_and_generation",
    "evaluation.chunking_strategy": a.strategy,
    "evaluation.dataset": str(questions_path.relative_to(ROOT)) if questions_path.is_relative_to(ROOT) else str(questions_path),
    "evaluation.dataset_sha256": dataset_sha256,
    "evaluation.run_status": "running",
})
mlflow.log_params({
    "embed_model": EMBED_MODEL, "reranker": RERANK_MODEL, "llm": a.llm, "ragas_judge": a.judge,
    "reranker_max_length": RERANKER_MAX_LENGTH,
    "chunking": a.strategy, "target_words": rag_prepare.TARGET_WORDS,
    "max_words": rag_prepare.MAX_WORDS, "overlap": rag_prepare.OVERLAP,
    "k_retrieval": RETRIEVAL_K, "k_generation": GENERATION_K, "candidate_k": a.candidate_k,
    "reranking_enabled": True, "min_rerank_score": "disabled_for_ranking_metrics",
    "ragas_questions_requested": a.ragas_n, "ragas_questions_actual": min(a.ragas_n, len(qs)),
    "question_file": str(questions_path), "question_file_sha256": dataset_sha256,
    "n_questions": len(qs), "chunks_dir": rag_dir.name,
    "python_version": platform.python_version(), "platform": platform.platform(),
})
t = time.perf_counter()
r = Retriever(rag_dir=rag_dir, use_query_cache=False)
retriever_load_s = time.perf_counter() - t
r.search("warmup", k=1, candidate_k=a.candidate_k)  # load embedder + reranker outside the timers
reranker_device = str(r._get_reranker().model.device)
mlflow.log_param("reranker_device", reranker_device)
mlflow.log_param("torch_num_threads", torch.get_num_threads())
mlflow.set_tag("evaluation.reranker_device", reranker_device)

# ── retrieval: classic metrics + latency ─────────────────────────────────────
hit = {1: 0, 3: 0, 5: 0}; rr = 0.0; t_ret = []; rows = []
by_lang = {}
for question_index, q in enumerate(qs):
    expected_ids = expected_doc_ids(q)
    expected_set = set(expected_ids)
    t = time.perf_counter()
    with span("evaluate_retrieval_question", "CHAIN") as sp:
        sp.set_attributes({
            "question_index": question_index, "question": q["q"],
            "expected_doc_ids": json.dumps(expected_ids, ensure_ascii=False),
            "lang": q.get("lang", ""),
        })
        hits = r.search(q["q"], k=RETRIEVAL_K, lang=q.get("lang"), candidate_k=a.candidate_k,
                        min_score=None)   # min_score=None: measure ranking, not abstention
        ids = [h.doc_id for h in hits]
        matching = [(rank, doc_id) for rank, doc_id in enumerate(ids, 1) if doc_id in expected_set]
        pos = matching[0][0] if matching else 0
        matched_doc_id = matching[0][1] if matching else ""
        sp.set_attributes({
            "expected_found": pos > 0, "expected_rank": pos,
            "matched_doc_id": matched_doc_id,
            "hit_at_1": pos == 1, "hit_at_3": 0 < pos <= 3, "hit_at_5": 0 < pos <= 5,
            "n_results": len(hits), "retrieved_doc_ids": json.dumps(ids, ensure_ascii=False),
            "top_doc_id": hits[0].doc_id if hits else "",
            "top_title": hits[0].title if hits else "",
            "top_score": hits[0].score if hits else 0.0,
            "top_bi_score": hits[0].bi_score if hits and hits[0].bi_score is not None else 0.0,
        })
    t_ret.append((time.perf_counter() - t) * 1000)
    for k in hit: hit[k] += 0 < pos <= k
    rr += 1 / pos if pos else 0
    lang = q.get("lang") or "unknown"
    lang_stats = by_lang.setdefault(lang, {"n": 0, "hit1": 0, "hit3": 0, "hit5": 0, "rr": 0.0})
    lang_stats["n"] += 1
    lang_stats["hit1"] += pos == 1
    lang_stats["hit3"] += 0 < pos <= 3
    lang_stats["hit5"] += 0 < pos <= 5
    lang_stats["rr"] += 1 / pos if pos else 0
    rows.append({
        "question_index": question_index, "question": q["q"], "lang": lang,
        "expected_doc_ids": expected_ids, "matched_doc_id": matched_doc_id,
        "expected_rank": pos,
        "retrieval_ms": round(t_ret[-1], 1),
        "retrieved": [{
            "rank": rank, "doc_id": h.doc_id, "title": h.title, "lang": h.lang,
            "domain": h.domain, "subtopic": h.subtopic, "section": h.section,
            "score": h.score, "bi_score": h.bi_score, "url": h.url,
        } for rank, h in enumerate(hits, 1)],
        "hits": hits,
    })
n = len(qs)
metrics = {f"hit_at_{k}": v / n for k, v in hit.items()}
metrics |= {"mrr": rr / n, "retrieval_ms_mean": mean(t_ret), "retrieval_ms_p95": p95(t_ret)}
metrics["retriever_load_s"] = retriever_load_s
cw = sorted(len(c["text"].split()) for c in r.chunks)
metrics |= {"n_chunks": len(cw), "chunk_words_median": cw[len(cw) // 2],
            "chunk_words_p90": cw[9 * len(cw) // 10]}
metrics["n_unique_documents"] = len({(c["doc_id"], c["lang"]) for c in r.chunks})
metrics["n_questions"] = n
metrics["n_questions_with_lang"] = sum(bool(q.get("lang")) for q in qs)
metrics["n_questions_without_lang"] = n - metrics["n_questions_with_lang"]
for lang, values in by_lang.items():
    for metric_name, value in (
        ("hit_at_1", values["hit1"] / values["n"]),
        ("hit_at_3", values["hit3"] / values["n"]),
        ("hit_at_5", values["hit5"] / values["n"]),
        ("mrr", values["rr"] / values["n"]),
    ):
        metrics[f"{metric_name}_{lang}"] = value
if build_s is not None:
    metrics["build_s"] = build_s
mlflow.log_metrics(metrics)
mlflow.log_dict([{
    key: value for key, value in row.items() if key != "hits"
} for row in rows], "per_question_retrieval.json")

# ── generation latency + RAGAS (on a sample: RAGAS = many LLM calls) ─────────
generated = []
if a.ragas_n:
    sample = random.Random(0).sample(rows, min(a.ragas_n, n))
    t_gen, samples = [], []
    for row in sample:
        top3 = row["hits"][:GENERATION_K]
        t = time.perf_counter()
        try:
            with span("evaluate_generation_question", "CHAIN") as sp:
                sp.set_attributes({
                    "question_index": row["question_index"], "question": row["question"],
                    "expected_doc_ids": json.dumps(row["expected_doc_ids"], ensure_ascii=False),
                    "retrieved_doc_ids": json.dumps([h.doc_id for h in top3], ensure_ascii=False),
                    "context_count": len(top3),
                })
                text = generate_text(row["question"], top3, a.llm)
                sp.set_attributes({"answer_chars": len(text)})
        except Exception as exc:
            if generated:
                mlflow.log_dict(generated, "per_question_generation.json")
            fail_run("generation", exc)
            raise
        t_gen.append(time.perf_counter() - t)
        samples.append({"user_input": row["question"], "response": text,
                        "retrieved_contexts": [h.text for h in top3]})
        generated.append({
            "question_index": row["question_index"], "question": row["question"],
            "expected_doc_ids": row["expected_doc_ids"], "answer": text,
            "retrieved_contexts": [{
                "doc_id": h.doc_id, "title": h.title, "url": h.url,
                "score": h.score, "text": h.text,
            } for h in top3],
            "generation_s": round(t_gen[-1], 3),
        })
    metrics |= {"generation_s_mean": mean(t_gen), "generation_s_p95": p95(t_gen)}
    mlflow.log_dict(generated, "per_question_generation.json")

    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
        from ragas.run_config import RunConfig

        ragas_start = time.perf_counter()
        with span("ragas_evaluation", "CHAIN") as sp:
            sp.set_attributes({"judge_model": a.judge, "n_samples": len(samples)})
            res = evaluate(
                EvaluationDataset.from_list(samples),
                metrics=[Faithfulness(), ResponseRelevancy(strictness=1),   # strictness=1: Groq rejects n>1
                         LLMContextPrecisionWithoutReference()],            # reference-free: we have no gold answers
                llm=LangchainLLMWrapper(ChatGroq(model=a.judge, temperature=0)),
                embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL)),
                run_config=RunConfig(max_workers=2, timeout=120))           # stay under Groq rate limits
        metrics["ragas_evaluation_s"] = time.perf_counter() - ragas_start
    except Exception as exc:
        fail_run("ragas", exc)
        raise

    ragas_df = res.to_pandas()
    for col, v in ragas_df.select_dtypes("number").mean().items():
        if not math.isnan(v): metrics[f"ragas_{col}"] = float(v)
    for generated_row, score_row in zip(generated, ragas_df.to_dict(orient="records")):
        generated_row["ragas_scores"] = {
            key: float(value) if isinstance(value, numbers.Real) and math.isfinite(float(value)) else None
            for key, value in score_row.items()
        }
        with span("ragas_question_scores", "CHAIN") as sp:
            sp.set_attributes({
                "question_index": generated_row["question_index"],
                "question": generated_row["question"],
                "judge_model": a.judge,
                **{f"score_{key}": value for key, value in generated_row["ragas_scores"].items()
                   if value is not None},
            })
    mlflow.log_dict(generated, "per_question_generation_and_ragas.json")

# ── log ──────────────────────────────────────────────────────────────────────
mlflow.log_metrics(metrics)
mlflow.set_tag("evaluation.run_status", "completed")
mlflow.log_dict({
    "branch": branch, "commit": commit, "git_dirty": dirty,
    "chunking_strategy": a.strategy, "chunks_dir": rag_dir.name,
    "question_file": str(questions_path), "question_file_sha256": dataset_sha256,
    "question_count": n, "language_counts": {lang: stats["n"] for lang, stats in by_lang.items()},
    "embedding_model": EMBED_MODEL, "reranker_model": RERANK_MODEL,
    "reranker_max_length": RERANKER_MAX_LENGTH,
    "reranker_device": reranker_device,
    "candidate_k": a.candidate_k, "generation_model": a.llm,
    "judge_model": a.judge, "metrics": metrics,
}, "evaluation_summary.json")
if (rag_dir / "report.md").exists():
    mlflow.log_artifact(str(rag_dir / "report.md"))
mlflow.end_run()
print(json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=1))