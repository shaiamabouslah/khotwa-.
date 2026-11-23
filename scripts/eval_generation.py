"""Generation evaluation: a sample of benchmark questions through the same pipeline as the web app
(retrieval + Groq), scored without an LLM judge.

  python scripts/eval_generation.py                 # 16 questions + 4 out-of-scope, paced for Groq's free tier
  python scripts/eval_generation.py --n 8 --negatives 2

Per answer: the outcome the web app would log (answer / uncited / no_match / clarification),
whether the answer cites the benchmark's gold document, the grounding check (scripts/grounding.py)
and the latency. Out-of-scope questions measure refusals end to end -- the score floor stops few
of them, so this is where the generation prompt's own refusal is tested.

Writes data/rag/metrics/generation.json. With MLFLOW_TRACKING_URI set (scripts/tracking.py), also
logs an MLflow run on DagsHub -- parameters, metrics, a table of every answer -- and one trace per
question: retrieval span (documents and scores), the Groq call (prompt, response, tokens), outcome.

Uses about 4,000 Groq tokens per answer (~20 answers = 80k of the 200k daily free tier).
"""
import argparse
import contextlib
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import generate  # noqa: E402
import grounding  # noqa: E402
import tracking  # noqa: E402
from api import WEB_RERANK  # noqa: E402
from retriever import Retriever  # noqa: E402

UNANSWERED = ("no_match", "uncited")


def load(path: str) -> list[dict]:
    with open(ROOT / path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(positives: list[dict], negatives: list[dict], n: int, n_neg: int, seed: int) -> list[dict]:
    """Stratified by question language, so every language is represented in a small sample."""
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for q in positives:
        by_lang[q["lang"]].append(q)
    for qs in by_lang.values():
        rng.shuffle(qs)
    picked, langs = [], sorted(by_lang)
    while len(picked) < n and any(by_lang.values()):
        for lang in langs:
            if by_lang[lang] and len(picked) < n:
                picked.append(dict(by_lang[lang].pop(), kind="in_scope"))
    return picked + [dict(q, kind="out_of_scope", lang="?") for q in rng.sample(negatives, min(n_neg, len(negatives)))]


def outcome(result: dict, body: str) -> str:
    """Same classification as scripts/api.py logs on the monitoring page."""
    if result.get("empty"):
        return "empty"
    if generate.clarify_options(body):
        return "clarification"
    if result["abstained"]:
        return "no_match"
    return "answer" if re.search(r"\[\d+\]", body) else "uncited"


def evaluate(question: dict, retriever: Retriever, model: str) -> dict:
    t0 = time.perf_counter()
    result = generate.answer(question["q"], retriever, rerank=WEB_RERANK, use_cache=False, model=model)
    latency = round(time.perf_counter() - t0, 2)
    if result.get("busy") and not result.get("empty"):
        return {"busy": True, "daily": result["answer"] in generate.DAILY_LIMIT_MESSAGE.values()}
    body, _ = generate.split_sources(result["answer"])
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
    cited_docs = [result["hits"][n - 1].doc_id for n in cited if 0 < n <= len(result["hits"])]
    gold = {question.get("doc_id"), *question.get("also_ok", [])} - {None}
    check = grounding.check(body, grounding.source_texts(result["hits"]), question["q"]) if result["hits"] else None
    return {
        "question": question["q"], "kind": question["kind"], "lang": question.get("lang"),
        "outcome": outcome(result, body), "answer": body, "cited_docs": cited_docs,
        "gold_cited": bool(gold & set(cited_docs)) if gold else None,
        "grounded": check["grounded"] if check else None,
        "unsupported": (check["unsupported_numbers"] + check["unsupported_urls"]
                        + [f"[{n}]" for n in check["bad_citations"]]) if check else [],
        "latency_s": latency,
    }


def summarize(rows: list[dict]) -> dict:
    inscope = [r for r in rows if r["kind"] == "in_scope"]
    outscope = [r for r in rows if r["kind"] == "out_of_scope"]
    answers = [r for r in rows if r["outcome"] == "answer"]
    rate = lambda part, whole: round(len(part) / len(whole), 4) if whole else None
    latencies = sorted(r["latency_s"] for r in rows)
    return {
        "n_in_scope": len(inscope), "n_out_of_scope": len(outscope),
        "answer_rate": rate([r for r in inscope if r["outcome"] == "answer"], inscope),
        "clarification_rate": rate([r for r in inscope if r["outcome"] == "clarification"], inscope),
        "empty_answer_rate": rate([r for r in rows if r["outcome"] == "empty"], rows),
        "gold_document_cited_rate": rate([r for r in inscope if r["gold_cited"]], inscope),
        "grounded_answer_rate": rate([r for r in answers if r["grounded"]], answers),
        "out_of_scope_refused_rate": rate([r for r in outscope if r["outcome"] in UNANSWERED], outscope),
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))] if latencies else None,
    }


def traced_retrieve(mlflow, original):
    from mlflow.entities import SpanType

    def retrieve(retriever, query, **kwargs):
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": query, **kwargs})
            hits = original(retriever, query, **kwargs)
            span.set_outputs([{"id": h.chunk_id, "page_content": h.text,
                               "metadata": {"doc_id": h.doc_id, "title": h.title, "lang": h.lang,
                                            "score": round(h.score, 4), "url": h.url}} for h in hits])
            return hits
    return retrieve


def answer_traced(mlflow, q: dict, retriever: Retriever, model: str) -> dict:
    from mlflow.entities import SpanType
    with mlflow.start_span(name="khotwa_answer", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"question": q["q"]})
        row = evaluate(q, retriever, model)
        span.set_outputs({k: row.get(k) for k in ("outcome", "answer", "grounded", "unsupported")})
        span.set_attributes({"kind": q["kind"], "lang": q.get("lang") or "?"})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16, help="in-scope questions (stratified by language)")
    ap.add_argument("--negatives", type=int, default=4, help="out-of-scope questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=20, help="seconds between questions (Groq tokens/minute cap)")
    ap.add_argument("--model", default=generate.DEFAULT_MODEL)
    ap.add_argument("--out", default="data/rag/metrics/generation.json")
    a = ap.parse_args()

    questions = sample(load("data/rag/questions.jsonl"), load("data/rag/negative_questions.jsonl"),
                       a.n, a.negatives, a.seed)
    retriever = Retriever(use_query_cache=False)

    mlflow = None
    if tracking.enabled():
        mlflow = tracking.start("generation")
        mlflow.groq.autolog()
        generate.retrieve = traced_retrieve(mlflow, generate.retrieve)
    run = (mlflow.start_run(run_name=f"generation {tracking.git_commit()}") if mlflow
           else contextlib.nullcontext())

    with run:
        if mlflow:
            mlflow.set_tag("git_commit", tracking.git_commit())
            mlflow.log_params({"model": a.model, "rerank": WEB_RERANK, "k": 5, "n_in_scope": a.n,
                               "n_out_of_scope": a.negatives, "seed": a.seed})
        rows = []
        for i, q in enumerate(questions, 1):
            row = answer_traced(mlflow, q, retriever, a.model) if mlflow else evaluate(q, retriever, a.model)
            if row.get("busy"):
                print("Groq daily quota reached -- stopping here." if row["daily"] else
                      "Groq per-minute limit -- stopping; rerun with a larger --sleep.")
                break
            rows.append(row)
            flag = "" if row["grounded"] in (True, None) else f"  UNSUPPORTED {row['unsupported']}"
            print(f"[{i}/{len(questions)}] {row['kind']:<12} {row['outcome']:<13} gold_cited={row['gold_cited']} "
                  f"{row['latency_s']}s  {q['q'][:55]}{flag}", flush=True)
            if i < len(questions):
                time.sleep(a.sleep)

        if not rows:
            sys.exit("no question evaluated")
        metrics = summarize(rows)
        print("\n" + json.dumps(metrics, indent=2))
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")

        if mlflow:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            mlflow.log_table({k: [", ".join(map(str, r[k])) if isinstance(r[k], list) else r[k] for r in rows]
                              for k in rows[0]}, "answers.json")
            mlflow.flush_trace_async_logging()
            print("run and traces logged to MLflow (experiment generation)")


if __name__ == "__main__":
    main()
