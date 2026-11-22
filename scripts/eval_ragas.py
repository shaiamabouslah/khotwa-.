#!/usr/bin/env python3
"""RAGAS evaluation of the full generation pipeline (retriever + Groq), on real questions.

Two reference-free metrics (no hand-written "ground truth answer" needed, so this runs
directly on data/rag/questions.jsonl):
  - faithfulness: for each atomic claim in the generated answer, can it be verified against
    the retrieved context? Low score = the model said something the context doesn't support,
    i.e. hallucination. This is the direct, standard-methodology answer to "prove it doesn't
    hallucinate" instead of just eyeballing a few examples by hand.
  - context_relevance: is the retrieved context actually relevant to the question? A low score
    here points at a retrieval problem, not a generation problem -- useful to tell the two
    apart when something scores badly.

Both are judged by an LLM (here: the same Groq model used for generation, via an
AsyncOpenAI client pointed at Groq's OpenAI-compatible endpoint -- ragas's own groq/anthropic
adapters hit unrelated bugs in this ragas release, the openai-compatible path is the one that
actually works).

Usage:
  python scripts/eval_ragas.py                       # all questions in data/rag/questions.jsonl
  python scripts/eval_ragas.py --file data/rag/hard_questions.jsonl
  python scripts/eval_ragas.py --limit 10             # first N only, for a quick/cheap run
  python scripts/eval_ragas.py --sleep 25             # pace calls, stay under Groq's free-tier
                                                        # per-minute token cap
  python scripts/eval_ragas.py --sleep 25 --resume    # continue a run stopped by the daily cap

Groq's free tier also caps tokens per day (200k for the default model); one evaluated answer is a
generation call plus several judge calls of a few thousand tokens each, so a full run spans more
than one day: the run stops at the daily cap and --resume picks up where it left off.

Results are written incrementally to data/rag/ragas_results.json (one entry per question, as
soon as it's scored) so a 429 or a crash partway through a long run loses only the questions
not yet reached, not everything scored so far.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import types
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ragas 0.4.3's instructor adapter unconditionally builds a dict covering every provider's
# patch (anthropic, google, ...) even when you only ever use one -- so it crashes on an
# unrelated provider's missing attribute unless every single one of them happens to resolve.
# Stubbing this one module is the one that was actually missing in this environment.
_fake = types.ModuleType("langchain_community.chat_models.vertexai")
_fake.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _fake

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import ContextRelevance, Faithfulness  # noqa: E402

from api import WEB_RERANK  # noqa: E402  (score the exact retrieval setting the web app uses)
from generate import DEFAULT_MODEL, answer, clarify_options, split_sources  # noqa: E402
from retriever import Retriever  # noqa: E402


def build_judge(model: str):
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Export it, or put GROQ_API_KEY=... in a .env file at the repo root.")
    client = AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    # max_tokens default is too low for the NLI-verdict judge prompt (it was truncating mid-
    # response and crashing the whole run on question 2 of the first smoke test).
    return llm_factory(model=model, provider="openai", client=client, max_tokens=4096)


async def _score(q: dict, result: dict, faithfulness: Faithfulness, context_relevance: ContextRelevance):
    body, _ = split_sources(result["answer"])
    # Same text per hit the generator saw (build_context): title + chunk, not the chunk alone.
    contexts = [f"{h.title}\n{h.text}" for h in result["hits"]]
    faith = await faithfulness.ascore(user_input=q["q"], response=body, retrieved_contexts=contexts)
    ctx = await context_relevance.ascore(user_input=q["q"], retrieved_contexts=contexts)
    return faith.value, ctx.value


async def eval_one(q: dict, retriever: Retriever, faithfulness: Faithfulness,
                    context_relevance: ContextRelevance, gen_model: str, retry_wait: float,
                    retries: int) -> dict:
    """Scores what the web app would answer (same rerank setting as scripts/api.py, the source list
    appended by sources_block() stripped so only the model's own claims are judged). Turns that
    are not a generated answer -- abstention, clarification question, Groq busy, or a reply
    citing no source -- are counted by outcome instead of being scored."""
    row = {"q": q["q"], "lang": q.get("lang"), "outcome": None, "faithfulness": None,
           "context_relevance": None, "error": None}
    for attempt in range(1, retries + 2):
        # Everything here can hit a Groq rate limit (generation included, not just the judge) --
        # one question's 429 must never crash the batch; retry it after a pause.
        try:
            result = answer(q["q"], retriever, lang=q.get("lang"), model=gen_model, rerank=WEB_RERANK)
            body, _ = split_sources(result["answer"])
            # Checked first: a rate-limited generation also comes back with no hits, and was once
            # miscounted as an abstention for most of a run.
            if result.get("busy"):
                raise RuntimeError(f"Groq rate limit: {result['answer']}")
            if result["abstained"]:
                row["outcome"] = "abstained"
            elif clarify_options(body):
                row["outcome"] = "clarification"
            elif not re.search(r"\[\d+\]", body):
                row["outcome"] = "declined"
            else:
                row["outcome"] = "answer"
                row["faithfulness"], row["context_relevance"] = await _score(
                    q, result, faithfulness, context_relevance)
            row["error"] = None
            return row
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if attempt <= retries:
                time.sleep(retry_wait)
    return row


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/rag/questions.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-model", default=DEFAULT_MODEL)
    ap.add_argument("--gen-model", default=DEFAULT_MODEL)
    ap.add_argument("--sleep", type=float, default=0.0,
                     help="seconds to wait between questions, to stay under Groq's free-tier TPM cap")
    ap.add_argument("--retry-wait", type=float, default=20.0,
                     help="seconds to wait before retrying a question that failed (usually a Groq 429)")
    ap.add_argument("--retries", type=int, default=3, help="retries per question after a failure")
    ap.add_argument("--out", default="data/rag/ragas_results.json")
    ap.add_argument("--resume", action="store_true",
                     help="keep the questions already evaluated in --out, run only the missing or failed "
                          "ones -- a full run needs more than one day of Groq's free-tier token quota")
    a = ap.parse_args()

    questions = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    if a.limit:
        questions = questions[: a.limit]

    judge = build_judge(a.judge_model)
    faithfulness = Faithfulness(llm=judge)
    context_relevance = ContextRelevance(llm=judge)
    retriever = Retriever()

    out_path = ROOT / a.out
    stopped = False
    done = {}
    if a.resume and out_path.exists():
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["q"]: r for r in (saved["rows"] if isinstance(saved, dict) else saved)
                if r.get("outcome") and not r.get("error")}
        print(f"resuming: {len(done)} question(s) already evaluated in {a.out}")
    rows = []
    for i, q in enumerate(questions, 1):
        if q["q"] in done:
            rows.append(done[q["q"]])
            continue
        if stopped:
            rows.append({"q": q["q"], "lang": q.get("lang"), "outcome": None, "faithfulness": None,
                         "context_relevance": None, "error": "not run: Groq daily quota reached"})
            continue
        r = await eval_one(q, retriever, faithfulness, context_relevance, a.gen_model, a.retry_wait, a.retries)
        rows.append(r)
        if r["error"] and "per day" in r["error"]:
            # Nothing more can run until the rolling daily window frees up; --resume continues later.
            stopped = True
            print("Groq daily token quota reached -- stopping; rerun with --resume later to continue.")
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        if r["error"]:
            tag = f"ERROR: {r['error']}"
        elif r["outcome"] == "answer":
            tag = f"faith={r['faithfulness']:.2f} ctx_rel={r['context_relevance']:.2f}"
        else:
            tag = r["outcome"].upper()
        print(f"[{i}/{len(questions)}] [{r['lang']}] {tag}  {q['q'][:70]}")
        if a.sleep and i < len(questions) and not stopped:
            time.sleep(a.sleep)

    def means(group: list[dict]) -> dict:
        scored = [r for r in group if r["outcome"] == "answer" and not r["error"]]
        out = {"n": len(group), "n_scored": len(scored)}
        for k in ("abstained", "clarification", "declined"):
            out[f"n_{k}"] = sum(1 for r in group if r["outcome"] == k)
        out["n_errors"] = sum(1 for r in group if r["error"])
        if scored:
            out["mean_faithfulness"] = sum(r["faithfulness"] for r in scored) / len(scored)
            out["mean_context_relevance"] = sum(r["context_relevance"] for r in scored) / len(scored)
        return out

    summary = {"file": a.file, "gen_model": a.gen_model, "judge_model": a.judge_model,
               "rerank": WEB_RERANK, **means(rows),
               "by_lang": {lang: means([r for r in rows if (r["lang"] or "?") == lang])
                           for lang in sorted({r["lang"] or "?" for r in rows})}}
    print(f"\n=== {a.file} ===")
    print(f"{'lang':<5}{'n':>4}{'scored':>8}{'abst':>6}{'clar':>6}{'decl':>6}{'err':>5}{'faith':>8}{'ctx_rel':>9}")
    for lang, m in [*summary["by_lang"].items(), ("ALL", summary)]:
        f = m.get("mean_faithfulness")
        c = m.get("mean_context_relevance")
        print(f"{lang:<5}{m['n']:>4}{m['n_scored']:>8}{m['n_abstained']:>6}{m['n_clarification']:>6}"
              f"{m['n_declined']:>6}{m['n_errors']:>5}{'' if f is None else f'{f:.3f}':>8}"
              f"{'' if c is None else f'{c:.3f}':>9}")
    low_faith = [r for r in rows if r["faithfulness"] is not None and r["faithfulness"] < 0.8]
    if low_faith:
        print(f"\n{len(low_faith)} answer(s) with faithfulness < 0.8 (possible hallucination):")
        for r in low_faith:
            print(f"  {r['faithfulness']:.2f}  [{r['lang']}] {r['q'][:70]}")

    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8", newline="\n")
    print(f"\nFull results written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main_async())
