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
  python scripts/eval_ragas.py                       # all 33 questions in data/rag/questions.jsonl
  python scripts/eval_ragas.py --file data/rag/hard_questions.jsonl
  python scripts/eval_ragas.py --limit 10             # first N only, for a quick/cheap run
"""
import argparse
import asyncio
import json
import os
import sys
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

from generate import DEFAULT_MODEL, answer  # noqa: E402
from retriever import Retriever  # noqa: E402


def build_judge(model: str):
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Export it, or put GROQ_API_KEY=... in a .env file at the repo root.")
    client = AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    # max_tokens default is too low for the NLI-verdict judge prompt (it was truncating mid-
    # response and crashing the whole run on question 2 of the first smoke test).
    return llm_factory(model=model, provider="openai", client=client, max_tokens=4096)


async def eval_one(q: dict, retriever: Retriever, faithfulness: Faithfulness,
                    context_relevance: ContextRelevance, gen_model: str) -> dict:
    # Everything here can hit a Groq rate limit (generation call included, not just the judge
    # calls) -- one question's 429 must never crash the whole batch and lose every other result.
    try:
        result = answer(q["q"], retriever, lang=q.get("lang"), model=gen_model)
        if result["abstained"] or not result["hits"]:
            return {"q": q["q"], "abstained": True, "faithfulness": None, "context_relevance": None, "error": None}

        contexts = [h.text for h in result["hits"]]
        faith_score = await faithfulness.ascore(
            user_input=q["q"], response=result["answer"], retrieved_contexts=contexts)
        ctx_score = await context_relevance.ascore(user_input=q["q"], retrieved_contexts=contexts)
    except Exception as e:
        return {"q": q["q"], "abstained": False, "faithfulness": None, "context_relevance": None,
                "error": f"{type(e).__name__}: {e}"}
    return {
        "q": q["q"], "abstained": False,
        "faithfulness": faith_score.value, "context_relevance": ctx_score.value, "error": None,
    }


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/rag/questions.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-model", default=DEFAULT_MODEL)
    ap.add_argument("--gen-model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    questions = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    if a.limit:
        questions = questions[: a.limit]

    judge = build_judge(a.judge_model)
    faithfulness = Faithfulness(llm=judge)
    context_relevance = ContextRelevance(llm=judge)
    retriever = Retriever()

    rows = []
    for i, q in enumerate(questions, 1):
        r = await eval_one(q, retriever, faithfulness, context_relevance, a.gen_model)
        rows.append(r)
        if r["abstained"]:
            tag = "ABSTAINED"
        elif r["error"]:
            tag = f"ERROR: {r['error']}"
        else:
            tag = f"faith={r['faithfulness']:.2f} ctx_rel={r['context_relevance']:.2f}"
        print(f"[{i}/{len(questions)}] {tag}  {q['q'][:70]}")

    scored = [r for r in rows if not r["abstained"] and not r["error"]]
    n_abstained = sum(1 for r in rows if r["abstained"])
    n_errors = sum(1 for r in rows if r["error"])
    print(f"\n=== {a.file} ({len(rows)} questions, {n_abstained} abstained, {n_errors} errors) ===")
    if scored:
        avg_faith = sum(r["faithfulness"] for r in scored) / len(scored)
        avg_ctx = sum(r["context_relevance"] for r in scored) / len(scored)
        print(f"mean faithfulness:      {avg_faith:.3f}")
        print(f"mean context_relevance: {avg_ctx:.3f}")
        low_faith = [r for r in scored if r["faithfulness"] < 0.8]
        if low_faith:
            print(f"\n{len(low_faith)} question(s) with faithfulness < 0.8 (possible hallucination):")
            for r in low_faith:
                print(f"  {r['faithfulness']:.2f}  {r['q'][:70]}")


if __name__ == "__main__":
    asyncio.run(main_async())
