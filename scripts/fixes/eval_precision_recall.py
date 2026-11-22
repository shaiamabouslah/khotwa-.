"""Precision@k / recall@k / MAP for Retriever.search() on a real question set.

Each line in the questions file is either single-relevant:
    {"q": "...", "doc_id": "...", "lang": "fr"}
or multi-relevant (several docs answer the same question):
    {"q": "...", "doc_ids": ["...", "..."], "lang": "fr"}

  python scripts/fixes/eval_precision_recall.py --file data/rag/questions.jsonl
  python scripts/fixes/eval_precision_recall.py --file data/rag/questions.jsonl --hybrid
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retriever import Retriever
from metrics import mean_at_k, precision_at_k, recall_at_k


def relevant_ids(q: dict) -> set[str]:
    return set(q["doc_ids"]) if "doc_ids" in q else {q["doc_id"]}


def evaluate(r: Retriever, questions: list[dict], k_max: int, hybrid: bool, rerank: bool):
    per_query = []
    for q in questions:
        hits = r.search(q["q"], k=k_max, lang=q.get("lang"), rerank=rerank, hybrid=hybrid, min_score=None)
        per_query.append(([h.doc_id for h in hits], relevant_ids(q)))
    return per_query


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/rag/questions.jsonl")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--ks", default="1,3,5")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    questions = [json.loads(l) for l in Path(a.file).open(encoding="utf-8") if l.strip()]
    print(f"{len(questions)} questions from {a.file}, hybrid={a.hybrid}, rerank={a.rerank}")

    r = Retriever()
    per_query = evaluate(r, questions, max(ks), a.hybrid, a.rerank)

    print(f"\n{'k':>3} {'precision@k':>12} {'recall@k':>10} {'MAP':>7}")
    for k in ks:
        m = mean_at_k(per_query, k)
        print(f"{k:>3} {m['precision@k']:>12.3f} {m['recall@k']:>10.3f} {m['map']:>7.3f}")

    worst_k = max(ks)
    ranked = sorted(
        zip(questions, per_query),
        key=lambda qp: recall_at_k(qp[1][0], qp[1][1], worst_k),
    )
    print(f"\nworst by recall@{worst_k}:")
    for q, (retrieved, relevant) in ranked[:8]:
        p = precision_at_k(retrieved, relevant, worst_k)
        rec = recall_at_k(retrieved, relevant, worst_k)
        print(f"  p={p:.2f} r={rec:.2f}  {q['q']}  (relevant={sorted(relevant)}, got={retrieved})")


if __name__ == "__main__":
    main()
