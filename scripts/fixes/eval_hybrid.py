"""Compare Retriever.search(hybrid=False) against hybrid=True (dense+BM25 RRF fusion) on the
same real question set, loading the model/embeddings ONCE and reusing them for every question
and both conditions -- unlike eval_retrieval.py/eval_dense.py (one subprocess per strategy via
compare_chunking.py), this needs only a single embedding pass since both conditions run against
the same chunks.jsonl/chunks.bge-m3.npy.

  python scripts/fixes/eval_hybrid.py --file data/rag/questions.jsonl
  python scripts/fixes/eval_hybrid.py --file data/rag/questions.jsonl --no-rerank

Reports hit@1/3/5 and MRR for both conditions, plus which questions flipped (hit with one
condition, miss with the other) so a difference is traceable to a specific case, not just a
delta in a summary number.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retriever import Retriever


def evaluate(r: Retriever, questions: list[dict], hybrid: bool, rerank: bool):
    hits_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    per_question = {}
    for q in questions:
        hits = r.search(q["q"], k=5, lang=q.get("lang"), rerank=rerank, hybrid=hybrid, min_score=None)
        doc_ids = [h.doc_id for h in hits]
        pos = doc_ids.index(q["doc_id"]) + 1 if q["doc_id"] in doc_ids else 0
        for k in (1, 3, 5):
            hits_at[k] += 1 if 0 < pos <= k else 0
        rr_sum += 1 / pos if pos else 0
        per_question[q["q"]] = pos
    n = len(questions)
    return {"n": n, "hit@1": hits_at[1] / n, "hit@3": hits_at[3] / n, "hit@5": hits_at[5] / n,
            "mrr": rr_sum / n, "per_question": per_question}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/rag/questions.jsonl")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    a = ap.parse_args()

    questions = [json.loads(l) for l in Path(a.file).open(encoding="utf-8") if l.strip()]
    print(f"{len(questions)} questions from {a.file}, rerank={a.rerank}")
    print("loading retriever (embeds once, reused for both conditions below)...", flush=True)
    r = Retriever(use_query_cache=False)  # fresh scores each run -- comparing two conditions, cache would mix them

    print("evaluating hybrid=False (current default)...", flush=True)
    dense = evaluate(r, questions, hybrid=False, rerank=a.rerank)
    print("evaluating hybrid=True (dense+BM25 RRF)...", flush=True)
    hyb = evaluate(r, questions, hybrid=True, rerank=a.rerank)

    print("\n" + "=" * 60)
    print(f"{'condition':<12} {'n':>4} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'MRR':>7}")
    print("-" * 60)
    for label, m in (("dense-only", dense), ("hybrid", hyb)):
        print(f"{label:<12} {m['n']:>4} {m['hit@1']:>7.2f} {m['hit@3']:>7.2f} {m['hit@5']:>7.2f} {m['mrr']:>7.2f}")
    print("=" * 60)

    flipped = [(q, dense["per_question"][q], hyb["per_question"][q])
               for q in dense["per_question"] if dense["per_question"][q] != hyb["per_question"][q]]
    if flipped:
        print(f"\n{len(flipped)} question(s) changed rank (0 = miss):")
        for q, dp, hp in flipped:
            arrow = "better" if (hp and (not dp or hp < dp)) else "worse"
            print(f"  [{arrow:>6}] dense_pos={dp} hybrid_pos={hp}  {q}")
    else:
        print("\nno question changed rank between conditions")


if __name__ == "__main__":
    main()
