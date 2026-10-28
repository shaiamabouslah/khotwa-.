"""Calibrate a confidence threshold for Retriever.search(): compares top-1 rerank scores on
genuine questions (data/rag/questions.jsonl, a real match is expected) against top-1 scores
on negative/out-of-scope questions (data/rag/negative_questions.jsonl, no real match exists),
so the gap between the two distributions picks a cutoff for "no confident answer"."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retriever import Retriever

r = Retriever(use_query_cache=True)

pos = [json.loads(l) for l in open("data/rag/questions.jsonl", encoding="utf-8") if l.strip()]
neg = [json.loads(l) for l in open("data/rag/negative_questions.jsonl", encoding="utf-8") if l.strip()]

print("=== positive (genuine) questions: top-1 score ===")
pos_scores = []
for q in pos:
    hits = r.search(q["q"], k=1, lang=q.get("lang"))
    s = hits[0].score if hits else float("-inf")
    pos_scores.append(s)
pos_scores.sort()
print(f"n={len(pos_scores)}  min={pos_scores[0]:.3f}  p10={pos_scores[len(pos_scores)//10]:.3f}  "
      f"median={pos_scores[len(pos_scores)//2]:.3f}  max={pos_scores[-1]:.3f}")

print("\n=== negative (out-of-scope) questions: top-1 score ===")
neg_scores = []
for q in neg:
    hits = r.search(q["q"], k=1)
    s = hits[0].score if hits else float("-inf")
    neg_scores.append(s)
    print(f"  {s:+.3f}  [{q['expect']:<22}] {q['q'][:60]}  -> {hits[0].title[:50] if hits else '(none)'}")
neg_scores.sort()
print(f"\nn={len(neg_scores)}  min={neg_scores[0]:.3f}  median={neg_scores[len(neg_scores)//2]:.3f}  max={neg_scores[-1]:.3f}")

print(f"\nlowest genuine-hit score:  {pos_scores[0]:.3f}")
print(f"highest negative score:    {neg_scores[-1]:.3f}")
if neg_scores[-1] < pos_scores[0]:
    mid = (neg_scores[-1] + pos_scores[0]) / 2
    print(f"clean separation -> suggested threshold: {mid:.3f}")
else:
    print("OVERLAP: no clean threshold separates the two groups on this sample.")
