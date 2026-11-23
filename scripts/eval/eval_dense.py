"""Retrieval check on a chunks.jsonl using a real embedding model (default: BGE-M3, multilingual
fr/en/ar) instead of BM25 -- same protocol as eval_retrieval.py so the two are directly
comparable: same questions, same hit@1/3/5 and MRR, only the scoring differs.
  python scripts/eval/eval_dense.py --auto
  python scripts/eval/eval_dense.py --chunks data/rag/cmp_semantic/chunks.jsonl --auto
  python scripts/eval/eval_dense.py --file data/rag/questions.jsonl
  python scripts/eval/eval_dense.py --file data/rag/questions.jsonl --rerank
  python scripts/eval/eval_dense.py --file data/rag/questions.jsonl --by-subtopic
Embeddings are cached next to --chunks (<name>.<model>.npy) since BGE-M3 on CPU is not instant;
delete that file (or pass --no-cache) after the chunks actually change.
--rerank: bi-encoder fetches --candidate-k candidates, a cross-encoder (default
BAAI/bge-reranker-v2-m3) rescores them, slower per query but usually more precise at k=1.
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="data/rag/chunks.jsonl")
ap.add_argument("--file")
ap.add_argument("--auto", action="store_true")
ap.add_argument("--embed-model", default="BAAI/bge-m3")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--rerank", action="store_true")
ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
ap.add_argument("--candidate-k", type=int, default=20)
ap.add_argument("--by-subtopic", action="store_true", help="also break hit@1/MRR down by domain/subtopic")
a = ap.parse_args()

chunks_path = Path(a.chunks)
chunks = [json.loads(l) for l in chunks_path.open(encoding="utf-8")]
N = len(chunks)
if N == 0:
    sys.exit(f"{chunks_path}: no chunks")

model_slug = a.embed_model.split("/")[-1]
cache_path = chunks_path.with_suffix(f".{model_slug}.npy")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(a.embed_model)


def embed(texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


if not a.no_cache and cache_path.exists():
    mat = np.load(cache_path)
    if mat.shape[0] != N:
        mat = None
else:
    mat = None
if mat is None:
    # Chunk texts identical to production chunks (every short document, and all of the heading
    # strategy) reuse the production embeddings instead of re-embedding: same model, same text,
    # same vector -- cuts the CPU time of compare_chunking several-fold.
    known = {}
    prod_chunks, prod_npy = Path("data/rag/chunks.jsonl"), Path(f"data/rag/chunks.{model_slug}.npy")
    if prod_npy.exists() and prod_chunks.resolve() != chunks_path.resolve():
        prod_mat = np.load(prod_npy)
        prod_texts = [json.loads(l)["text"] for l in prod_chunks.open(encoding="utf-8")]
        if len(prod_texts) == prod_mat.shape[0]:
            known = dict(zip(prod_texts, prod_mat))
    missing = [c["text"] for c in chunks if c["text"] not in known]
    print(f"embedding {len(missing)} of {N} chunks ({N - len(missing)} reused from data/rag/)", file=sys.stderr)
    new = dict(zip(missing, np.asarray(embed(missing), dtype=np.float32))) if missing else {}
    mat = np.stack([known.get(c["text"], new.get(c["text"])) for c in chunks]).astype(np.float32)
    np.save(cache_path, mat)


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(a.rerank_model)
    return _reranker


def search(q, lang, k=5):
    qv = np.asarray(embed([q])[0], dtype=np.float32)
    sims = mat @ qv  # both normalized -> dot product = cosine similarity
    order = np.argsort(-sims)
    fetch_k = max(a.candidate_k, k) if a.rerank else k
    cand, seen = [], set()
    for i in order:
        c = chunks[i]
        if lang and c["lang"] != lang:
            continue
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"]); cand.append(c)
        if len(cand) == fetch_k:
            break
    if not a.rerank or not cand:
        return [c["doc_id"] for c in cand[:k]]
    scores = get_reranker().predict([(q, c["text"]) for c in cand])
    cand = [c for _, c in sorted(zip(scores, cand), key=lambda p: -p[0])]
    return [c["doc_id"] for c in cand[:k]]


if a.file:
    qs = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
else:
    seen = {}
    for c in chunks:
        seen.setdefault((c["doc_id"], c["lang"]), c["title"])
    qs = [{"q": t, "doc_id": d, "lang": l} for (d, l), t in seen.items()]

doc_group = {}
for c in chunks:
    doc_group.setdefault(c["doc_id"], (c["domain"], c["subtopic"]))

hits = collections.Counter(); rr = 0; miss = []
by_group = collections.defaultdict(lambda: {"n": 0, "hit1": 0, "rr": 0.0})
for q in qs:
    # Only --auto restricts to the document's own language (a title must find its own page);
    # a real question can be answered by a page in another language, as in the app.
    r = search(q["q"], None if a.file else q.get("lang"))
    gold = {q["doc_id"], *q.get("also_ok", [])}
    pos = next((i for i, d in enumerate(r, 1) if d in gold), 0)
    for k in (1, 3, 5): hits[k] += 0 < pos <= k
    rr += 1 / pos if pos else 0
    if not pos: miss.append(q)
    g = doc_group.get(q["doc_id"], ("?", "?"))
    by_group[g]["n"] += 1
    by_group[g]["hit1"] += pos == 1
    by_group[g]["rr"] += 1 / pos if pos else 0
n = len(qs)
rerank_tag = f" +rerank({a.rerank_model})" if a.rerank else ""
print(f"[{a.embed_model}{rerank_tag}] {chunks_path}")
print(f"{n} questions | hit@1 {hits[1]/n:.2f}  hit@3 {hits[3]/n:.2f}  hit@5 {hits[5]/n:.2f}  MRR {rr/n:.2f}")

if a.by_subtopic:
    print("\nper domain/subtopic:")
    for (domain, subtopic), s in sorted(by_group.items()):
        print(f"  {domain}/{subtopic:<22} n={s['n']:>2}  hit@1 {s['hit1']/s['n']:.2f}  MRR {s['rr']/s['n']:.2f}")

for m in miss[:8]:
    print("\nMISS:", m["q"], "->", m["doc_id"])
    print("Retrieved:", search(m["q"], None if a.file else m.get("lang"), k=5))
