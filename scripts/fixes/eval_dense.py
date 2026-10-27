"""Retrieval check on a chunks.jsonl using a real embedding model (default: BGE-M3, multilingual
fr/en/ar) instead of BM25 -- same protocol as eval_retrieval.py so the two are directly
comparable: same questions, same hit@1/3/5 and MRR, only the scoring differs.
  python scripts/fixes/eval_dense.py --auto
  python scripts/fixes/eval_dense.py --chunks data/rag/cmp_semantic/chunks.jsonl --auto
  python scripts/fixes/eval_dense.py --file my_questions.jsonl
Embeddings are cached next to --chunks (<name>.<model>.npy) since BGE-M3 on CPU is not instant;
delete that file (or pass --no-cache) after the chunks actually change.
"""
import argparse, collections, hashlib, json, sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="data/rag/chunks.jsonl")
ap.add_argument("--file")
ap.add_argument("--auto", action="store_true")
ap.add_argument("--embed-model", default="BAAI/bge-m3")
ap.add_argument("--no-cache", action="store_true")
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
    mat = np.asarray(embed([c["text"] for c in chunks]), dtype=np.float32)
    np.save(cache_path, mat)


def search(q, lang, k=5):
    qv = np.asarray(embed([q])[0], dtype=np.float32)
    sims = mat @ qv  # both normalized -> dot product = cosine similarity
    order = np.argsort(-sims)
    docs, seen = [], set()
    for i in order:
        c = chunks[i]
        if lang and c["lang"] != lang:
            continue
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"]); docs.append(c["doc_id"])
        if len(docs) == k:
            break
    return docs


if a.file:
    qs = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
else:
    seen = {}
    for c in chunks:
        seen.setdefault((c["doc_id"], c["lang"]), c["title"])
    qs = [{"q": t, "doc_id": d, "lang": l} for (d, l), t in seen.items()]

hits = collections.Counter(); rr = 0; miss = []
for q in qs:
    r = search(q["q"], q.get("lang"))
    pos = r.index(q["doc_id"]) + 1 if q["doc_id"] in r else 0
    for k in (1, 3, 5): hits[k] += 0 < pos <= k
    rr += 1 / pos if pos else 0
    if not pos: miss.append(q)
n = len(qs)
print(f"[{a.embed_model}] {chunks_path}")
print(f"{n} questions | hit@1 {hits[1]/n:.2f}  hit@3 {hits[3]/n:.2f}  hit@5 {hits[5]/n:.2f}  MRR {rr/n:.2f}")
for m in miss[:8]:
    print("\nMISS:", m["q"], "->", m["doc_id"])
    print("Retrieved:", search(m["q"], m.get("lang"), k=5))
