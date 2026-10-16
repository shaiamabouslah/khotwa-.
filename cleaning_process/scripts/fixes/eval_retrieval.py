"""Retrieval check on data/rag/chunks.jsonl with a dependency-free BM25 (a floor: your real retriever should beat it).
  python scripts/fixes/eval_retrieval.py --auto                       # each doc must be found from its own title
  python scripts/fixes/eval_retrieval.py --file my_questions.jsonl    # {"q": "...", "doc_id": "...", "lang": "fr"} per line
Reports hit@1/3/5, MRR and lists the misses. Real questions beat --auto: write 30-50."""
import argparse, collections, json, math, re
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--chunks", default="data/rag/chunks.jsonl")
ap.add_argument("--file"); ap.add_argument("--auto", action="store_true"); a = ap.parse_args()
tok = lambda s: re.findall(r"\w+", s.lower())
chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8")]
N = len(chunks); toks = [tok(c["text"]) for c in chunks]; avg = sum(map(len, toks)) / N
df = collections.Counter(t for ts in toks for t in set(ts)); tf = [collections.Counter(ts) for ts in toks]


def search(q, lang, k=5):
    scores = []
    for i, c in enumerate(chunks):
        if lang and c["lang"] != lang: continue
        s = sum(math.log(1 + (N - df[t] + .5) / (df[t] + .5)) * tf[i][t] * 2.2 / (tf[i][t] + 1.2 * (.25 + .75 * len(toks[i]) / avg))
                for t in set(tok(q)) if t in tf[i])
        scores.append((s, c["doc_id"]))
    docs, seen = [], set()
    for s, d in sorted(scores, reverse=True):
        if d not in seen: seen.add(d); docs.append(d)
        if len(docs) == k: break
    return docs


if a.file: qs = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
else:
    seen = {}
    for c in chunks: seen.setdefault((c["doc_id"], c["lang"]), c["title"])
    qs = [{"q": t, "doc_id": d, "lang": l} for (d, l), t in seen.items()]
hits = collections.Counter(); rr = 0; miss = []
for q in qs:
    r = search(q["q"], q.get("lang"))
    pos = r.index(q["doc_id"]) + 1 if q["doc_id"] in r else 0
    for k in (1, 3, 5): hits[k] += 0 < pos <= k
    rr += 1 / pos if pos else 0
    if not pos: miss.append(q)
n = len(qs)
print(f"{n} questions | hit@1 {hits[1]/n:.2f}  hit@3 {hits[3]/n:.2f}  hit@5 {hits[5]/n:.2f}  MRR {rr/n:.2f}")
for m in miss[:8]: print("  miss:", m["q"][:70], "->", m["doc_id"][:60])
