#!/usr/bin/env python3
"""Dense retriever over data/rag/chunks.jsonl, using BGE-M3 (multilingual fr/en/ar).

Library usage:
    from retriever import Retriever
    r = Retriever()                              # loads data/rag/, embeds (cached to .npy)
    hits = r.search("Comment affilier un salarie a la CNSS ?", k=5)
    for h in hits:
        print(h.score, h.title, h.url)
    full_text = r.expand(hits[0])                 # None if hits[0] is already a whole doc

    # rerank defaults to True: bi-encoder fetches candidate_k candidates, a cross-encoder
    # rescores them and returns the final k (slower per query, +0.06 hit@1 on the 33-question
    # eval set). Pass rerank=False to skip it.
    hits = r.search(q, k=5, rerank=False)

    # A repeat of the exact same (query, k, lang, domain, subtopic, rerank, candidate_k) call
    # skips both the embedding and the rerank pass -- see query_cache.json below.

    # min_score (default DEFAULT_MIN_RERANK_SCORE, rerank=True only): search() returns []
    # instead of a confident-looking wrong answer when the top hit's score is too low --
    # calibrated on real questions vs. out-of-scope ones, see scripts/fixes/calibrate_threshold.py.
    hits = r.search("Quelle est la recette du couscous ?")   # -> [] (out of scope)

CLI usage (manual testing, one question at a time):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py "Comment affilier un salarie ?"
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --domain employment "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --lang ar "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --no-rerank "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --no-query-cache "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --no-min-score "..."

Query cache: each CLI call is its own process, so an in-memory cache alone is useless here --
results are persisted to data/rag/query_cache.json, keyed on (normalized query, k, lang, domain,
subtopic, rerank, candidate_k). It's invalidated as a whole whenever chunks.jsonl changes (size +
mtime signature), so a kb/ rebuild can never serve a stale answer.
"""
import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = ROOT / "data" / "rag"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Calibrated via scripts/fixes/calibrate_threshold.py on the 33 real questions (data/rag/
# questions.jsonl) vs. 10 out-of-scope/adjacent-but-uncovered ones (data/rag/negative_questions
# .jsonl): with rerank=True, genuine top-1 hits scored >=0.664 (one exception: a wrong top-1
# answer that scored 0.024 -- correctly low), true negatives topped out at 0.221. 0.3 sits in
# that gap. Only meaningful for rerank=True -- the raw bi-encoder cosine score was not
# calibrated this way and sits on a different scale.
#
# That calibration used fr/ar questions only. Found by hand: an English query ("new high
# school graduate scholarship") whose correct top-1 doc had a perfectly normal bi_score
# (0.621, in line with every other genuine match seen all session) but a reranker score of
# 0.018 -- the cross-encoder's cross-lingual behavior isn't reliable for every EN<->FR term
# pair (idiomatic terms like "nouveau bachelier" have no direct lexical match), even though
# the same reranker handles plenty of other EN queries fine (a second English query scored
# 0.719, no issue). Rather than reject a confidently-embedded hit on a shaky rerank score,
# BI_SCORE_OVERRIDE lets a high bi-encoder score override a low rerank score -- not calibrated
# against real negatives the way DEFAULT_MIN_RERANK_SCORE was, so it's deliberately
# conservative (only fires when the bi-encoder itself is quite confident).
DEFAULT_MIN_RERANK_SCORE = 0.3
BI_SCORE_OVERRIDE = 0.6


@dataclass
class Hit:
    score: float
    chunk_id: str
    doc_id: str
    lang: str
    domain: str
    subtopic: str
    title: str
    org: str
    section: str
    url: str
    text: str
    n_chunks: int
    bi_score: float | None = None  # pre-rerank cosine score, set only when rerank=True


def _norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


class Retriever:
    def __init__(self, rag_dir: Path = RAG_DIR, embed_model: str = EMBED_MODEL,
                 rerank_model: str = RERANK_MODEL, use_cache: bool = True,
                 use_query_cache: bool = True):
        self.rag_dir = Path(rag_dir)
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model
        self._reranker = None  # lazy: only load the cross-encoder if search(rerank=True) is used
        chunks_path = self.rag_dir / "chunks.jsonl"
        self.chunks = [json.loads(l) for l in chunks_path.open(encoding="utf-8")]
        if not self.chunks:
            raise ValueError(f"{chunks_path}: no chunks -- run clean_pipeline.py --apply first")

        self.use_query_cache = use_query_cache
        self.query_cache_path = self.rag_dir / "query_cache.json"
        st = chunks_path.stat()
        self._chunks_sig = f"{st.st_size}:{st.st_mtime_ns}"
        self._query_cache = {}
        if use_query_cache and self.query_cache_path.exists():
            try:
                saved = json.loads(self.query_cache_path.read_text(encoding="utf-8"))
                if saved.get("chunks_sig") == self._chunks_sig:
                    self._query_cache = saved.get("entries", {})
            except (json.JSONDecodeError, OSError):
                pass  # corrupt/unreadable cache -> start fresh rather than fail the search

        self.parents = {}
        parents_path = self.rag_dir / "parents.jsonl"
        if parents_path.exists():
            for line in parents_path.open(encoding="utf-8"):
                p = json.loads(line)
                self.parents[(p["doc_id"], p["lang"])] = p

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embed_model)

        model_slug = embed_model.split("/")[-1]
        cache_path = chunks_path.with_suffix(f".{model_slug}.npy")
        mat = None
        if use_cache and cache_path.exists():
            mat = np.load(cache_path)
            if mat.shape[0] != len(self.chunks):
                mat = None
        if mat is None:
            mat = np.asarray(self._embed([c["text"] for c in self.chunks]), dtype=np.float32)
            if use_cache:
                np.save(cache_path, mat)
        self.matrix = mat

    def _embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.rerank_model_name)
        return self._reranker

    def _cache_key(self, query, k, lang, domain, subtopic, rerank, candidate_k) -> str:
        raw = "|".join(str(x) for x in (_norm_query(query), k, lang, domain, subtopic, rerank, candidate_k))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _save_query_cache(self):
        payload = {"chunks_sig": self._chunks_sig, "entries": self._query_cache}
        self.query_cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _apply_min_score(hits: list["Hit"], rerank: bool, min_score: float | None) -> list["Hit"]:
        if not rerank or min_score is None or not hits or hits[0].score >= min_score:
            return hits
        if hits[0].bi_score is not None and hits[0].bi_score >= BI_SCORE_OVERRIDE:
            return hits  # reranker scored it low, but the bi-encoder is confident -- see BI_SCORE_OVERRIDE
        return []

    def search(self, query: str, k: int = 5, lang: str | None = None,
               domain: str | None = None, subtopic: str | None = None,
               rerank: bool = True, candidate_k: int = 20,
               min_score: float | None = DEFAULT_MIN_RERANK_SCORE) -> list[Hit]:
        """rerank=True: fetch `candidate_k` candidates with the fast bi-encoder, then rescore
        them with a cross-encoder (query, chunk_text pairs) and return the top `k` by that
        score -- slower per query but typically more precise, especially at k=1.

        min_score: abstain (return []) when the top hit's score falls below this -- a question
        with no real match in the KB (or genuinely ambiguous) scores low; forcing an answer
        anyway is how a future generation layer would end up confidently citing the wrong
        procedure. Only applied when rerank=True (see DEFAULT_MIN_RERANK_SCORE); pass
        min_score=None to always return the top k regardless of confidence.

        An exact repeat of the same call (same normalized query, k, lang, domain, subtopic,
        rerank, candidate_k -- min_score is applied after the cache lookup, so it's not part
        of the key) is served from query_cache.json without touching the embedding model or
        the reranker."""
        cache_key = self._cache_key(query, k, lang, domain, subtopic, rerank, candidate_k)
        if self.use_query_cache and cache_key in self._query_cache:
            hits = [Hit(**h) for h in self._query_cache[cache_key]]
            return self._apply_min_score(hits, rerank, min_score)

        qv = np.asarray(self._embed([query])[0], dtype=np.float32)
        sims = self.matrix @ qv
        order = np.argsort(-sims)
        fetch_k = max(candidate_k, k) if rerank else k
        hits, seen_docs = [], set()
        for i in order:
            c = self.chunks[i]
            if lang and c["lang"] != lang:
                continue
            if domain and c["domain"] != domain:
                continue
            if subtopic and c["subtopic"] != subtopic:
                continue
            key = (c["doc_id"], c["lang"])
            if key in seen_docs:
                continue
            seen_docs.add(key)
            hits.append(Hit(score=float(sims[i]), chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                             lang=c["lang"], domain=c["domain"], subtopic=c["subtopic"],
                             title=c["title"], org=c.get("org", ""), section=c["section"],
                             url=c.get("url", ""), text=c["text"], n_chunks=c.get("n_chunks", 1)))
            if len(hits) == fetch_k:
                break
        if rerank and hits:
            reranker = self._get_reranker()
            scores = reranker.predict([(query, h.text) for h in hits])
            for h, s in zip(hits, scores):
                h.bi_score = h.score
                h.score = float(s)
            hits.sort(key=lambda h: -h.score)
            hits = hits[:k]

        if self.use_query_cache:
            self._query_cache[cache_key] = [asdict(h) for h in hits]
            self._save_query_cache()
        return self._apply_min_score(hits, rerank, min_score)

    def expand(self, hit: Hit) -> str | None:
        """Full procedure text for a hit that came from a multi-chunk document (small-to-big
        retrieval). Returns None for a hit whose chunk already IS the whole document."""
        parent = self.parents.get((hit.doc_id, hit.lang))
        return parent["text"] if parent else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--lang")
    ap.add_argument("--domain")
    ap.add_argument("--subtopic")
    ap.add_argument("--expand", action="store_true", help="also print the full parent procedure for the top hit")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                     help="skip the cross-encoder rerank stage (on by default: +0.06 hit@1 on the eval set)")
    ap.add_argument("--no-query-cache", dest="query_cache", action="store_false",
                     help="don't read/write data/rag/query_cache.json for this call")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_RERANK_SCORE,
                     help=f"abstain if the top (reranked) hit scores below this (default {DEFAULT_MIN_RERANK_SCORE})")
    ap.add_argument("--no-min-score", dest="min_score", action="store_const", const=None,
                     help="always return the top k, regardless of confidence")
    a = ap.parse_args()

    r = Retriever(use_query_cache=a.query_cache)
    hits = r.search(a.query, k=a.k, lang=a.lang, domain=a.domain, subtopic=a.subtopic,
                     rerank=a.rerank, min_score=a.min_score)
    if not hits:
        print("no confident match for this question in the KB" if a.rerank and a.min_score is not None
              else "no results")
        return
    for i, h in enumerate(hits):
        bi = f" (bi={h.bi_score:.3f})" if h.bi_score is not None else ""
        print(f"\n[{i + 1}] score={h.score:.3f}{bi}  {h.domain}/{h.subtopic}  [{h.lang}]  n_chunks={h.n_chunks}")
        print(f"    {h.title}  ({h.org})")
        print(f"    section: {h.section}")
        print(f"    {h.url}")
        print(f"    {h.text[:200].replace(chr(10), ' ')}...")

    if a.expand:
        full = r.expand(hits[0])
        print("\n--- expanded parent (top hit) ---")
        print(full if full else "(top hit is already the whole document, nothing to expand)")


if __name__ == "__main__":
    main()
