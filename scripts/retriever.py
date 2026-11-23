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
    # calibrated on real questions vs. out-of-scope ones, see scripts/eval/calibrate_threshold.py.
    hits = r.search("Quelle est la recette du couscous ?")   # -> [] (out of scope)

    # hybrid=True: candidates come from Reciprocal Rank Fusion of the dense ranking and a BM25
    # ranking -- meant to catch exact-term matches (decree numbers, acronyms, figures). Measured
    # by scripts/benchmark_retrieval.py: lower than dense on every language (Arabic and Derja
    # most), so the app stays dense-only.
    hits = r.search(q, k=5, hybrid=True)

CLI usage (manual testing, one question at a time):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py "Comment affilier un salarie ?"
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --domain employment "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --lang ar "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --no-rerank "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --hybrid "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --no-query-cache "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --no-min-score "..."

Query cache: each CLI call is its own process, so an in-memory cache alone is useless here --
results are persisted to data/rag/query_cache.json, keyed on (normalized query, k, lang, domain,
subtopic, rerank, candidate_k). It's invalidated as a whole whenever chunks.jsonl changes (size +
mtime signature), so a kb/ rebuild can never serve a stale answer.
"""
import argparse
import collections
import hashlib
import json
import math
import os
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


def _model_cached(repo_id: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


# Once the embedding model is in the local cache, never touch the network to load it. Without
# this, sentence-transformers still calls huggingface.co to check for updates on every load --
# measured: where that call is slow or blocked, Retriever() hung for 10+ minutes; with offline
# mode the same load takes ~12s. Must run before huggingface_hub is first imported (it reads the
# variable at import time); an explicit HF_HUB_OFFLINE set by the user always wins.
if _model_cached(EMBED_MODEL) and _model_cached(RERANK_MODEL):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Calibrated via scripts/eval/calibrate_threshold.py on the 33 real questions (data/rag/
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

# hybrid=True fuses the dense ranking with a BM25 (lexical) ranking via Reciprocal Rank Fusion
# instead of relying on dense/cosine similarity alone -- catches exact-term matches an embedding
# can miss (decree numbers, program acronyms, SMIG-style figures: all seen by hand this project
# where the exact string mattered and a paraphrase-tolerant embedding wasn't the point). RRF_K is
# the standard smoothing constant from the original RRF paper (Cormack et al. 2009). Benchmarked
# against dense-only in scripts/benchmark_retrieval.py: lower on every language, not the default.
RRF_K = 60

# Abstention floor for rerank=False, on the bi-encoder cosine scale (measured 2026-11-22). The 41
# benchmark questions all score >= 0.524, but in-scope questions phrased in Arabic or Tunisian
# Derja go lower (internship/PFE in Arabic 0.47, "كيفاش ناخذ منحة جامعية؟" 0.44) while
# out-of-scope ones reach 0.46 (weather in Arabic) -- the ranges overlap, no threshold separates
# them. A wrong "I don't know" is worse than letting a question through: the LLM itself correctly
# declined all 7 out-of-scope neighbours tested (passport, visa, driving licence...). So this only
# short-circuits the clearly absurd (restaurants 0.41, couscous in Arabic 0.37); everything above
# relies on the generation prompt's "say so if the sources don't answer" rule.
DEFAULT_MIN_DENSE_SCORE = 0.42


def _embedding_fingerprint(embed_model: str, texts: list[str]) -> str:
    h = hashlib.sha256(f"{embed_model}|normalize=True".encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


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
        self._bm25 = None      # lazy: only built if search(hybrid=True) is used
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
        # The cache used to be validated by row count alone: a stale file (530 rows vs 533 chunks,
        # from an outdated dvc.lock) made the server silently re-embed for many minutes, and edited
        # chunks with an unchanged count would have been served with the WRONG embeddings, no
        # warning at all. The .sig file holds a fingerprint of the model + the exact embedded texts.
        sig_path = cache_path.with_suffix(".sig")
        fingerprint = _embedding_fingerprint(embed_model, [c["text"] for c in self.chunks])
        mat = None
        if use_cache and cache_path.exists():
            mat = np.load(cache_path)
            saved = sig_path.read_text(encoding="utf-8").strip() if sig_path.exists() else None
            if mat.shape[0] != len(self.chunks) or (saved is not None and saved != fingerprint):
                print(f"WARNING: {cache_path.name} doesn't match {chunks_path.name} (rows {mat.shape[0]} vs "
                      f"{len(self.chunks)} chunks, fingerprint {'ok' if saved == fingerprint else 'differs'}) "
                      "-- re-embedding everything (slow on CPU). Fix for good: `dvc pull`, or "
                      "`dvc repro embed_kb` then `dvc push`.", flush=True)
                mat = None
        if mat is None:
            mat = np.asarray(self.model.encode([c["text"] for c in self.chunks], normalize_embeddings=True,
                                               show_progress_bar=True), dtype=np.float32)
            if use_cache:
                np.save(cache_path, mat)
        if use_cache and (not sig_path.exists() or sig_path.read_text(encoding="utf-8").strip() != fingerprint):
            # Also covers a cache from before .sig files existed: its row count matched, so it's
            # adopted (as it always was) -- every later change to the chunks is then caught.
            sig_path.write_text(fingerprint + "\n", encoding="utf-8")
        self.matrix = mat

    def _embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.rerank_model_name)
        return self._reranker

    def _get_bm25(self):
        """Same BM25 formula/tokenization as scripts/eval/eval_retrieval.py (kept in sync by
        hand: that script is the standalone eval floor, this is the live-search copy used for
        hybrid fusion) -- built once per Retriever instance, pure Python/regex, no model."""
        if self._bm25 is None:
            tok = lambda s: re.findall(r"\w+", s.lower())
            toks = [tok(c["text"]) for c in self.chunks]
            self._bm25 = {
                "toks": toks,
                "tf": [collections.Counter(t) for t in toks],
                "df": collections.Counter(t for ts in toks for t in set(ts)),
                "avg_len": sum(map(len, toks)) / max(len(toks), 1),
            }
        return self._bm25

    def _bm25_order(self, query: str, lang=None, domain=None, subtopic=None) -> list[int]:
        """Chunk indices ranked by BM25 score (best first), restricted to chunks matching the
        lang/domain/subtopic filters -- same filters search() applies to the dense ranking, so
        the two rankings being fused in _rrf_fuse cover the same candidate universe."""
        b = self._get_bm25()
        N = len(self.chunks)
        q_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for i, c in enumerate(self.chunks):
            if lang and c["lang"] != lang: continue
            if domain and c["domain"] != domain: continue
            if subtopic and c["subtopic"] != subtopic: continue
            tf, dl = b["tf"][i], len(b["toks"][i])
            s = sum(math.log(1 + (N - b["df"][t] + .5) / (b["df"][t] + .5)) * tf[t] * 2.2 /
                    (tf[t] + 1.2 * (.25 + .75 * dl / b["avg_len"]))
                    for t in q_terms if t in tf)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    @staticmethod
    def _rrf_fuse(*orders: list[int], k: int = RRF_K) -> list[int]:
        """Reciprocal Rank Fusion: combine any number of ranked-index lists into one ranking by
        summing 1/(k + rank) per list (0 contribution from a list an index is absent from) --
        robust to the two input rankings living on incomparable scales (cosine similarity vs.
        a BM25 score), unlike averaging raw scores would be."""
        scores = collections.defaultdict(float)
        for order in orders:
            for rank, idx in enumerate(order, start=1):
                scores[idx] += 1.0 / (k + rank)
        return [idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    def _cache_key(self, query, k, lang, domain, subtopic, rerank, candidate_k, hybrid) -> str:
        raw = "|".join(str(x) for x in (_norm_query(query), k, lang, domain, subtopic, rerank, candidate_k, hybrid))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _save_query_cache(self):
        payload = {"chunks_sig": self._chunks_sig, "entries": self._query_cache}
        self.query_cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _apply_min_score(hits: list["Hit"], rerank: bool, min_score: float | None,
                         min_dense_score: float | None = None) -> list["Hit"]:
        if not rerank:
            if min_dense_score is not None and hits and hits[0].score < min_dense_score:
                return []
            return hits
        if min_score is None or not hits or hits[0].score >= min_score:
            return hits
        if hits[0].bi_score is not None and hits[0].bi_score >= BI_SCORE_OVERRIDE:
            return hits  # reranker scored it low, but the bi-encoder is confident -- see BI_SCORE_OVERRIDE
        return []

    def search(self, query: str, k: int = 5, lang: str | None = None,
               domain: str | None = None, subtopic: str | None = None,
               rerank: bool = True, candidate_k: int = 20, hybrid: bool = False,
               min_score: float | None = DEFAULT_MIN_RERANK_SCORE,
               min_dense_score: float | None = None) -> list[Hit]:
        """rerank=True: fetch `candidate_k` candidates with the fast bi-encoder, then rescore
        them with a cross-encoder (query, chunk_text pairs) and return the top `k` by that
        score -- slower per query but typically more precise, especially at k=1.

        hybrid=True: candidates come from Reciprocal Rank Fusion of the dense ranking and a
        BM25 (lexical) ranking instead of dense similarity alone -- see RRF_K above for why.
        Still goes through the same rerank step afterwards when rerank=True; hybrid only changes
        which candidates reach the reranker (benchmarked: lower than dense-only, not the default).

        min_score: abstain (return []) when the top hit's score falls below this -- a question
        with no real match in the KB (or genuinely ambiguous) scores low; forcing an answer
        anyway is how a future generation layer would end up confidently citing the wrong
        procedure. Only applied when rerank=True (see DEFAULT_MIN_RERANK_SCORE); pass
        min_score=None to always return the top k regardless of confidence.

        An exact repeat of the same call (same normalized query, k, lang, domain, subtopic,
        rerank, candidate_k, hybrid -- min_score is applied after the cache lookup, so it's not
        part of the key) is served from query_cache.json without touching the embedding model,
        BM25 or the reranker."""
        cache_key = self._cache_key(query, k, lang, domain, subtopic, rerank, candidate_k, hybrid)
        if self.use_query_cache and cache_key in self._query_cache:
            hits = [Hit(**h) for h in self._query_cache[cache_key]]
            return self._apply_min_score(hits, rerank, min_score, min_dense_score)

        qv = np.asarray(self._embed([query])[0], dtype=np.float32)
        sims = self.matrix @ qv
        dense_order = np.argsort(-sims)
        if hybrid:
            # Cap each side's contribution to a bounded window before fusing -- RRF over the
            # full corpus would let a near-zero-relevance BM25 match (score > 0 from one stray
            # shared word) still nudge the ranking; capping to a few times candidate_k keeps
            # fusion to each side's genuinely top candidates.
            window = candidate_k * 3
            bm25_order = self._bm25_order(query, lang, domain, subtopic)[:window]
            order = self._rrf_fuse(list(dense_order[:window]), bm25_order)
            # hits below still get h.score = raw dense cosine (not the RRF score) even though
            # `order` is RRF-ranked -- harmless when rerank=True (the default; h.score is
            # overwritten by the cross-encoder below regardless of how the candidate was
            # found), but means hybrid=True, rerank=False hits are RRF-ordered with a
            # dense-cosine score label, not an RRF one. Fine for now since hybrid is meant to
            # feed the reranker, not stand alone -- revisit if that changes.
        else:
            order = dense_order
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
        return self._apply_min_score(hits, rerank, min_score, min_dense_score)

    def expand(self, hit: Hit) -> str | None:
        """Full procedure text for a hit that came from a multi-chunk document (small-to-big
        retrieval). Returns None for a hit whose chunk already IS the whole document."""
        parent = self.parents.get((hit.doc_id, hit.lang))
        return parent["text"] if parent else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build-embeddings", action="store_true",
                     help="only (re)build data/rag/chunks.<model>.npy + .sig and exit (dvc.yaml: embed_kb)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--lang")
    ap.add_argument("--domain")
    ap.add_argument("--subtopic")
    ap.add_argument("--expand", action="store_true", help="also print the full parent procedure for the top hit")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                     help="skip the cross-encoder rerank stage (on by default: +0.06 hit@1 on the eval set)")
    ap.add_argument("--hybrid", action="store_true",
                     help="fuse dense + BM25 candidate rankings (Reciprocal Rank Fusion) before reranking")
    ap.add_argument("--no-query-cache", dest="query_cache", action="store_false",
                     help="don't read/write data/rag/query_cache.json for this call")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_RERANK_SCORE,
                     help=f"abstain if the top (reranked) hit scores below this (default {DEFAULT_MIN_RERANK_SCORE})")
    ap.add_argument("--no-min-score", dest="min_score", action="store_const", const=None,
                     help="always return the top k, regardless of confidence")
    a = ap.parse_args()
    if not a.build_embeddings and not a.query:
        ap.error("a query is required unless --build-embeddings is used")

    r = Retriever(use_query_cache=a.query_cache)
    if a.build_embeddings:
        print(f"embeddings ready: {r.matrix.shape[0]} chunks x {r.matrix.shape[1]} dims")
        return
    hits = r.search(a.query, k=a.k, lang=a.lang, domain=a.domain, subtopic=a.subtopic,
                     rerank=a.rerank, hybrid=a.hybrid, min_score=a.min_score)
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
