#!/usr/bin/env python3
"""Dense retriever over data/rag/chunks.jsonl, using BGE-M3 (multilingual fr/en/ar).

Library usage:
    from retriever import Retriever
    r = Retriever()                              # loads data/rag/, embeds (cached to .npy)
    hits = r.search("Comment affilier un salarie a la CNSS ?", k=5)
    for h in hits:
        print(h.score, h.title, h.url)
    full_text = r.expand(hits[0])                 # None if hits[0] is already a whole doc

CLI usage (manual testing, one question at a time):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py "Comment affilier un salarie ?"
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --domain employment "..."
    PYTHONPATH=. .venv/Scripts/python.exe scripts/retriever.py --lang ar "..."
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = ROOT / "data" / "rag"
EMBED_MODEL = "BAAI/bge-m3"


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


class Retriever:
    def __init__(self, rag_dir: Path = RAG_DIR, embed_model: str = EMBED_MODEL, use_cache: bool = True):
        self.rag_dir = Path(rag_dir)
        self.embed_model_name = embed_model
        chunks_path = self.rag_dir / "chunks.jsonl"
        self.chunks = [json.loads(l) for l in chunks_path.open(encoding="utf-8")]
        if not self.chunks:
            raise ValueError(f"{chunks_path}: no chunks -- run clean_pipeline.py --apply first")

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

    def search(self, query: str, k: int = 5, lang: str | None = None,
               domain: str | None = None, subtopic: str | None = None) -> list[Hit]:
        qv = np.asarray(self._embed([query])[0], dtype=np.float32)
        sims = self.matrix @ qv
        order = np.argsort(-sims)
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
            if len(hits) == k:
                break
        return hits

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
    a = ap.parse_args()

    r = Retriever()
    hits = r.search(a.query, k=a.k, lang=a.lang, domain=a.domain, subtopic=a.subtopic)
    if not hits:
        print("no results")
        return
    for i, h in enumerate(hits):
        print(f"\n[{i + 1}] score={h.score:.3f}  {h.domain}/{h.subtopic}  [{h.lang}]  n_chunks={h.n_chunks}")
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
