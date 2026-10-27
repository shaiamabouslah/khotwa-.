#!/usr/bin/env python3
"""kb/ -> RAG-ready chunks.   Usage: python scripts/rag_prepare.py [--kb kb] [--out data/rag]

Stages (each one is deterministic except the chunker itself, which has 3 interchangeable
strategies -- see --strategy below):
  1 load+validate   front matter schema, path/metadata consistency
  2 normalize       unicode, bidi marks, list/heading repair, nav boilerplate, empty sections
  3 language gate   declared `lang` must match the script actually used
  4 quality gate    too short / too noisy  -> quarantine (never silently dropped)
  5 dedup           exact + near-duplicate (same lang); borderline pairs go to a review list
  6 chunk           --strategy heading (default, production): split on headings, group up to
                    TARGET_WORDS, pack oversize sections row/sentence-safely.
                    --strategy fixed: naive sliding word-count window, no heading-awareness --
                    the control group, to check heading-awareness is actually worth it.
                    --strategy semantic: embed paragraph units (sentence-transformers, default
                    BGE-M3) and cut where topic similarity drops, not at a fixed size.
                    All three reuse the same row/sentence-safe overlap packing.
  7 emit            chunks.jsonl, parents.jsonl (multi-chunk docs' full text, for "expand to
                    the whole procedure" hierarchical retrieval), quarantine.jsonl,
                    review_dupes.json, report.md

To compare the 3 strategies side by side (chunk quality + BM25/dense retrieval score), use
scripts/compare_chunking.py instead of calling this directly.
"""
import argparse, collections, hashlib, json, re, sys, unicodedata
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = ["id", "title", "domain", "subtopic", "lang", "sources", "last_changed"]
MIN_WORDS = 40            # after cleaning
TARGET_WORDS, MAX_WORDS, OVERLAP = 280, 420, 40
DUP_DROP, DUP_REVIEW = 0.95, 0.85

PLACEHOLDER = re.compile(r"^\s*(aucun(e)?|néant|n/?a|-|—|لا شيء|لا يوجد|بدون)\s*\.?\s*$", re.I)
NAV_LINE = re.compile(
    r"(cookies?|menu principal|aller au contenu|skip to|partager sur|facebook|twitter|linkedin|"
    r"newsletter|plan du site|mentions légales|tous droits réservés|nous contacter|اتصل بنا|مشاركة|خريطة الموقع)", re.I)
INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
MOJIBAKE = re.compile(r"Ã.|â€|ï¿½|\ufffd")


def words(s): return len(s.split())


def uid(fm): return f"{fm['id']}.{fm['lang']}"   # id is shared by FR/AR twins; id+lang is unique


def script_ratio(s):
    ar = len(re.findall(r"[\u0600-\u06FF]", s)); la = len(re.findall(r"[A-Za-zÀ-ÿ]", s))
    return ar / (ar + la + 1), la / (ar + la + 1)


# ───────────────────────── 1. load + validate ─────────────────────────
def load(path):
    path = Path(path)  # Convert manually provided path to Path object

    t = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n(.*)", t, re.S)

    if not m:
        return None, t, ["no_front_matter"]

    fm = yaml.safe_load(m.group(1)) or {}
    errs = [f"missing:{k}" for k in REQUIRED if not fm.get(k)]

    parts = path.parts
    if "kb" in parts:
        rel = parts[parts.index("kb") + 1:]
        if len(rel) >= 3:
            if fm.get("domain") != rel[0]:
                errs.append("domain_path_mismatch")
            if fm.get("subtopic") != rel[1]:
                errs.append("subtopic_path_mismatch")

    if fm.get("lang") not in ("fr", "ar"):
        errs.append("bad_lang")

    return fm, m.group(2), errs


# ───────────────────────── 2. normalize ─────────────────────────
def normalize(body):
    notes = []
    b = unicodedata.normalize("NFC", body)
    b = INVISIBLE.sub("", b).replace("\u00a0", " ").replace("\r", "")
    b = re.sub(r"^(\s*)(\d+)\.(?=[^\s\d])", r"\1\2. ", b, flags=re.M)         # "1.Texte" -> "1. Texte"
    b = re.sub(r"^(\s*)[•▪●·]\s*", r"\1- ", b, flags=re.M)                  # odd bullets
    b = re.sub(r"[ \t]+\n", "\n", b)
    # drop short navigation/boilerplate lines
    kept = []
    for line in b.split("\n"):
        if NAV_LINE.search(line) and words(line) <= 12 and not line.lstrip().startswith("#"):
            notes.append("nav_removed"); continue
        kept.append(line)
    b = "\n".join(kept)
    # drop sections whose whole body is a placeholder ("Aucun") but remember which fields were absent
    sections = re.split(r"(?m)^(?=#{2,4} )", b)
    out, missing = [], []
    for s in sections:
        m = re.match(r"(#{2,4} [^\n]*)\n+(.*)", s, re.S)
        if m and PLACEHOLDER.match(m.group(2).strip()):
            missing.append(re.sub(r"^#+\s*", "", m.group(1))); continue
        out.append(s)
    b = "".join(out)
    b = re.sub(r"\n{3,}", "\n\n", b).strip()
    # repeated identical consecutive headings
    b = re.sub(r"(?m)^(#{2,4} .+)\n+\1$", r"\1", b)
    return b, missing, notes


# ───────────────────────── 3/4. gates ─────────────────────────
def gates(fm, body):
    reasons = []
    ar, la = script_ratio(body)
    if fm["lang"] == "fr" and ar > 0.30: reasons.append("lang_mismatch:declared_fr_but_arabic")
    if fm["lang"] == "ar" and la > 0.50: reasons.append("lang_mismatch:declared_ar_but_latin")
    if words(body) < MIN_WORDS: reasons.append("too_short")
    if MOJIBAKE.search(body): reasons.append("mojibake")
    alnum = sum(c.isalnum() for c in body) / (len(body) + 1)
    if alnum < 0.55: reasons.append("noisy_symbols")
    return reasons


# ───────────────────────── 5. dedup ─────────────────────────
def shingles(b):
    w = re.findall(r"\w+", b.lower()); return set(zip(w, w[1:], w[2:]))


def jaccard(a, b): return len(a & b) / len(a | b) if a and b else 0.0


def dedup(docs):
    dropped, review = {}, []
    by_lang = collections.defaultdict(list)
    for d in docs: by_lang[d["fm"]["lang"]].append(d)
    for lang, ds in by_lang.items():
        # best first: longest, then most recently changed
        ds.sort(key=lambda d: (-words(d["body"]), str(d["fm"].get("last_changed", ""))), reverse=False)
        for d in ds: d["sh"] = shingles(d["body"])
        for i, a in enumerate(ds):
            if uid(a["fm"]) in dropped: continue
            for b in ds[i + 1:]:
                if uid(b["fm"]) in dropped: continue
                j = jaccard(a["sh"], b["sh"])
                if j >= DUP_DROP:
                    dropped[uid(b["fm"])] = {"duplicate_of": uid(a["fm"]), "jaccard": round(j, 3)}
                elif j >= DUP_REVIEW:
                    review.append({"a": uid(a["fm"]), "b": uid(b["fm"]), "jaccard": round(j, 3),
                                   "titles": [a["fm"]["title"], b["fm"]["title"]]})
    return dropped, review


# ───────────────────────── 6. chunking ─────────────────────────
def split_blocks(text):
    """paragraph-level blocks; numbered lists and tables stay together."""
    blocks, cur = [], []
    for line in text.split("\n"):
        if line.strip() == "":
            if cur: blocks.append("\n".join(cur)); cur = []
        else: cur.append(line)
    if cur: blocks.append("\n".join(cur))
    return blocks


def hard_split(u):
    w = u.split()
    return [" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS)]


def units_of(blk):
    if words(blk) <= MAX_WORDS: return [blk]
    parts = blk.split("\n") if "\n" in blk else re.split(r"(?<=[.!?؟])\s+", blk)
    out = []
    for u in parts:
        out += [u] if words(u) <= MAX_WORDS else hard_split(u)
    return out


def pack(blocks):
    chunks, cur, n = [], [], 0
    for blk in blocks:
        for u in units_of(blk):
            uw = words(u)
            if n + uw > MAX_WORDS and cur:
                chunks.append("\n\n".join(cur))
                # Overlap = whole trailing units (table rows / sentences), not a raw word-count
                # slice -- a blind tail-slice cuts mid-row/mid-sentence (measured: 87% of
                # non-final chunks on long headingless docs ended with no terminal punctuation).
                tail, tn = [], 0
                for prev in reversed(cur):
                    pw = words(prev)
                    if tn and tn + pw > OVERLAP:
                        break
                    tail.insert(0, prev)
                    tn += pw
                cur, n = tail, tn
            cur.append(u); n += uw
    if cur: chunks.append("\n\n".join(cur))
    return chunks


def chunk_heading(fm, body):
    """Structure-aware: split on ## / ### / #### headings, group consecutive sections up to
    TARGET_WORDS, pack oversize sections row/sentence-safely. Current production default."""
    title, org = fm["title"], fm.get("org", "")
    prefix = f"[{org}] " if org else ""
    # a whole procedure that fits stays ONE chunk: best unit for "how do I ..." questions
    if words(body) <= MAX_WORDS:
        return [{"section": title, "text": f"{prefix}{title}\n\n{body}"}]
    parts = [p for p in re.split(r"(?m)^(?=#{2,4} )", body) if p.strip()]
    secs, path = [], [title]
    for p in parts:
        m = re.match(r"(#{2,4}) ([^\n]*)\n?(.*)", p, re.S)
        if m:
            lvl, head = len(m.group(1)), m.group(2).strip()
            path = path[:max(lvl - 2, 0) + 1] + ([head] if head != title else [])
        secs.append((" > ".join(path), p.strip()))
    # greedy: group consecutive sections up to TARGET_WORDS; oversize sections are packed separately
    groups, cur, n = [], [], 0
    for sp, txt in secs:
        w = words(txt)
        if cur and n + w > TARGET_WORDS:
            groups.append(cur); cur, n = [], 0
        cur.append((sp, txt)); n += w
    if cur: groups.append(cur)
    out = []
    for g in groups:
        sp = g[0][0]
        for piece in pack(split_blocks("\n\n".join(t for _, t in g))):
            out.append({"section": sp, "text": f"{prefix}{sp}\n\n{piece}"})
    return out


def chunk_fixed(fm, body):
    """Naive baseline: ignores headings entirely, one sliding word-count window over the whole
    body. Row/sentence-safe packing is reused (same pack()), so this isolates exactly one
    variable against chunk_heading: does heading-awareness help, or is it not worth the
    complexity? "section" stays the title for every chunk since there's no structure to name."""
    title, org = fm["title"], fm.get("org", "")
    prefix = f"[{org}] " if org else ""
    if words(body) <= MAX_WORDS:
        return [{"section": title, "text": f"{prefix}{title}\n\n{body}"}]
    return [{"section": title, "text": f"{prefix}{title}\n\n{piece}"}
            for piece in pack(split_blocks(body))]


def _cosine(a, b):
    import numpy as np
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def chunk_semantic(fm, body, embed, threshold=0.45):
    """Embedding-based: split into paragraph-level units, embed each, greedily merge
    consecutive units into a chunk while they stay topically close (cosine similarity to the
    chunk's running mean embedding above `threshold`) and under MAX_WORDS; cut a new chunk on
    either a topic shift or the size cap. Needs an `embed(list[str]) -> list[vector]` callable
    (dependency injected so this module has no hard torch/sentence-transformers import)."""
    import numpy as np
    title, org = fm["title"], fm.get("org", "")
    prefix = f"[{org}] " if org else ""
    if words(body) <= MAX_WORDS:
        return [{"section": title, "text": f"{prefix}{title}\n\n{body}"}]
    units = [u for blk in split_blocks(body) for u in units_of(blk)]
    if len(units) <= 1:
        return [{"section": title, "text": f"{prefix}{title}\n\n{p}"} for p in pack(split_blocks(body))]
    vecs = embed(units)
    chunks, cur, cur_vecs, n = [], [], [], 0
    for u, v in zip(units, vecs):
        uw = words(u)
        if cur:
            mean_vec = np.mean(cur_vecs, axis=0)
            sim = _cosine(mean_vec, v)
            if n + uw > MAX_WORDS or sim < threshold:
                chunks.append("\n\n".join(cur))
                cur, cur_vecs, n = [], [], 0
        cur.append(u); cur_vecs.append(v); n += uw
    if cur: chunks.append("\n\n".join(cur))
    return [{"section": title, "text": f"{prefix}{title}\n\n{piece}"} for piece in chunks]


STRATEGIES = {"heading": chunk_heading, "fixed": chunk_fixed}  # "semantic" needs an embedder, wired in main()


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb"); ap.add_argument("--out", default="data/rag")
    ap.add_argument("--strategy", default="heading", choices=["heading", "fixed", "semantic"],
                     help="chunking strategy; 'semantic' needs --embed-model (sentence-transformers)")
    ap.add_argument("--embed-model", default="BAAI/bge-m3", help="only used by --strategy semantic")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.strategy == "semantic":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.embed_model)
        def chunker(fm, body): return chunk_semantic(fm, body, embed=lambda texts: model.encode(texts, normalize_embeddings=True))
    else:
        chunker = STRATEGIES[a.strategy]

    stats, quarantine, docs = collections.Counter(), [], []

    for p in sorted(Path(a.kb).rglob("*.md")):
        if p.name == "README.md" or "_templates" in p.parts: continue
        stats["files"] += 1
        fm, raw, errs = load(p)
        if errs and (fm is None or any(e.startswith("missing") or e == "bad_lang" for e in errs)):
            quarantine.append({"file": str(p), "reasons": errs}); continue
        body, missing, notes = normalize(raw)
        reasons = errs + gates(fm, body)
        if reasons:
            quarantine.append({"file": str(p), "id": uid(fm), "reasons": reasons, "words": words(body)}); continue
        docs.append({"path": p, "fm": fm, "body": body, "missing": missing, "notes": notes})

    dropped, review = dedup(docs)
    for i, d in dropped.items():
        quarantine.append({"id": i, "reasons": ["near_duplicate"], **d})
    kept = [d for d in docs if uid(d["fm"]) not in dropped]

    n_chunks, n_parents, cw = 0, 0, []
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f, \
         open(out / "parents.jsonl", "w", encoding="utf-8") as pf:
        for d in kept:
            fm = d["fm"]
            src = (fm.get("sources") or [{}])[0]
            doc_chunks = chunker(fm, d["body"])
            for k, c in enumerate(doc_chunks):
                rec = {
                    "chunk_id": f"{uid(fm)}#{k:02d}", "doc_id": fm["id"], "lang": fm["lang"],
                    "domain": fm["domain"], "subtopic": fm["subtopic"], "title": fm["title"],
                    "org": fm.get("org", ""), "section": c["section"], "url": src.get("url", ""),
                    "last_changed": str(fm.get("last_changed", "")), "missing_fields": d["missing"],
                    "text": c["text"], "n_chunks": len(doc_chunks),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_chunks += 1; cw.append(words(c["text"]))
            # Hierarchical retrieval ("small-to-big"): a chunk hit can be expanded to its full
            # procedure via doc_id+lang. Only worth storing for docs that actually got split --
            # a single-chunk doc's chunk text already IS the whole document.
            if len(doc_chunks) > 1:
                pf.write(json.dumps({"doc_id": fm["id"], "lang": fm["lang"], "title": fm["title"],
                                      "domain": fm["domain"], "subtopic": fm["subtopic"],
                                      "url": src.get("url", ""), "text": f"{fm['title']}\n\n{d['body']}"},
                                     ensure_ascii=False) + "\n")
                n_parents += 1
    (out / "quarantine.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in quarantine), encoding="utf-8")
    (out / "review_dupes.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")

    reasons = collections.Counter(r.split(":")[0] if r.startswith("lang") is False else r for q in quarantine for r in q["reasons"])
    cw.sort()
    rep = [f"# RAG prepare report", f"- files scanned: {stats['files']}", f"- kept docs: {len(kept)}",
           f"- chunks: {n_chunks}  (words p10/median/p90: {cw[len(cw)//10]}/{cw[len(cw)//2]}/{cw[9*len(cw)//10]})" if cw else "- chunks: 0",
           f"- parent docs (multi-chunk, expandable): {n_parents}",
           f"- quarantined: {len(quarantine)}", f"- borderline duplicate pairs to review: {len(review)}", "", "## Quarantine reasons"]
    rep += [f"- {k}: {v}" for k, v in reasons.most_common()]
    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    sys.exit(main())
