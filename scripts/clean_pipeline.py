"""The ONE cleaning pipeline for data/corpus/ -- no LLM, no Ollama. Run this single script,
single command, to bring the corpus up to date after a crawl, populate kb/ from it, and
prepare RAG-ready chunks. Consolidates every separate cleaning/prep script that has existed
for this project (audit_noise, dedupe_corpus, mark_nav_stubs, fix_subtopic_labels,
resync_catalog, the fine-grained scripts/fixes/* pass, rag_prepare, plus the now-dropped
Ollama-based llm_classify/llm_clean) into one pipeline, so there's one thing to run, not many
files scattered across folders to remember and re-run in the right order.

Steps, always in this order:
  1. fine-grained document fixes (scripts/fixes/run_all.py): unreadable text quarantined as
     rag_exclude, layout junk stripped, wrong `lang` labels corrected from actual content,
     link noise moved to front matter, doc types inferred, generic titles rebuilt from the
     breadcrumb, stale `source_updated` flagged, catalog fields resynced from front matter.
     Each fix is its own file under scripts/fixes/ (one fix = one file, easy to audit or
     disable individually) but this step is what runs them all, in order, to a fixed point.
  2. remove link-only / pure-navigation stub documents (status=removed)
     -- automatic, not a hand-curated list: a doc with ~0 real prose (just links) is removed
     UNLESS it's guarded (contains a phone number, an email, or a price -- an address/contact
     card is short on purpose and is real information, not noise).
  3. deduplicate: exact-hash and near-duplicate (Jaccard on word shingles) documents, keeping
     the longest as canonical (status=removed on the rest, with `duplicate_of` for traceability)
  4. reclassify domain/subtopic with the rule-based keyword Classifier (crawler/classify.py,
     kb/taxonomy.yaml) -- no LLM call, deterministic, re-run any time the taxonomy keywords change
  5. resync catalog.jsonl and sync_state.json from the corpus files' own current front matter
     (the single source of truth after steps 1-4 edited files directly) -- copies every
     front-matter field generically, not a hand-maintained list, so a field a fix script adds
     later (e.g. rag_exclude) is never silently dropped from the catalog.
  6. populate kb/<domain>/<subtopic>/<id>.<lang>.md from every cleaned, classified, active
     corpus doc -- NO forced 8-section template (that approach was tried and rejected: it kept
     producing empty placeholder sections and misrouted bold-label content into the wrong
     section). The body is the doc's own cleaned content as-is. Hand-written kb/ files are
     never touched or overwritten; only files this step itself generated (marked
     `generated: true`) are refreshed or pruned on a later run.
  7. prepare RAG chunks (scripts/rag_prepare.py): kb/ -> data/rag/chunks.jsonl, with its own
     independent language/quality gate and near-duplicate check as a second safety net, plus
     data/rag/quarantine.jsonl and review_dupes.json for anything it held back.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/clean_pipeline.py            # dry run, report only
    PYTHONPATH=. .venv/Scripts/python.exe scripts/clean_pipeline.py --apply    # write changes
    PYTHONPATH=. .venv/Scripts/python.exe scripts/clean_pipeline.py --apply --skip-fixes --skip-rag
        # steps 2-6 only (the original corpus/kb cleaning, skip the fine-grained fixes and RAG prep)
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crawler.classify import Classifier, normalize  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
KB = ROOT / "kb"
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
STATUS_RE = re.compile(r"^status: \w+$", re.M)
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

# -- step 1: link-only stub detection -------------------------------------------------------
PHONE_RE = re.compile(r"(\+?\d[\d .\-]{6,}\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PRICE_RE = re.compile(r"\d+[,.]\d{2,3}\s*(d|dt|dinars?|tnd)\b", re.I)


def real_word_count(body: str) -> int:
    """Visible prose only: link targets stripped (kept as link text), markdown syntax
    stripped. A page that is just a menu of links scores near zero here even though its raw
    word count (counting URLs) looks substantial."""
    text = LINK_RE.sub(r"\1", body)
    text = re.sub(r"[|\-:>#*]", " ", text)
    return len(text.split())


def is_address_card(body: str) -> bool:
    """A short page with a phone/email/price is real contact/pricing information, not a nav
    stub -- never auto-remove it just for being short (confirmed false-positive risk earlier
    this session on annuaire_administration / equivalence_rnu contact pages)."""
    return bool(PHONE_RE.search(body) or EMAIL_RE.search(body) or PRICE_RE.search(body))


def find_link_only_stubs(min_words: int = 30):
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        if "status: active" not in front:
            continue
        if real_word_count(body) < min_words and not is_address_card(body):
            yield p, front, body


# -- step 2: dedup ----------------------------------------------------------------------------
def shingles(text: str, n: int = 5) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


HASH_RE = re.compile(r"^content_hash: (sha256:\w+)$", re.M)
DOC_ID_RE = re.compile(r"^doc_id: (.+)$", re.M)


def dedup_pass():
    by_hash = defaultdict(list)
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, _ = m.groups()
        if "status: active" not in front:
            continue
        h = HASH_RE.search(front)
        if h:
            by_hash[h.group(1)].append(p)

    exact_dupes = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    removals = []  # (path, canonical_doc_id)
    for h, paths in exact_dupes.items():
        paths = sorted(paths, key=lambda p: (len(p.relative_to(ROOT).as_posix()), p.as_posix()))
        canonical, rest = paths[0], paths[1:]
        canon_text = canonical.read_text(encoding="utf-8")
        canon_id_m = DOC_ID_RE.search(canon_text)
        canon_id = canon_id_m.group(1) if canon_id_m else canonical.as_posix()
        for p in rest:
            removals.append((p, canon_id))

    # near-dup pass (Jaccard), scoped per site+lang so unrelated sites never merge
    remaining = [p for p in CORPUS.rglob("*.md")
                 if p not in {r[0] for r in removals}
                 and "status: active" in p.read_text(encoding="utf-8", errors="ignore")]
    by_site_lang = defaultdict(list)
    for p in remaining:
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        site = re.search(r"^site: (.+)$", front, re.M)
        lang = re.search(r"^lang: (.+)$", front, re.M)
        key = (site.group(1) if site else "?", lang.group(1) if lang else "?")
        by_site_lang[key].append((p, body))

    for key, docs in by_site_lang.items():
        if len(docs) < 2:
            continue
        docs = sorted(docs, key=lambda d: -len(d[1]))
        shingle_cache = [shingles(b) for _, b in docs]
        used = [False] * len(docs)
        for i in range(len(docs)):
            if used[i]:
                continue
            used[i] = True
            canon_text = docs[i][0].read_text(encoding="utf-8")
            canon_id_m = DOC_ID_RE.search(canon_text)
            canon_id = canon_id_m.group(1) if canon_id_m else docs[i][0].as_posix()
            for j in range(i + 1, len(docs)):
                if used[j]:
                    continue
                if shingle_cache[i] and shingle_cache[j]:
                    jacc = len(shingle_cache[i] & shingle_cache[j]) / len(shingle_cache[i] | shingle_cache[j])
                    if jacc >= threshold_near_dup:
                        used[j] = True
                        removals.append((docs[j][0], canon_id))
    return removals


threshold_near_dup = 0.8


# -- step 3: reclassify (rule-based, no LLM) --------------------------------------------------
def reclassify_all(state: dict, path_to_key: dict):
    taxonomy = yaml.safe_load((ROOT / "kb" / "taxonomy.yaml").read_text(encoding="utf-8"))
    classifier = Classifier(taxonomy)
    sites = yaml.safe_load((ROOT / "sources.yaml").read_text(encoding="utf-8"))["sites"]
    changed = 0
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        meta = yaml.safe_load(front)
        if meta.get("status") != "active" or meta.get("rag_exclude") or not body.strip():
            continue
        site = sites.get(meta.get("site"), {})
        old = (meta.get("domain_hint"), meta.get("subtopic_hint"))
        hint = classifier.classify(meta.get("title", ""), meta.get("breadcrumb") or [], body, site.get("domain"))
        new = (hint.domain, hint.subtopic)
        if new != old:
            meta["domain_hint"], meta["subtopic_hint"] = hint.domain, hint.subtopic
            meta["hint_score"], meta["hint_keywords"] = hint.score, hint.matched[:6]
            new_front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()
            p.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")
            changed += 1
        rel = p.relative_to(ROOT).as_posix()
        key = path_to_key.get(rel)
        if key and "meta" in state["pages"][key]:
            state["pages"][key]["meta"]["domain_hint"] = meta.get("domain_hint")
            state["pages"][key]["meta"]["subtopic_hint"] = meta.get("subtopic_hint")
            state["pages"][key]["meta"]["hint_score"] = meta.get("hint_score")
            state["pages"][key]["meta"]["hint_keywords"] = meta.get("hint_keywords")
    return changed


# -- step 4: resync catalog/state --------------------------------------------------------------
def resync(state: dict):
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        meta = yaml.safe_load(m.group(1))
        rel = p.relative_to(ROOT).as_posix()
        key = None
        for k, page in state["pages"].items():
            if page["path"] == rel:
                key = k
                break
        if key is None:
            doc_id = meta.get("doc_id")
            key = f"{doc_id}:{meta.get('lang')}" if doc_id else rel
            state["pages"][key] = {"site": meta.get("site"), "path": rel, "url": meta.get("url"),
                                    "hash": None, "status": meta.get("status", "active"),
                                    "last_checked": meta.get("last_changed"), "meta": {}}
        page = state["pages"][key]
        page["status"] = meta.get("status", page.get("status"))
        page.setdefault("meta", {})
        # Copy every front-matter field, not a hand-maintained whitelist -- a fixed field list
        # silently drops whatever a new fix script adds (found: content_hash was missing from
        # every catalog row because it was never in this list, same for rag_exclude etc. added
        # by scripts/fixes/*.py). "status" is tracked separately above.
        page["meta"].update({k: v for k, v in meta.items() if k != "status"})

    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    ghosts = 0
    with (CORPUS / "catalog.jsonl").open("w", encoding="utf-8") as f:
        for key, page in sorted(state["pages"].items()):
            if not (ROOT / page["path"]).exists():
                ghosts += 1
                continue
            row = {**page.get("meta", {}), "path": page["path"], "status": page["status"],
                   "last_checked": page.get("last_checked")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return ghosts


# -- step 5: populate kb/ (no forced section template) ---------------------------------------
SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 70) -> str:
    s = SLUG_RE.sub("-", text.lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    if len(s) > max_len:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        s = s[:max_len].rstrip("-") + "-" + h
    return s


def build_kb_entries(taxonomy: dict):
    """Group every cleaned, classified, active corpus doc by a stable kb id derived from its
    doc_id (shared by a doc's fr/ar variants, so both languages land on the same id and pair
    up as translations automatically). Returns {kb_id: {lang: (meta, body, domain, subtopic)}}."""
    domains = taxonomy["domains"]
    kb_languages = set(taxonomy["languages"])  # kb/ only holds fr+ar -- other corpus langs
    # (e.g. "en" foreign-university pages picked up by a keyword hint) are left out of kb/.
    by_id = defaultdict(dict)
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        meta = yaml.safe_load(front)
        if meta.get("status") != "active" or not body.strip():
            continue
        lang = meta.get("lang")
        if lang not in kb_languages:
            continue
        domain, subtopic = meta.get("domain_hint"), meta.get("subtopic_hint")
        if domain not in domains or subtopic not in domains[domain].get("subtopics", {}):
            continue
        doc_id = meta.get("doc_id") or p.as_posix()
        kb_id = f"{subtopic}-{slugify(doc_id)}"
        # Demote a leading H1 (the extractor sometimes duplicates the title as a "# " line,
        # sometimes indented -- an indented "#" is invisible to a bare "^#" regex) -- the kb
        # contract forbids top-level headings in the body.
        body = re.sub(r"^[ \t]*#(?!#)[ \t]+", "## ", body.strip(), flags=re.M)
        by_id[kb_id][lang] = (meta, body, domain, subtopic)
    return by_id


def populate_kb(apply: bool):
    taxonomy = yaml.safe_load((KB / "taxonomy.yaml").read_text(encoding="utf-8"))
    by_id = build_kb_entries(taxonomy)
    today = dt.date.today().isoformat()

    kept_paths = set()
    written = skipped_handwritten = 0
    for kb_id, langs in by_id.items():
        all_langs = sorted(langs)
        for lang, (meta, body, domain, subtopic) in langs.items():
            out_path = KB / domain / subtopic / f"{kb_id}.{lang}.md"
            kept_paths.add(out_path)
            if out_path.exists():
                existing = out_path.read_text(encoding="utf-8", errors="ignore")
                m = FRONTMATTER_RE.match(existing)
                if not (m and "generated: true" in m.group(1)):
                    # Either no front matter, or front matter without the `generated: true`
                    # marker -> hand-written, never overwrite it.
                    skipped_handwritten += 1
                    continue
            if not apply:
                written += 1
                continue
            keywords = meta.get("hint_keywords") or taxonomy["domains"][domain]["subtopics"][subtopic].get("keywords") or []
            out_front = {
                "id": kb_id,
                "title": meta.get("title") or "",
                "domain": domain,
                "subtopic": subtopic,
                "lang": lang,
                "translations": [l for l in all_langs if l != lang],
                "org": meta.get("org") or "",
                "keywords": keywords,
                "sources": [{
                    "url": meta.get("url"),
                    "breadcrumb": meta.get("breadcrumb") or [],
                    **({"source_updated": meta["source_updated"]} if meta.get("source_updated") else {}),
                }],
                "last_checked": today,
                "last_changed": str(meta.get("last_changed") or today)[:10],
                "status": "active",
                "generated": True,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            front_yaml = yaml.safe_dump(out_front, allow_unicode=True, sort_keys=False, width=1000).strip()
            out_path.write_text(f"---\n{front_yaml}\n---\n\n{body}\n", encoding="utf-8")
            written += 1

    stale = 0
    if apply:
        for p in KB.rglob("*.md"):
            if p.parent.name == "_templates" or p.name == "README.md" or p in kept_paths:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            m = FRONTMATTER_RE.match(text)
            if m and "generated: true" in m.group(1):
                p.unlink()
                stale += 1
    else:
        for p in KB.rglob("*.md"):
            if p.parent.name == "_templates" or p.name == "README.md" or p in kept_paths:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            m = FRONTMATTER_RE.match(text)
            if m and "generated: true" in m.group(1):
                stale += 1
    return written, stale, skipped_handwritten


def mark_removed(path: Path, front: str, body: str, duplicate_of: str | None = None):
    new_front = STATUS_RE.sub("status: removed", front, count=1)
    if duplicate_of and "duplicate_of:" not in new_front:
        new_front += f"\nduplicate_of: {duplicate_of}"
    path.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")


def run_fine_grained_fixes(apply: bool):
    cmd = [sys.executable, str(ROOT / "scripts" / "fixes" / "run_all.py")]
    if apply:
        cmd.append("--apply")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout, end="")
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"scripts/fixes/run_all.py failed (exit {r.returncode})")


def run_rag_prepare():
    cmd = [sys.executable, str(ROOT / "scripts" / "rag_prepare.py"), "--kb", str(KB), "--out", str(ROOT / "data" / "rag")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout, end="")
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"scripts/rag_prepare.py failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run, report only)")
    ap.add_argument("--skip-fixes", action="store_true", help="skip step 1 (scripts/fixes/run_all.py)")
    ap.add_argument("--skip-rag", action="store_true", help="skip step 7 (scripts/rag_prepare.py)")
    args = ap.parse_args()

    print("Step 1/7: fine-grained document fixes (scripts/fixes/)")
    if args.skip_fixes:
        print("  skipped (--skip-fixes)")
    else:
        run_fine_grained_fixes(args.apply)

    print("\nStep 2/7: link-only / pure-navigation stubs")
    stubs = list(find_link_only_stubs())
    print(f"  {len(stubs)} found")
    if args.apply:
        for p, front, body in stubs:
            mark_removed(p, front, body)

    print("\nStep 3/7: deduplication (exact hash + near-duplicate)")
    dupes = dedup_pass()
    print(f"  {len(dupes)} duplicate(s) found")
    if args.apply:
        for p, canon_id in dupes:
            text = p.read_text(encoding="utf-8")
            m = FRONTMATTER_RE.match(text)
            if m and "status: active" in m.group(1):
                mark_removed(p, m.group(1), m.group(2), duplicate_of=canon_id)

    if args.apply:
        print("\nStep 4/7: reclassify domain/subtopic (rule-based, no LLM)")
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        path_to_key = {p["path"]: k for k, p in state["pages"].items()}
        changed = reclassify_all(state, path_to_key)
        print(f"  {changed} doc(s) reclassified")

        print("\nStep 5/7: resync catalog.jsonl / sync_state.json")
        ghosts = resync(state)
        print(f"  done, {ghosts} ghost entrie(s) excluded from catalog.jsonl")
    else:
        print("\nSteps 4-5 (reclassify, resync) only run with --apply.")

    print("\nStep 6/7: populate kb/ from the cleaned corpus (no forced section template)")
    written, stale, skipped = populate_kb(args.apply)
    verb = "written" if args.apply else "would be written"
    print(f"  {written} file(s) {verb}, {stale} stale generated file(s) "
          f"{'removed' if args.apply else 'would be removed'}, "
          f"{skipped} hand-written file(s) left untouched")

    print("\nStep 7/7: prepare RAG chunks (scripts/rag_prepare.py)")
    if not args.apply:
        print("  only runs with --apply (needs the kb/ step 6 just wrote)")
    elif args.skip_rag:
        print("  skipped (--skip-rag)")
    else:
        run_rag_prepare()

    if not args.apply:
        print("\nDry run only -- re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
