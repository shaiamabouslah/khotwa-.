"""Canonical-dedup pass: when multiple active documents share an identical content_hash,
keep one (the canonical copy) and mark the rest status=removed, recording which doc replaced
them via a new `duplicate_of:` front-matter field so provenance isn't silently lost.

Confirmed by hand (see session notes) that the bulk of these are genuine source-side
duplication: CNSS republishes the exact same procedure text (e.g. "Attestation relative a la
situation d'un travailleur non salarie...") across several near-identical "secteur" URLs
(non agricole 1-6, agricole 1-3, peche 1-3) -- same information, different entry point. A RAG
index gains nothing from embedding the same paragraph 10 times and loses relevance ranking
precision by doing so.

The canonical pick is deterministic (shortest relative path, then alphabetical) so re-running
this script is idempotent.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/dedupe_corpus.py            # dry run, report only
    PYTHONPATH=. .venv/Scripts/python.exe scripts/dedupe_corpus.py --apply    # write status=removed
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

HASH_RE = re.compile(r"^content_hash: (sha256:\w+)$", re.M)
STATUS_RE = re.compile(r"^status: (\w+)$", re.M)
DOC_ID_RE = re.compile(r"^doc_id: (.+)$", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    by_hash = defaultdict(list)
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        h = HASH_RE.search(text)
        s = STATUS_RE.search(text)
        if h and s and s.group(1) == "active":
            by_hash[h.group(1)].append(p)

    dupes = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    total_removed = 0
    for h, paths in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
        paths = sorted(paths, key=lambda p: (len(p.relative_to(ROOT).as_posix()), p.as_posix()))
        canonical, rest = paths[0], paths[1:]
        canonical_text = canonical.read_text(encoding="utf-8")
        canonical_doc_id_m = DOC_ID_RE.search(canonical_text)
        canonical_doc_id = canonical_doc_id_m.group(1) if canonical_doc_id_m else canonical.as_posix()

        print(f"KEEP  {canonical.relative_to(ROOT).as_posix()}")
        for p in rest:
            rel = p.relative_to(ROOT).as_posix()
            print(f"  DROP {rel}")
            if args.apply:
                text = p.read_text(encoding="utf-8")
                text = STATUS_RE.sub("status: removed", text, count=1)
                if "duplicate_of:" not in text:
                    text = STATUS_RE.sub(
                        f"status: removed\nduplicate_of: {canonical_doc_id}", text, count=1)
                p.write_text(text, encoding="utf-8")
            total_removed += 1

    print(f"\n{len(dupes)} duplicate group(s), {total_removed} doc(s) "
          f"{'marked removed' if args.apply else 'would be marked removed (dry run)'}")
    if not args.apply:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
