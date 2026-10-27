"""Corpus quality gate: counts the known problems on usable docs (active, not rag_exclude).
Run BEFORE and AFTER the fixes and compare. Exit code 1 if a check exceeds its allowed maximum."""
import collections, re, sys
from _common import CORPUS, Doc, catalog, count_words, script_share, ROOT
from fix_unreadable import reason
from fix_layout_junk import flatten_tables, NAV_LINE, STAMP

LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
LINK_ITEM = re.compile(r"^\s*[-*+] \[[^\]]+\]\([^)]+\)\s*$")

docs = [d for d in (Doc(p) for p in CORPUS.rglob("*.md")) if d.fm.get("status") == "active"]
use = [d for d in docs if not d.fm.get("rag_exclude")]
titles = collections.Counter((d.fm["site"], d.fm["lang"], d.fm["title"].strip().lower()) for d in use)
C = collections.Counter()
for d in use:
    b, fm = d.body, d.fm
    ar, _ = script_share(b)
    lead = 0
    for l in b.split("\n"):
        if l.strip() and not LINK_ITEM.match(l): break
        lead += bool(LINK_ITEM.match(l))
    C["generic_title (shared by 5+)"] += titles[(fm["site"], fm["lang"], fm["title"].strip().lower())] >= 5
    C["wrong_language"] += (fm["lang"] in ("fr", "en") and ar > 0.6) or (fm["lang"] == "ar" and ar < 0.1 and len(b) > 300)
    C["unreadable_not_flagged"] += bool(reason(b))
    C["layout_junk (empty tables/nav)"] += flatten_tables(b) != b or bool(STAMP.search(b)) or any(NAV_LINE.match(l) for l in b.split("\n"))
    C["link_heavy (>35% links)"] += sum(len(m[0]) for m in LINK.finditer(b)) > 0.35 * max(len(b), 1)
    C["leading_nav_block"] += lead >= 3
    C["still_untyped_page"] += fm.get("doc_type") == "page"
    C["too_short (<40 words)"] += count_words(b) < 40
rows = catalog()
by_path = {r["path"]: r for r in rows}
C["catalog_drift"] = sum(any(by_path[d.path.relative_to(ROOT).as_posix()].get(k) != v for k, v in d.fm.items() if k in by_path[d.path.relative_to(ROOT).as_posix()]) for d in docs)
C["catalog_missing_hash"] = sum("content_hash" not in r for r in rows)
# allowed leftovers: ~23 letterhead PDF forms ("République tunisienne") have no breadcrumb/heading to build a title from
MAX = {"unreadable_not_flagged": 0, "wrong_language": 0, "layout_junk (empty tables/nav)": 0, "leading_nav_block": 0,
       "catalog_drift": 0, "catalog_missing_hash": 0, "generic_title (shared by 5+)": 30}
print(f"usable docs: {len(use)} / active {len(docs)}  |  rag_exclude: {len(docs) - len(use)}")
bad = 0
for k, v in C.items():
    flag = "info" if k not in MAX else "FAIL" if v > MAX[k] else "ok  "
    bad += flag == "FAIL"
    print(f"  {flag}  {k:34} {v}")
sys.exit(1 if bad else 0)
