"""Shared helpers for the corpus fixes. Every fix is: fix(doc) -> note | None.

* dry run by default, `--apply` writes (commit first: git is the backup)
* docs flagged rag_exclude are left untouched by every other fix
* never deletes data: removed/moved values go into front matter (original_title, nav_links, ...)
* content_hash is NEVER touched: the crawler uses it to detect source changes, rewriting it
  would make the next crawl overwrite our fixes. `words` is kept in sync instead.
"""
import argparse, collections, json, re, sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # samples can contain Arabic titles;
# Windows' default console codepage (cp1252) crashes on them otherwise

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/corpus/catalog.jsonl").exists())
sys.path.insert(0, str(ROOT))                       # lets fixes import crawler.*
CORPUS, CATALOG = ROOT / "data/corpus", ROOT / "data/corpus/catalog.jsonl"
LOG = ROOT / "data/state/fixes.jsonl"
FM = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
_VISIBLE = re.compile(r"\]\([^)]*\)|<https?://[^>]*>|https?://\S+")
_WORD = re.compile(r"\w*[^\W\d_]\w*")


def count_words(body): return len(_WORD.findall(_VISIBLE.sub("", body)))
def script_share(s):
    """(arabic share of letters, latin share of letters)"""
    s = _VISIBLE.sub("", s)                      # URLs are Latin letters but not language
    ar, la = len(re.findall(r"[\u0600-\u06FF]", s)), len(re.findall(r"[A-Za-zÀ-ÿ]", s))
    return ar / (ar + la + 1), la / (ar + la + 1)


class Doc:
    def __init__(self, path, row=None):
        self.path, self.row = path, row or {}
        self.raw = path.read_text(encoding="utf-8")
        m = FM.match(self.raw)
        self.fm, self.body = (yaml.safe_load(m[1]) or {}, m[2]) if m else ({}, self.raw)

    def dump(self):
        head = yaml.safe_dump(self.fm, allow_unicode=True, sort_keys=False, width=1000).strip()
        return f"---\n{head}\n---\n{self.body}"

    def dirty(self): return self.dump() != self.raw


def catalog():
    return [json.loads(l) for l in CATALOG.read_text(encoding="utf-8").splitlines() if l.strip()]


def run(name, fix, status="active", skip_excluded=True):
    ap = argparse.ArgumentParser(description=fix.__doc__)
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--site")
    a = ap.parse_args()
    rows = {r["path"]: r for r in catalog()}
    seen = changed = 0; by_site = collections.Counter(); samples = []; log = []
    for p in sorted(CORPUS.rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        d = Doc(p, rows.get(rel))
        if (status and d.fm.get("status") != status) or (a.site and d.fm.get("site") != a.site): continue
        if skip_excluded and d.fm.get("rag_exclude"): continue            # unreadable docs stay byte-for-byte
        seen += 1
        note = fix(d)
        if not note: continue
        changed += 1; by_site[d.fm.get("site")] += 1
        if len(samples) < 5: samples.append(f"  {rel.split('corpus/')[1][:75]}  ->  {note}")
        if a.apply:
            if d.dirty(): p.write_text(d.dump(), encoding="utf-8")
            if d.row: d.row.update({k: v for k, v in d.fm.items() if k in d.row or k in ("rag_exclude", "possibly_outdated")})
            log.append({"fix": name, "path": rel, "note": note})
    if a.apply and changed:
        CATALOG.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows.values()) + "\n", encoding="utf-8")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f: f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in log)
    print(f"[{name}] {'APPLIED' if a.apply else 'dry run'}: {changed}/{seen} docs changed  {dict(by_site.most_common(5))}")
    print("\n".join(samples))
