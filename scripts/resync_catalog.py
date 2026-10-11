"""Rebuild catalog.jsonl and sync_state.json's meta/status fields from the corpus .md files'
own front matter -- the source of truth after several scripts this session (llm_classify.py,
mark_nav_stubs.py, dedupe_corpus.py) edited files directly without touching shared state, to
avoid concurrent-write races while a long background job was still running.

Safe to re-run any time; it only reads front matter and writes state/catalog.
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


def main():
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    path_to_key = {p["path"]: k for k, p in state["pages"].items()}

    updated = added = 0
    for md_path in CORPUS.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        meta = yaml.safe_load(m.group(1))
        rel = md_path.relative_to(ROOT).as_posix()
        key = path_to_key.get(rel)
        if not key:
            # On disk but missing from sync_state.json (stale state file, e.g. from a script
            # that moved/created files without going through sync_doc) -- add it rather than
            # silently drop it from catalog.jsonl.
            doc_id = meta.get("doc_id")
            key = f"{doc_id}:{meta.get('lang')}" if doc_id else rel
            path_to_key[rel] = key
            state["pages"][key] = {"site": meta.get("site"), "path": rel, "url": meta.get("url"),
                                    "hash": None, "status": meta.get("status", "active"),
                                    "last_checked": meta.get("last_changed"), "meta": {}}
            added += 1
        page = state["pages"][key]
        if page.get("status") != meta.get("status"):
            page["status"] = meta.get("status", page.get("status"))
            updated += 1
        page.setdefault("meta", {})
        for field in ("domain_hint", "subtopic_hint", "hint_score", "hint_keywords",
                      "hint_source", "doc_type", "duplicate_of"):
            if field in meta:
                page["meta"][field] = meta[field]
        # catalog.jsonl rows are built from `meta`, not the file -- keep these in sync too so
        # a freshly-added entry (or one whose front matter changed) renders a complete row.
        for field in ("doc_id", "site", "org", "trust", "url", "lang", "title", "section",
                      "breadcrumb", "words", "render", "discovered_via"):
            if field in meta:
                page["meta"][field] = meta[field]

    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    # sync_state.json keeps removed docs forever as a historical ledger (some point to paths
    # that were physically deleted by an old cleanup script) -- catalog.jsonl is "the single
    # entry point for the cleaning stage" per pipeline.py's own docstring, so it should only
    # describe documents that actually exist on disk right now.
    ghosts = 0
    with (CORPUS / "catalog.jsonl").open("w", encoding="utf-8") as f:
        for key, page in sorted(state["pages"].items()):
            if not (ROOT / page["path"]).exists():
                ghosts += 1
                continue
            row = {**page.get("meta", {}), "path": page["path"], "status": page["status"],
                   "last_checked": page.get("last_checked")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{updated} status field(s) resynced, {added} missing entrie(s) added to sync_state.json, "
          f"{ghosts} ghost entrie(s) (no file on disk) excluded from catalog.jsonl")


if __name__ == "__main__":
    main()
