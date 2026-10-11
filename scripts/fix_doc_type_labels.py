"""Retroactively apply the content-based procedure detector (classify._looks_like_procedure)
to every already-crawled document, patching only the `doc_type:` front-matter line in place.

Re-crawling doesn't work for this: sync_doc()'s change detection hashes the markdown body
only, so a metadata-only reclassification (same text, different doc_type) never rewrites
the file -- confirmed empirically (re-crawl reported "unchanged" for known-wrong labels).
"""
import json
import re
from pathlib import Path

import yaml

from crawler.classify import doc_type as compute_doc_type

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
DOC_TYPE_LINE_RE = re.compile(r"^doc_type: (\w+)$", re.M)


def main():
    reg = yaml.safe_load((ROOT / "sources.yaml").read_text(encoding="utf-8"))
    sites = reg["sites"]
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    # path (posix, relative to ROOT) -> state key, so a file edit can find its state entry
    path_to_key = {p["path"]: k for k, p in state["pages"].items()}

    changed = []
    for md_path in CORPUS.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        meta = yaml.safe_load(front)
        site_key = meta.get("site")
        site = sites.get(site_key, {})
        current_type = meta.get("doc_type")
        is_pdf = meta.get("render") == "pdf"
        url = meta.get("url", "")
        title = meta.get("title", "")
        new_type = compute_doc_type(url, title, is_pdf, site.get("doc_types"), body)
        rel = md_path.relative_to(ROOT).as_posix()
        if new_type != current_type:
            new_text = DOC_TYPE_LINE_RE.sub(f"doc_type: {new_type}", text, count=1)
            md_path.write_text(new_text, encoding="utf-8")
            changed.append((site_key, current_type, new_type, rel))
        # Always resync state (even when the file was already correct from a prior run
        # of this script -- sync_state.json's cached meta doesn't get patched by that).
        key = path_to_key.get(rel)
        if key and "meta" in state["pages"][key]:
            state["pages"][key]["meta"]["doc_type"] = new_type

    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    with (CORPUS / "catalog.jsonl").open("w", encoding="utf-8") as f:
        for key, page in sorted(state["pages"].items()):
            row = {**page.get("meta", {}), "path": page["path"], "status": page["status"],
                   "last_checked": page.get("last_checked")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(changed)} document(s) relabeled; sync_state.json and catalog.jsonl updated")
    for site, old, new, path in changed:
        print(f"  {site}: {old} -> {new}  ({path})")


if __name__ == "__main__":
    main()
