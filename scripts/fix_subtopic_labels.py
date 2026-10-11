"""Retroactively re-run the domain/subtopic classifier (crawler.classify.Classifier) against
every already-crawled document, patching domain_hint/subtopic_hint/hint_score/hint_keywords
in front matter in place.

Same rationale as fix_doc_type_labels.py: sync_doc()'s change detection hashes the markdown
body only, so a metadata-only reclassification (same text, different hints) never gets
rewritten by a normal re-crawl -- this has to patch the corpus files, sync_state.json and
catalog.jsonl directly.

Run this after editing kb/taxonomy.yaml's keyword lists.
"""
import json
import re
from pathlib import Path

import yaml

from crawler.classify import Classifier

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def main():
    reg = yaml.safe_load((ROOT / "sources.yaml").read_text(encoding="utf-8"))
    sites = reg["sites"]
    taxonomy = yaml.safe_load((ROOT / "kb" / "taxonomy.yaml").read_text(encoding="utf-8"))
    classifier = Classifier(taxonomy)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
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
        title = meta.get("title", "")
        breadcrumb = meta.get("breadcrumb") or []
        old = (meta.get("domain_hint"), meta.get("subtopic_hint"), meta.get("hint_score"))
        hint = classifier.classify(title, breadcrumb, body, site.get("domain"))
        new = (hint.domain, hint.subtopic, hint.score)
        rel = md_path.relative_to(ROOT).as_posix()
        if new != old:
            meta["domain_hint"] = hint.domain
            meta["subtopic_hint"] = hint.subtopic
            meta["hint_score"] = hint.score
            meta["hint_keywords"] = hint.matched[:6]
            new_front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()
            md_path.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")
            changed.append((site_key, old, new, rel))
        key = path_to_key.get(rel)
        if key and "meta" in state["pages"][key]:
            state["pages"][key]["meta"]["domain_hint"] = hint.domain
            state["pages"][key]["meta"]["subtopic_hint"] = hint.subtopic
            state["pages"][key]["meta"]["hint_score"] = hint.score
            state["pages"][key]["meta"]["hint_keywords"] = hint.matched[:6]

    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    with (CORPUS / "catalog.jsonl").open("w", encoding="utf-8") as f:
        for key, page in sorted(state["pages"].items()):
            row = {**page.get("meta", {}), "path": page["path"], "status": page["status"],
                   "last_checked": page.get("last_checked")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(changed)} document(s) reclassified; sync_state.json and catalog.jsonl updated")
    for site, old, new, path in changed:
        print(f"  {site}: {old} -> {new}  ({path})")


if __name__ == "__main__":
    main()
