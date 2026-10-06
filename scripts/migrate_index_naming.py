"""One-off migration: move data/corpus/<site>/.../<slug>/index.<lang>.md to
.../<slug>.<lang>.md, matching the new doc_identity() path scheme (no extra
'index' layer for whole-page documents). Split-page documents (section leaf
isn't "index") are untouched. Also updates data/state/sync_state.json so the
freshness check (`out.exists()`) keeps working without a full re-crawl.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"

INDEX_RE = re.compile(r"^index\.([a-z]{2})\.md$")


def main():
    dry_run = "--dry-run" in sys.argv
    moves = []  # (old_path, new_path)

    for md_file in CORPUS.rglob("index.*.md"):
        m = INDEX_RE.match(md_file.name)
        if not m:
            continue
        lang = m.group(1)
        slug_dir = md_file.parent
        new_path = slug_dir.parent / f"{slug_dir.name}.{lang}.md"
        moves.append((md_file, new_path))

    print(f"{len(moves)} file(s) to migrate")

    if dry_run:
        for old, new in moves[:20]:
            print(f"  {old.relative_to(ROOT)} -> {new.relative_to(ROOT)}")
        if len(moves) > 20:
            print(f"  ... and {len(moves) - 20} more")
        return

    old_to_new_rel = {}
    for old, new in moves:
        if new.exists():
            print(f"  SKIP (target exists): {old.relative_to(ROOT)}")
            continue
        old.rename(new)
        old_to_new_rel[old.relative_to(ROOT).as_posix()] = new.relative_to(ROOT).as_posix()
        try:
            old.parent.rmdir()  # only succeeds if now empty
        except OSError:
            pass

    print(f"moved {len(old_to_new_rel)} file(s)")

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        updated = 0
        for key, page in state.get("pages", {}).items():
            p = page.get("path")
            if p in old_to_new_rel:
                page["path"] = old_to_new_rel[p]
                updated += 1
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {updated} path(s) in sync_state.json")


if __name__ == "__main__":
    main()
