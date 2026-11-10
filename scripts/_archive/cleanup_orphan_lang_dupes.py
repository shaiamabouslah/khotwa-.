"""Delete corpus files whose sync_state status is "removed" AND whose sibling
(same doc_id, different language) is currently active -- these are duplicate
artifacts (mostly from a since-fixed language mis-detection bug on bilingual
PDF forms), not legitimate historical content. A "removed" file with no active
sibling is left untouched (it's real historical record, per the project's
"never silently delete" policy).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"


def main():
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    pages = state["pages"]

    # group by doc_id (key without the trailing ":lang")
    by_doc = {}
    for key, page in pages.items():
        doc_id = key.rsplit(":", 1)[0]
        by_doc.setdefault(doc_id, []).append((key, page))

    to_delete = []
    for doc_id, variants in by_doc.items():
        removed = [(k, p) for k, p in variants if p.get("status") == "removed"]
        active = [(k, p) for k, p in variants if p.get("status") != "removed"]
        if removed and active:
            for k, p in removed:
                path = ROOT / p["path"]
                if path.exists():
                    to_delete.append((k, path))

    print(f"{len(to_delete)} orphaned duplicate file(s) to delete")
    for k, path in to_delete:
        print(f"  {path.relative_to(ROOT).as_posix()}")
        path.unlink()
        del pages[k]
        try:
            path.parent.rmdir()
        except OSError:
            pass

    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"removed {len(to_delete)} stale state entries")


if __name__ == "__main__":
    main()
