"""Second pass: find remaining <uuid>.ar.md / <uuid>.fr.md duplicate pairs under
document_library (bilingual PDF forms) directly on disk, and keep only the file
matching the URL's own /ar/ path convention -- same logic as pipeline.py's
url_lang_hint(), applied retroactively to pairs sync_state hasn't caught up on yet.
"""
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "cnss"

FILE_RE = re.compile(r"^(.*)\.(ar|fr)\.md$")


def main():
    groups = defaultdict(dict)  # base_path -> {lang: Path}
    for md in CORPUS.rglob("*.md"):
        if "document_library" not in md.as_posix():
            continue
        m = FILE_RE.match(md.name)
        if not m:
            continue
        base, lang = m.groups()
        groups[(md.parent, base)][lang] = md

    deleted = []
    for (parent, base), langs in groups.items():
        if "ar" not in langs or "fr" not in langs:
            continue
        is_ar_path = "\\ar\\" in str(parent) or "/ar/" in parent.as_posix()
        keep, drop = (langs["ar"], langs["fr"]) if is_ar_path else (langs["fr"], langs["ar"])
        deleted.append(drop)
        drop.unlink()

    print(f"deleted {len(deleted)} duplicate file(s)")
    for p in deleted:
        print(f"  {p.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
