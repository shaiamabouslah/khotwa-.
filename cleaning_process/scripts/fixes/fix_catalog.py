"""Problem 7: catalog drift. Front matter is the source of truth: recompute `words` and re-sync every
catalog row (fills the missing content_hash / domain_hint / status). Run this LAST. All statuses."""
from _common import run, count_words


def fix(d):
    """catalog resync"""
    d.fm["words"] = count_words(d.body)
    drift = [k for k, v in d.fm.items() if k in d.row and d.row[k] != v]
    missing = [k for k in ("content_hash", "domain_hint", "subtopic_hint") if k in d.fm and k not in d.row]
    for k in missing: d.row[k] = d.fm[k]
    return ("fields: " + ",".join(drift + missing)) if (drift or missing or d.dirty()) else None


if __name__ == "__main__": run("catalog", fix, status=None, skip_excluded=False)
