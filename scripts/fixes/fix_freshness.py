"""Problem 8: possibly outdated content. Docs whose source_updated is older than the cutoff get
possibly_outdated: true (kept, so the RAG can show a caveat or rank them lower). Cutoff: --cutoff or 2022-01-01."""
import sys
from _common import run

CUTOFF = next((a.split("=")[1] for a in sys.argv if a.startswith("--cutoff=")), "2022-01-01")
sys.argv = [a for a in sys.argv if not a.startswith("--cutoff=")]


def fix(d):
    """freshness"""
    s = str(d.fm.get("source_updated") or "")
    if not s or s >= CUTOFF or d.fm.get("possibly_outdated"): return None
    d.fm["possibly_outdated"] = True
    return f"source_updated {s}"


if __name__ == "__main__": run("freshness", fix)
