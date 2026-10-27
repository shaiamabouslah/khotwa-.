"""Problem 1: generic titles ("Employeurs - CNSS" x57). A title shared by 5+ active docs of the same
site+language is rebuilt from the breadcrumb (or first heading). Old title kept as original_title."""
import re, collections
from _common import run, catalog

MIN_SHARED = 5
SHARED = collections.Counter((r["site"], r["lang"], r["title"].strip().lower()) for r in catalog() if r["status"] == "active")


def tidy(s):
    s = re.sub(r"[_\s]+", " ", s).strip()
    return s[:1].upper() + s[1:]


def fix(d):
    """generic titles"""
    t, bc = d.fm.get("title", ""), d.fm.get("breadcrumb") or []
    if SHARED[(d.fm.get("site"), d.fm.get("lang"), t.strip().lower())] < MIN_SHARED: return None
    new = tidy(bc[-1]) if bc and tidy(bc[-1]).lower() != t.lower() else ""
    if not new and (m := re.search(r"^#{1,4} (.+)$", d.body, re.M)): new = tidy(m[1])
    if not new or new.lower() == t.lower(): return None
    ctx = tidy(bc[1]) if len(bc) > 2 else ""                       # e.g. "Employeur" separates look-alikes
    if ctx and ctx.lower() not in new.lower(): new = f"{new} ({ctx})"
    d.fm["original_title"], d.fm["title"] = t, new
    return f"{t!r} -> {new!r}"


if __name__ == "__main__": run("titles", fix)
