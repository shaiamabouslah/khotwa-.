"""Problem 4: link noise. A leading block of link-only list items is site navigation -> moved to
front matter (nav_links). Link-heavy bodies keep their anchor text, URLs move to front matter (links)."""
import re
from _common import run, count_words

LINK_ITEM = re.compile(r"^\s*[-*+] \[([^\]]+)\]\(([^)]+)\)\s*$")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def fix(d):
    """link noise"""
    lines, i, nav = d.body.split("\n"), 0, []
    while i < len(lines) and (not lines[i].strip() or LINK_ITEM.match(lines[i])):    # leading nav block
        if (m := LINK_ITEM.match(lines[i])): nav.append({"text": m[1], "url": m[2]})
        i += 1
    notes, body = [], d.body
    if len(nav) >= 3 and count_words("\n".join(lines[i:])) >= 40:                     # keep docs that are only links
        body = "\n".join(lines[i:]).lstrip("\n"); d.fm["nav_links"] = nav; notes.append(f"{len(nav)} nav links moved")
    if sum(len(m[0]) for m in LINK.finditer(body)) > 0.35 * max(len(body), 1):       # link-heavy -> text + urls aside
        links = [{"text": m[1], "url": m[2]} for m in LINK.finditer(body)]
        body = LINK.sub(lambda m: m[1], body); d.fm["links"] = links; notes.append(f"{len(links)} urls moved")
        if d.fm.get("doc_type") == "page": d.fm["doc_type"] = "directory"
    if not notes: return None
    d.body = body
    return ", ".join(notes)


if __name__ == "__main__": run("links", fix)
