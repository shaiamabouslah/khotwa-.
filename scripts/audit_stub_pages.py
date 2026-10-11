"""Scan the whole corpus for likely "stub" documents: pages whose real prose content
(excluding front matter, headings, and link syntax) is too thin to be useful on its
own -- usually because the real information lives one click away on a linked page
that was never crawled.

Heuristic: strip YAML front matter, markdown link syntax (keep link TEXT, drop URLs),
heading markers, table pipes and HTML comments, then count remaining words. Below
MIN_WORDS is flagged. This is a signal to review, not an automatic delete.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
MIN_WORDS = 40

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.S)
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
TABLE_PIPE_RE = re.compile(r"\|")


def real_word_count(md: str) -> int:
    body = FRONTMATTER_RE.sub("", md)
    body = COMMENT_RE.sub("", body)
    body = LINK_RE.sub(r"\1", body)          # keep link text, drop target
    body = HEADING_RE.sub("", body)
    body = TABLE_PIPE_RE.sub(" ", body)
    body = re.sub(r"[\-*>#]", " ", body)
    return len(body.split())


def main():
    by_site = defaultdict(list)
    total = 0
    for md_file in sorted(CORPUS.rglob("*.md")):
        if md_file.name == "catalog.jsonl":
            continue
        total += 1
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        wc = real_word_count(text)
        if wc < MIN_WORDS:
            site = md_file.relative_to(CORPUS).parts[0]
            by_site[site].append((wc, md_file.relative_to(ROOT).as_posix()))

    flagged = sum(len(v) for v in by_site.values())
    print(f"{flagged} / {total} documents flagged as likely stubs (< {MIN_WORDS} real words)\n")
    for site in sorted(by_site, key=lambda s: -len(by_site[s])):
        docs = by_site[site]
        print(f"{site}: {len(docs)} flagged")
        for wc, path in sorted(docs):
            print(f"    {wc:3d}w  {path}")


if __name__ == "__main__":
    main()
