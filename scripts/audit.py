"""Read-only, no-LLM triage of data/corpus/: two independent checks, run together or one at a
time. Neither modifies anything -- both are a signal to review (by hand, or by scripts/fixes/),
not an automatic fix. Merged from the former audit_noise.py + audit_stub_pages.py: same two
checks, same output, one file instead of two near-identical CLI scripts.

  noise   cheap markdown-noise triage (empty table pipes, link-only content, login/cookie-wall
          markers) on *active* docs only -- scopes a heavier cleaning pass to what needs it.
  stubs   real-prose word count (front matter/links/headings/tables stripped) on *every* doc
          regardless of status -- flags pages whose real content lives one click away on a
          linked page that was never crawled.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


# ---- noise: table/link-only/login-wall triage on active docs -------------------------------

STATUS_RE = re.compile(r"^status: (\w+)$", re.M)
LOGIN_RE = re.compile(r"\b(mot de passe|password|se connecter|login|nom d.utilisateur|username)\b", re.I)


def noise_score(body: str) -> dict:
    if not body.strip():
        return {"table_noise": 0.0, "link_only": True, "login_wall": False, "real_words": 0}

    # Pipes per word, not per line: some sites (aneti.tn's nested table layout) render an
    # entire page as 2-3 giant lines with no newlines between cells, so a per-line scan never
    # sees them. A normal prose document has ~0 "|" characters; a page full of empty markdown
    # table cells ("| |  |  | --- | --- |") has far more pipes than words.
    words = max(len(body.split()), 1)
    table_noise = body.count("|") / words

    no_links = LINK_RE.sub(r"\1", body)
    no_links = re.sub(r"[|\-:>#*]", " ", no_links)
    real_words = len(no_links.split())

    login_hits = len(LOGIN_RE.findall(body[:1000]))  # near the top = likely a gate, not a mention

    return {"table_noise": round(table_noise, 2), "link_only": real_words < 30,
            "login_wall": login_hits >= 2, "real_words": real_words}


def run_noise():
    flagged_by_site = {}
    total = 0
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        s = STATUS_RE.search(front)
        if not s or s.group(1) != "active":
            continue
        total += 1
        sc = noise_score(body)
        flags = [k for k in ("table_noise", "link_only", "login_wall")
                 if (sc[k] if isinstance(sc[k], bool) else sc[k] > 0.4)]
        if flags:
            site = p.relative_to(CORPUS).parts[0]
            flagged_by_site.setdefault(site, []).append(
                (p.relative_to(ROOT).as_posix(), flags, sc["real_words"], sc["table_noise"]))

    flagged_total = sum(len(v) for v in flagged_by_site.values())
    print(f"[noise] {flagged_total} / {total} active docs flagged for review\n")
    for site in sorted(flagged_by_site, key=lambda s: -len(flagged_by_site[s])):
        docs = flagged_by_site[site]
        print(f"{site}: {len(docs)} flagged")
        for path, flags, words, noise in sorted(docs, key=lambda d: -d[3])[:3]:
            print(f"    {','.join(flags):30s} words={words:4d} noise={noise:.2f}  {path}")
        if len(docs) > 3:
            print(f"    ... and {len(docs) - 3} more")


# ---- stubs: real-prose word count on every doc ----------------------------------------------

MIN_WORDS = 40
HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
TABLE_PIPE_RE = re.compile(r"\|")
FRONTMATTER_STRIP_RE = re.compile(r"^---\n.*?\n---\n", re.S)


def real_word_count(md: str) -> int:
    body = FRONTMATTER_STRIP_RE.sub("", md)
    body = COMMENT_RE.sub("", body)
    body = LINK_RE.sub(r"\1", body)          # keep link text, drop target
    body = HEADING_RE.sub("", body)
    body = TABLE_PIPE_RE.sub(" ", body)
    body = re.sub(r"[\-*>#]", " ", body)
    return len(body.split())


def run_stubs():
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
    print(f"[stubs] {flagged} / {total} documents flagged as likely stubs (< {MIN_WORDS} real words)\n")
    for site in sorted(by_site, key=lambda s: -len(by_site[s])):
        docs = by_site[site]
        print(f"{site}: {len(docs)} flagged")
        for wc, path in sorted(docs):
            print(f"    {wc:3d}w  {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["noise", "stubs", "all"], default="all")
    a = ap.parse_args()
    if a.mode in ("noise", "all"):
        run_noise()
    if a.mode == "all":
        print()
    if a.mode in ("stubs", "all"):
        run_stubs()


if __name__ == "__main__":
    main()
