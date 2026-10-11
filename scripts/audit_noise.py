"""Cheap, no-LLM triage pass: score every active document for markdown "noise" (empty table
pipes, link-only content, login/cookie-wall markers) so the heavier LLM cleaning pass
(scripts/llm_clean.py) only runs where it's actually needed instead of on the whole corpus.

Flags, independent of each other (a doc can have several):
  table_noise   many lines are pipe-heavy with near-zero real text (the ANETI pattern)
  link_only     stripped of links, almost no prose remains (the MES/transtu stub pattern)
  login_wall    body mentions a password/login field prominently, suggesting the real content
                is behind auth, not actually fetched
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
STATUS_RE = re.compile(r"^status: (\w+)$", re.M)
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
LOGIN_RE = re.compile(r"\b(mot de passe|password|se connecter|login|nom d.utilisateur|username)\b", re.I)


def score(body: str) -> dict:
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


def main():
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
        sc = score(body)
        flags = [k for k in ("table_noise", "link_only", "login_wall")
                 if (sc[k] if isinstance(sc[k], bool) else sc[k] > 0.4)]
        if flags:
            site = p.relative_to(CORPUS).parts[0]
            flagged_by_site.setdefault(site, []).append(
                (p.relative_to(ROOT).as_posix(), flags, sc["real_words"], sc["table_noise"]))

    flagged_total = sum(len(v) for v in flagged_by_site.values())
    print(f"{flagged_total} / {total} active docs flagged for review\n")
    for site in sorted(flagged_by_site, key=lambda s: -len(flagged_by_site[s])):
        docs = flagged_by_site[site]
        print(f"{site}: {len(docs)} flagged")
        for path, flags, words, noise in sorted(docs, key=lambda d: -d[3])[:3]:
            print(f"    {','.join(flags):30s} words={words:4d} noise={noise:.2f}  {path}")
        if len(docs) > 3:
            print(f"    ... and {len(docs) - 3} more")


if __name__ == "__main__":
    main()
