"""Problem 3: scraped layout junk -> mostly-empty tables flattened to text, share/print buttons and
stray bullets dropped, social links moved to front matter, scrape-date footers dropped, whitespace runs and broken characters repaired."""
import re
from _common import run

SOCIAL = re.compile(r"\[[^\]]*\]\((https?://(?:www\.)?(?:facebook|twitter|x|linkedin|youtube|instagram)\.[^)]*)\)", re.I)
STAMP = re.compile(r"Derni[èe]re mise [àa] jour\s*:?\s*\d{1,2}/\d{1,2}/\d{4}(?:\s*[àa]\s*\d{1,2}:\d{2})?", re.I)
SEP = re.compile(r":?-{3,}:?")
_NAV = r"(?:imprimer|print|partager|envoyer|facebook|twitter|linkedin|newsletters?|haut de page|retour en haut|طباعة|مشاركة)"
NAV_LINE = re.compile(rf"^\s*(?:[-*+]\s*(?:{_NAV})?|{_NAV}|#+ web-infos et newsletters)\s*$", re.I)   # tiny share/print buttons, stray bullets


def flatten_tables(body):
    out, block = [], []
    def flush():
        if not block: return
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in block]
        cells = [c for r in rows for c in r if not SEP.fullmatch(c)]
        if sum(1 for c in cells if c) > 0.5 * len(cells):          # real data table: keep as is
            out.extend(block)
        else:                                                       # layout table: keep the text only
            for r in rows:
                txt = list(dict.fromkeys(c for c in r if c and not SEP.fullmatch(c)))
                if txt: out.append(" - ".join(txt))
        block.clear()
    for line in body.split("\n"):
        if line.lstrip().startswith("|"): block.append(line)
        else: flush(); out.append(line)
    flush()
    return "\n".join(out)


def clean(b):
    b = b.replace("\ufffd", "").replace("\u00a0", " ").replace("\u200b", "")
    b = SOCIAL.sub("", b)
    b = STAMP.sub("", b)
    # an indented "# Title" is invisible to every markdown-heading regex downstream (chunker,
    # validate_kb's H1 check) -- the literal "#" just leaks into the text as noise instead
    b = re.sub(r"^[ \t]+(#{1,6} )", r"\1", b, flags=re.M)
    b = "\n".join(l for l in b.split("\n") if not NAV_LINE.match(l))
    b = flatten_tables(b)
    b = re.sub(r"(?<=\S) {3,}", " ", b)                           # inner whitespace runs, keeps indentation
    b = re.sub(r"[ \t]+\n", "\n", b)
    return re.sub(r"\n{3,}", "\n\n", b)


def fix(d):
    """layout junk"""
    social = [m[1] for m in SOCIAL.finditer(d.body)]
    b = d.body
    for _ in range(5):                                             # steps interact (a removed line can expose a table): repeat until stable
        nxt = clean(b)
        if nxt == b: break
        b = nxt
    if b == d.body: return None
    if social: d.fm["social_links"] = list(dict.fromkeys(d.fm.get("social_links", []) + social))
    d.body = b
    return f"cleaned ({len(social)} social links kept in front matter)"


if __name__ == "__main__": run("layout_junk", fix)
