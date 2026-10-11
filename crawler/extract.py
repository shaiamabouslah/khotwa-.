"""HTML / PDF -> clean Markdown that keeps the page hierarchy (headings, lists, tables, links).

Site-specific behaviour comes from `sites.<name>.extract` in sources.yaml:
  content:        CSS selectors of the main content; the matching element with the most text wins
  drop:           extra CSS selectors to remove (menus, polls, news boxes...)
  accordion:      true for Elementor <details> accordions (services.gov.tn, khadamet.gov.tn)
  breadcrumb:     CSS selector of breadcrumb links
  update_pattern: regex with one group capturing a dd/mm/yyyy date shown by the page
"""
import io
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Comment
from bidi.algorithm import get_display
from markdownify import markdownify

from .http import decode_html

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def _fix_pdf_bidi(text: str) -> str:
    """pdfplumber extracts PDF glyphs in left-to-right visual order, which reverses
    right-to-left Arabic runs (and swaps their position relative to Latin/numeric runs
    on the same line). HTML pages don't have this problem (the browser/parser already
    gives logical-order Unicode); only PDF extraction needs this fix."""
    if not text or not _ARABIC_RE.search(text):
        return text
    return "\n".join(get_display(line, base_dir="L") for line in text.splitlines())

ALWAYS_DROP = ["script", "style", "noscript", "iframe", "svg", "form", "button", "nav",
               "header", "footer", "aside", "img", "picture", "video", "audio"]


@dataclass
class Extracted:
    markdown: str
    title: str
    breadcrumb: list[str] = field(default_factory=list)
    source_updated: str | None = None      # ISO date displayed by the page itself
    sections: list[str] = field(default_factory=list)  # headings found (for --list-sections)


def normalize(text: str) -> str:
    """Compare titles robustly: OOUN headings contain double and non-breaking spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ـً-ْ]", "", text)   # Arabic tatweel (معادلـــة) and diacritics
    # Alif variants (hamza-above إأ, madda آ, bare ا) are different base Unicode characters,
    # not combining marks, so NFKC above doesn't merge them -- confirmed as a real bug in
    # scripts/clean_to_kb.py: a pattern written "آجال" (madda) silently failed to match source
    # text spelled "الأجال" (hamza), even though both are "the same word" to a reader.
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[*_#>]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _pick_content(soup: BeautifulSoup, selectors: list[str]):
    candidates = [el for sel in selectors for el in soup.select(sel)]
    if not candidates:
        return soup.body or soup
    return max(candidates, key=lambda el: len(el.get_text(" ", strip=True)))


def _strip_empty_table_rows(root):
    """Old table-layout sites (aneti.tn and similar) nest a nav/menu link inside 2-3 levels of
    single-column tables used purely for pixel spacing -- every level contributes its own empty
    "| | | |" / "| --- | --- |" markdown row once converted, burying one real link under a wall
    of noise. Uses .select() (not find_all(recursive=False)) throughout: lxml silently inserts
    a <tbody> wrapper even for HTML that never had one, which would make a recursive=False scan
    see zero rows and skip a table entirely. Processes deepest tables first (sorted by DOM
    nesting depth) so a parent benefits from its children already being simplified."""
    def depth(el):
        return sum(1 for _ in el.parents)

    for table in sorted(root.select("table"), key=depth, reverse=True):
        for row in table.select("tr"):
            if row.find_parent("table") is not table:
                continue                       # belongs to a nested table, handled separately
            cells = [c for c in row.find_all(["td", "th"], recursive=False)]
            if cells and all(not c.get_text(strip=True) for c in cells):
                row.decompose()
            elif len(cells) > 1:
                # A kept row can still mix a couple of real cells into a sea of empty spacer
                # cells (e.g. aneti's search box: "ANETI | <3 real links> | 15 empty tds"),
                # which still renders as a huge markdown separator line. Drop just the empty
                # cells -- the row survives as a compact table of only its real cells.
                empty = [c for c in cells if not c.get_text(strip=True)]
                if empty and len(empty) < len(cells):
                    for c in empty:
                        c.decompose()
        real_cells = [c for c in table.select("td, th") if c.get_text(strip=True)]
        if not real_cells:
            table.decompose()
        elif len(real_cells) == 1 and table.parent is not None:
            table.replace_with(*real_cells[0].contents) if real_cells[0].contents else table.decompose()


def _accordions_to_headings(root, soup):
    """<details><summary>Title</summary> fields...</details>  ->  <h2>Title</h2> + <h3> labels."""
    for item in root.select("details"):
        summary = item.find("summary")
        title = _clean_text(summary.get_text(" ")) if summary else ""
        if summary:
            summary.decompose()
        for label in item.select("h4, h5, h6"):
            label.name = "h3"
        # Each field sits in its own <ul><li>: unwrap so labels are real headings.
        for wrapper in item.select("li.eael-feature-list-item, ul.eael-feature-list-items"):
            wrapper.unwrap()
        heading = soup.new_tag("h2")
        heading.string = title
        item.insert_before(heading)
        item.name = "div"
    # Tab headers duplicate the panel titles: drop them.
    for el in root.select(".eael-tabs-nav"):
        el.decompose()


def _absolute_links(root, base_url: str):
    for a in root.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "#", "mailto:", "tel:")):
            a.unwrap()
        else:
            a["href"] = urllib.parse.urljoin(base_url, href)


def _tidy_markdown(md: str) -> str:
    md = md.replace("\xa0", " ")
    lines = []
    for line in md.splitlines():
        line = line.rstrip()
        if re.fullmatch(r"#{1,6}\s*", line):      # empty headings
            continue
        if re.fullmatch(r"[-*+]\s*", line):        # empty list items
            continue
        if re.fullmatch(r"[\d\s/|❯❮<>»«.,:-]+", line):  # slider counters, pagination ("1 / 21 2 / 21")
            continue
        if line.startswith("#"):                   # plain-text headings: no bold, no links
            hashes, _, text = line.partition(" ")
            text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
            text = text.replace("**", "").replace("__", "")
            line = f"{hashes} {_clean_text(text)}"
        lines.append(line)
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def html_to_markdown(raw: bytes, url: str, cfg: dict) -> Extracted:
    soup = BeautifulSoup(decode_html(raw), "lxml")

    # Page-level metadata before anything is removed.
    full_text = soup.get_text(" ")
    source_updated = None
    if cfg.get("update_pattern"):
        m = re.search(cfg["update_pattern"], full_text)
        if m:
            d, mth, y = m.group(1).split("/")
            source_updated = f"{y}-{mth}-{d}"
    breadcrumb = []
    if cfg.get("breadcrumb"):
        breadcrumb = [_clean_text(a.get_text(" ")) for a in soup.select(cfg["breadcrumb"])]
        breadcrumb = [b for b in breadcrumb if b]
    page_title = _clean_text(soup.title.get_text(" ")) if soup.title else ""

    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    root = _pick_content(soup, cfg.get("content", []))
    for sel in ALWAYS_DROP + cfg.get("drop", []):
        for el in root.select(sel):
            el.decompose()
    if cfg.get("accordion"):
        _accordions_to_headings(root, soup)
    _strip_empty_table_rows(root)
    _absolute_links(root, url)

    md = markdownify(str(root), heading_style="ATX", bullets="-", strip=["span", "div"])
    md = _tidy_markdown(md)
    h1 = re.search(r"^# (.+)$", md, re.M)
    title = h1.group(1) if h1 else page_title
    sections = [m.group(2) for m in re.finditer(r"^(#{1,3}) (.+)$", md, re.M)]
    return Extracted(markdown=md, title=title, breadcrumb=breadcrumb,
                     source_updated=source_updated, sections=sections)


def pdf_to_markdown(raw: bytes) -> Extracted:
    import pdfplumber  # imported lazily: only needed for PDF sources

    parts, title = [], ""
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = _fix_pdf_bidi((page.extract_text() or "").strip())
            if not title and text:
                title = _clean_text(text.splitlines()[0])
            tables = page.extract_tables() or []
            body = [text] if text else []
            for table in tables:
                rows = [[_fix_pdf_bidi(_clean_text(c or "")) for c in row] for row in table if row]
                if len(rows) >= 2:
                    header = "| " + " | ".join(rows[0]) + " |"
                    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
                    body.append("\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in rows[1:]]))
            if body:
                parts.append(f"<!-- page {i} -->\n" + "\n\n".join(body))
    return Extracted(markdown=_tidy_markdown("\n\n".join(parts)), title=title)


@dataclass
class Part:
    section: str | None   # None = the whole page
    markdown: str


def split_document(markdown: str, mode: str = "none", level: int = 2, titles=()) -> list[Part]:
    """Split one page into documents.

    none     the whole page is one document
    headings one document per heading of `level` (used for multi-procedure pages)
    titles   one document per title listed in sources.yaml (pages without real headings)
    Text before the first split heading is kept as an 'intro' part only if substantial.
    """
    if mode == "titles" and titles:
        markdown = promote_titles(markdown, titles)
        level = 2
    elif mode != "headings":
        return [Part(None, markdown)]
    marker = re.compile(rf"^#{{{level}}} (.+)$")
    lines = markdown.splitlines()
    starts = [i for i, line in enumerate(lines) if marker.match(line)]
    if not starts:
        return [Part(None, markdown)]
    parts = []
    intro = "\n".join(lines[:starts[0]]).strip()
    if len(intro.split()) >= 40:
        parts.append(Part("intro", _tidy_markdown(intro)))
    for n, start in enumerate(starts):
        title = marker.match(lines[start]).group(1).strip()
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block = _tidy_markdown("\n".join(["# " + title] + lines[start + 1:end]))
        if len(block.split()) > 3:
            parts.append(Part(title, block))
    return parts


def promote_titles(markdown: str, titles) -> str:
    """Turn plain lines that exactly match a known section title into '## ' headings.

    Some pages (OOUN in Arabic) mark their sections with plain paragraphs, not <h*>.
    Only titles declared in sources.yaml are promoted, so nothing is guessed."""
    wanted = {normalize(t) for t in titles}
    out = []
    for line in markdown.splitlines():
        if not line.startswith(("#", "-", "|", " ")) and normalize(line) in wanted:
            line = "## " + line.strip().replace("**", "")
        out.append(line)
    return "\n".join(out)


def extract_section(markdown: str, section: str, sibling_titles=()) -> str | None:
    """Return the block under the heading that matches `section` (prefix match, then
    substring match), up to the next heading of the same or higher level.
    If no heading matches, known titles written as plain lines are promoted and we retry."""
    found = _extract_section(markdown, section)
    if found is None:
        found = _extract_section(promote_titles(markdown, {section, *sibling_titles}), section)
    return found


def _extract_section(markdown: str, section: str) -> str | None:
    lines = markdown.splitlines()
    headings = [(i, len(m.group(1)), normalize(m.group(2)))
                for i, line in enumerate(lines) if (m := re.match(r"^(#{1,6}) (.+)$", line))]
    target = normalize(section)
    match = next((h for h in headings if h[2].startswith(target)), None) \
        or next((h for h in headings if target in h[2]), None)
    if match is None:
        return None
    start, level, _ = match
    end = next((i for i, lvl, _ in headings if i > start and lvl <= level), len(lines))
    block = lines[start:end]
    block[0] = "#" + block[0][level:]           # the section title becomes the page title
    return _tidy_markdown("\n".join(block))
