"""Problem 2: unreadable or broken pages: binary Word/PDF decoded as text, symbol soup, letter-spaced
"A d r e s s e", and server error pages (template errors crawled as content). Rewriting such text is lossy
(word boundaries are already gone), so the doc is KEPT untouched and flagged rag_exclude + reason:
nothing is deleted, the RAG simply skips it."""
import re
from _common import run

ERROR_PAGE = re.compile(r"error occurred while processing the template|Invocation of method|#set \(\$", re.I)
BINARY = re.compile(r"ÿÿ|bjbj|\x00")
SPACED = re.compile(r"(?<!\w)(?:[^\W\d_] ){3,}[^\W\d_](?!\w)")
NO_WORD = re.compile(r"[^\W\d_]{2}")                       # a real word has two letters in a row
MARKUP = re.compile(r"[\d|\-:.,%/()€$+–—_*#>\[\]]+")       # table pipes, numbers, bullets are not junk


def reason(body):
    text = re.sub(r"\]\([^)]*\)|https?://\S+", "", body)
    toks = text.split()
    if len(toks) < 5: return None
    junk = sum(not NO_WORD.search(t) and not MARKUP.fullmatch(t) for t in toks) / len(toks)
    if ERROR_PAGE.search(text[:1500]): return "server_error_page"
    if BINARY.search(text): return "binary_content"
    if junk > 0.3: return "symbol_soup"
    if sum(len(m[0]) for m in SPACED.finditer(text)) > 0.25 * len(text): return "letter_spaced"


def fix(d):
    """unreadable text"""
    if d.fm.get("rag_exclude"): return None
    r = reason(d.body)
    if not r: return None
    d.fm["rag_exclude"], d.fm["rag_exclude_reason"] = True, r
    return f"rag_exclude ({r})"


if __name__ == "__main__": run("unreadable", fix)
