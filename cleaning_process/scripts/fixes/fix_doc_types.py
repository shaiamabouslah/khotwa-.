"""Problem 6: unclassified `page` docs -> announcement | contact | procedure when the content says so.
Everything else stays `page`. (directory is set by fix_links.)"""
import re
from _common import run, count_words
from crawler.classify import _looks_like_procedure

ANNOUNCE_T = re.compile(r"^\W*(communiqu|avis|appel [àa]|بلاغ|إعلان|اعلان)", re.I)
ANNOUNCE_U = re.compile(r"/(actualit|news|annonce|communique|avis)", re.I)
CONTACT = re.compile(r"\b\d{2}[ .]\d{3}[ .]\d{3}\b|[\w.-]+@[\w-]+\.\w+|\b(?:T[ée]l|Fax)\b", re.I)


def fix(d):
    """doc types"""
    if d.fm.get("doc_type") != "page": return None
    t, u, b = d.fm.get("title", ""), d.fm.get("url", ""), d.body
    if ANNOUNCE_T.search(t) or ANNOUNCE_U.search(u): new = "announcement"
    elif count_words(b) < 200 and len(CONTACT.findall(b)) >= 2: new = "contact"
    elif _looks_like_procedure(b): new = "procedure"
    else: return None
    d.fm["doc_type"] = new
    return f"page -> {new}"


if __name__ == "__main__": run("doc_types", fix)
