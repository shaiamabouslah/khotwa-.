"""Find the pages of a site from its registry entry: sitemap(s) + seed pages + link rules.

  discover:
    sitemap: true | <url>      read sitemap(s) (robots.txt "Sitemap:" lines, /sitemap.xml, /sitemap_index.xml)
    seeds: [<url>, ...]        start pages for link following
    follow: [<regex>, ...]     links to traverse
    keep:   [<regex>, ...]     URLs that become documents (default: same as follow + seeds)
    exclude: [<regex>, ...]    never fetched
    drop_query: [<regex>, ...] URLs whose query string is noise (same page, different ?params)
    rewrite: [[<regex>, <replacement>], ...]  normalize duplicate URLs (e.g. /index.php/fr/1 -> /fr/1)
    max_depth: 2               link-following depth from seeds
    max_pages: 300             safety cap per site
Links matching `keep` but not `follow` (e.g. PDF forms) become documents without being traversed.
"""
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .http import PoliteSession, decode_html

log = logging.getLogger(__name__)

SKIP_EXT = re.compile(r"\.(jpe?g|png|gif|svg|webp|ico|css|js|zip|rar|mp4|mp3|docx?|xlsx?|pptx?)(\?|$)", re.I)


@dataclass
class Found:
    url: str
    via: str                      # "sitemap" | "seed" | "link"
    lastmod: str | None = None    # from the sitemap, used to skip unchanged pages


def canonical(url: str) -> str:
    """Drop fragments, session ids and trailing noise so one page = one URL."""
    url = re.sub(r";jsessionid=[^?#]*", "", url, flags=re.I)
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.urlencode(
        [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if not k.lower().startswith(("utm_", "fbclid"))])
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", query, ""))


def _bare_host(url: str) -> str:
    """www.ucar.rnu.tn and ucar.rnu.tn are the same site."""
    host = urllib.parse.urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _matches(url: str, patterns) -> bool:
    decoded = urllib.parse.unquote(url)
    return any(re.search(p, decoded) for p in patterns or [])


class Discoverer:
    def __init__(self, http: PoliteSession, site: dict):
        self.http = http
        self.site = site
        self.cfg = site.get("discover", {})
        self.host = _bare_host(site["base_url"])

    def clean(self, url: str) -> str:
        url = canonical(url)
        for pattern, repl in self.cfg.get("rewrite") or []:
            url = re.sub(pattern, repl, url)
        if _matches(url, self.cfg.get("drop_query")):
            url = url.split("?", 1)[0]
        return url

    def allowed_url(self, url: str) -> bool:
        return (_bare_host(url) == self.host
                and not SKIP_EXT.search(url)
                and not _matches(url, self.cfg.get("exclude")))

    def kept(self, url: str) -> bool:
        keep = self.cfg.get("keep") or self.cfg.get("follow") or [".*"]
        return _matches(url, keep)

    # -- sitemaps --------------------------------------------------------------------
    def sitemap_urls(self) -> list[str]:
        sm = self.cfg.get("sitemap")
        if not sm:
            return []
        if isinstance(sm, str):
            return [sm]
        base = self.site["base_url"].rstrip("/")
        found = []
        robots = self.http.fetch(base + "/robots.txt")
        if robots.ok:
            found += re.findall(r"(?im)^sitemap:\s*(\S+)", decode_html(robots.content))
        return found or [base + "/sitemap_index.xml", base + "/sitemap.xml"]

    def read_sitemap(self, url: str, depth: int = 0) -> list[Found]:
        resp = self.http.fetch(url)
        if not resp.ok or depth > 2:
            return []
        try:
            root = ET.fromstring(resp.content.lstrip())   # some sitemaps start with blank lines
        except ET.ParseError:
            return []
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        out: list[Found] = []
        for sm in root.findall("s:sitemap", ns):          # sitemap index
            loc = sm.findtext("s:loc", default="", namespaces=ns).strip()
            if loc:
                out += self.read_sitemap(loc, depth + 1)
        for u in root.findall("s:url", ns):
            loc = u.findtext("s:loc", default="", namespaces=ns).strip()
            if loc:
                out.append(Found(canonical(loc), "sitemap", u.findtext("s:lastmod", namespaces=ns)))
        return out

    # -- link following --------------------------------------------------------------
    def links(self, url: str, html: bytes) -> list[str]:
        soup = BeautifulSoup(decode_html(html), "lxml")
        out = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            out.append(self.clean(urllib.parse.urljoin(url, href)))
        return out

    def run(self, fetched_pages: dict) -> list[Found]:
        """`fetched_pages` (url -> Response) is shared with the crawl so no page is fetched twice."""
        results: dict[str, Found] = {}
        max_pages = self.cfg.get("max_pages", 300)

        for sm in self.sitemap_urls():
            for f in self.read_sitemap(sm):
                f.url = self.clean(f.url)
                if self.allowed_url(f.url) and self.kept(f.url):
                    results.setdefault(f.url, f)

        follow = self.cfg.get("follow", [])
        queue = deque((self.clean(s), 0) for s in self.cfg.get("seeds", []))
        seen = set()
        while queue and len(results) < max_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if self.kept(url):   # seeds that are only navigation hubs are traversed, not kept
                results.setdefault(url, Found(url, "seed" if depth == 0 else "link"))
            if depth >= self.cfg.get("max_depth", 2) or not (follow or self.cfg.get("keep")):
                continue
            if url not in fetched_pages:
                resp = self.http.fetch(url)
                if not resp.ok:
                    continue
                fetched_pages[url] = resp
            if fetched_pages[url].content[:5] == b"%PDF-":
                continue
            for link in self.links(url, fetched_pages[url].content):
                if link in seen or not self.allowed_url(link):
                    continue
                if _matches(link, follow):
                    queue.append((link, depth + 1))
                elif self.kept(link) and len(results) < max_pages:
                    results.setdefault(link, Found(link, "link"))   # leaf: kept, not traversed

        if len(results) >= max_pages:
            log.warning("%s: max_pages=%d reached, discovery truncated", self.host, max_pages)
        return list(results.values())[:max_pages]
