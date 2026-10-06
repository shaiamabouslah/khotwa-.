"""Inspect a site before adding it to the registry.

For each base URL: is it reachable, does robots.txt allow us, does it block AI crawlers,
is there a sitemap, is the TLS certificate valid, which CMS, static HTML or JS app.
"""
import re
import urllib.parse
from dataclasses import asdict, dataclass

from .http import PoliteSession, decode_html
from .render import detect_render

AI_BOTS = ["GPTBot", "ClaudeBot", "anthropic-ai", "CCBot", "Google-Extended", "PerplexityBot"]


@dataclass
class ProbeResult:
    url: str
    status: str
    final_url: str = ""
    render: str = ""
    render_reason: str = ""
    robots_allows_us: bool | None = None
    blocks_ai_bots: list | None = None
    sitemap: str = ""
    tls_valid: bool | None = None
    cms: str = ""
    title: str = ""


def _cms(html: str) -> str:
    m = re.search(r'<meta[^>]+name="generator"[^>]+content="([^"]+)"', html, re.I)
    if m:
        return m.group(1)
    for marker, name in [("wp-content", "WordPress"), ("Liferay", "Liferay"), ("drupal", "Drupal"),
                         ("ng-version", "Angular"), ("__NEXT_DATA__", "Next.js"), ("joomla", "Joomla")]:
        if marker.lower() in html.lower():
            return name
    return ""


def probe(http: PoliteSession, base_url: str) -> ProbeResult:
    res = ProbeResult(url=base_url, status="")
    resp = http.fetch(base_url)
    if not resp.ok:
        res.status = resp.error or f"HTTP {resp.status}"
        return res
    res.status = "ok"
    html = decode_html(resp.content)
    res.final_url = resp.url
    res.render, res.render_reason = detect_render(resp.content, resp.content_type)
    res.tls_valid = not resp.insecure_tls
    res.cms = _cms(html)
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    res.title = re.sub(r"\s+", " ", m.group(1)).strip()[:80] if m else ""

    parts = urllib.parse.urlsplit(base_url)
    root = f"{parts.scheme}://{parts.netloc}"
    res.robots_allows_us = http.allowed(base_url)
    robots = http.fetch(root + "/robots.txt")
    body = decode_html(robots.content) if robots.ok else ""
    if "<html" in body.lower():
        body = ""
    blocked = []
    for bot in AI_BOTS:
        block = re.search(rf"(?ims)^user-agent:\s*{re.escape(bot)}\s*$(.*?)(?=^user-agent:|\Z)", body)
        if block and re.search(r"(?im)^disallow:\s*/\s*$", block.group(1)):
            blocked.append(bot)
    res.blocks_ai_bots = blocked
    sitemaps = re.findall(r"(?im)^sitemap:\s*(\S+)", body)
    if not sitemaps:
        for candidate in ("/sitemap_index.xml", "/sitemap.xml"):
            sm = http.fetch(root + candidate)
            head = sm.content[:5000] if sm.ok else b""
            if b"<urlset" in head or b"<sitemapindex" in head:
                sitemaps = [root + candidate]
                break
    res.sitemap = sitemaps[0] if sitemaps else ""
    return res


def as_row(r: ProbeResult) -> dict:
    return asdict(r)
