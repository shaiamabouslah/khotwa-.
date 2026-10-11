"""Decide how a page must be fetched (static HTML vs JavaScript app) and render JS pages.

Detection is automatic: a page is fetched over plain HTTP first; if its visible text is
tiny and it carries single-page-app markers, it needs a real browser. Playwright is an
optional dependency, only installed for sites that need it:

    pip install -r requirements-js.txt && playwright install chromium
"""
import logging
import re

from bs4 import BeautifulSoup

from .http import decode_html

log = logging.getLogger(__name__)

SPA_MARKERS = [
    r"<app-root", r'id="root"', r'id="app"', r"ng-version=", r"__NEXT_DATA__", r"data-reactroot",
    r"window\.__NUXT__", r"<noscript>[^<]*(?:javascript|JavaScript)",
]
MIN_STATIC_WORDS = 80


def visible_words(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    return len(soup.get_text(" ", strip=True).split())


def detect_render(raw: bytes, content_type: str = "") -> tuple[str, str]:
    """Return (mode, reason) with mode in {"pdf", "html", "js"}."""
    if "pdf" in content_type or raw[:5] == b"%PDF-":
        return "pdf", "PDF document"
    html = decode_html(raw)
    words = visible_words(html)
    markers = [m for m in SPA_MARKERS if re.search(m, html)]
    meta_refresh = re.search(r'http-equiv="?refresh"?[^>]*url=', html, re.I)
    if meta_refresh and words < MIN_STATIC_WORDS:
        return "html", "meta refresh redirect (follow it)"
    scripts = len(re.findall(r"<script[^>]+src=", html, re.I))
    if words < MIN_STATIC_WORDS and markers:
        return "js", f"{words} visible words + SPA markers {markers}"
    if words < 20 and scripts >= 3:
        return "js", f"only {words} visible words, {scripts} external scripts"
    return "html", f"{words} visible words in static HTML"


class BrowserRenderer:
    """Lazy Playwright wrapper: the browser starts only if a JS page is actually met."""

    def __init__(self, user_agent: str, timeout_ms: int = 30000):
        self.user_agent = user_agent
        self.timeout_ms = timeout_ms
        self._pw = self._browser = None

    def available(self) -> bool:
        try:
            import playwright  # noqa: F401
            return True
        except ImportError:
            return False

    def render(self, url: str) -> bytes:
        if self._browser is None:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch()
        page = self._browser.new_page(user_agent=self.user_agent)
        try:
            # "networkidle" never fires on pages with persistent background connections
            # (analytics pings, polling, websockets) -- common enough to make it the
            # wrong default. "domcontentloaded" + an explicit wait for the SPA to paint
            # is more robust.
            page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass   # fine if it never goes idle; we already waited for the DOM
            # Open collapsed content (accordions, "show more") so it is in the DOM.
            page.evaluate("""() => {
                document.querySelectorAll('details').forEach(d => d.open = true);
                document.querySelectorAll('[aria-expanded="false"]').forEach(b => { try { b.click() } catch (e) {} });
            }""")
            page.wait_for_timeout(1000)
            return page.content().encode("utf-8")
        finally:
            page.close()

    def close(self):
        if self._browser is not None:
            self._browser.close()
            self._pw.stop()
            self._browser = self._pw = None
