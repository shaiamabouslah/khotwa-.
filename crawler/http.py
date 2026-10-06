"""Polite HTTP client: robots.txt, per-host delay, retries, conditional GET, honest decoding."""
import codecs
import logging
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass

import requests
import urllib3

log = logging.getLogger(__name__)


def _utf8_then_cp1252(error):
    """Decode handler: bytes that are not valid UTF-8 are read as cp1252.

    cnss.tn mixes both encodings in the same page (UTF-8 menus, cp1252 articles),
    so neither a single UTF-8 nor a single cp1252 decode gives clean text.
    """
    chunk = error.object[error.start:error.end]
    return chunk.decode("cp1252", errors="replace"), error.end


codecs.register_error("utf8_then_cp1252", _utf8_then_cp1252)


def decode_html(raw: bytes) -> str:
    return raw.decode("utf-8", errors="utf8_then_cp1252")


def to_uri(url: str) -> str:
    """Percent-encode non-ASCII paths (khadamet.gov.tn uses Arabic slugs)."""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/%()+-_.~")
    return urllib.parse.urlunsplit(parts._replace(path=path))


@dataclass
class Response:
    url: str
    status: int | None          # None when the request failed entirely
    content: bytes = b""
    content_type: str = ""
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None
    insecure_tls: bool = False

    @property
    def not_modified(self) -> bool:
        return self.status == 304

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.content)


class PoliteSession:
    def __init__(self, user_agent: str, delay: float = 2.0, timeout: float = 30.0, retries: int = 3,
                 respect_robots: bool = True):
        self.user_agent = user_agent
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._insecure_hosts: set[str] = set()

    # -- politeness -----------------------------------------------------------------
    def _wait(self, host: str):
        elapsed = time.monotonic() - self._last_hit.get(host, 0.0)
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_hit[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            lines: list[str] = []
            try:
                resp = self._get(base + "/robots.txt", {})
                body = decode_html(resp.content) if resp.content else ""
                # Some servers answer /robots.txt with an HTML page: no rules then.
                if resp.status == 200 and "<html" not in body.lower():
                    lines = body.splitlines()
            except requests.RequestException:
                pass
            parser.parse(lines)
            self._robots[base] = parser
        return self._robots[base].can_fetch(self.user_agent, url)

    # -- fetching -------------------------------------------------------------------
    def _get(self, url: str, headers: dict) -> Response:
        host = urllib.parse.urlsplit(url).netloc
        self._wait(host)
        verify = host not in self._insecure_hosts
        try:
            r = self.session.get(url, headers=headers, timeout=self.timeout, verify=verify)
        except requests.exceptions.SSLError:
            # Certificate chain invalid: read the public page anyway, but report it.
            log.warning("invalid TLS certificate on %s, fetching without verification", host)
            self._insecure_hosts.add(host)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            r = self.session.get(url, headers=headers, timeout=self.timeout, verify=False)
        return Response(
            url=r.url, status=r.status_code, content=r.content,
            content_type=r.headers.get("Content-Type", ""),
            etag=r.headers.get("ETag"), last_modified=r.headers.get("Last-Modified"),
            insecure_tls=host in self._insecure_hosts,
        )

    def fetch(self, url: str, etag: str | None = None, last_modified: str | None = None) -> Response:
        """GET with retries. Pass etag/last_modified from the previous run to get 304s."""
        uri = to_uri(url)
        if not self.allowed(uri):
            return Response(url=url, status=None, error="disallowed by robots.txt")
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        error = "unknown error"
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._get(uri, headers)
                if resp.status in (200, 304) or 400 <= resp.status < 500:
                    return resp
                error = f"HTTP {resp.status}"
            except requests.RequestException as exc:
                # Includes truncated bodies (IncompleteRead) on very large pages.
                error = f"{exc.__class__.__name__}: {exc}"
            log.info("retry %d/%d for %s (%s)", attempt, self.retries, url, error)
            time.sleep(self.delay * attempt)
        return Response(url=url, status=None, error=error)
