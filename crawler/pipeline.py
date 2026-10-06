"""One run = discover + crawl + sync, site by site (sites run in parallel, each politely).

For each active site of sources.yaml:
  1. discover   sitemap(s) + seed pages + link rules  ->  list of URLs
  2. fetch      plain HTTP; conditional GET; skipped when the sitemap lastmod is unchanged;
                JavaScript apps detected automatically and rendered with a headless browser
  3. extract    HTML/PDF -> Markdown keeping the hierarchy
  4. split      one page -> one or several documents (headings, accordions, known titles)
  5. classify   language, document type, domain/subtopic hints
  6. sync       content hash -> new / changed / unchanged; failures keep the last good
                version (stale); pages that disappear from the site are marked removed
Output: the unified corpus data/corpus/<site>/<url path>/<doc>.<lang>.md + catalog.jsonl
"""
import json
import logging
import re
import threading
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from .classify import Classifier, detect_lang, doc_type
from .discover import Discoverer, Found, canonical
from .extract import html_to_markdown, normalize, pdf_to_markdown, split_document
from .http import PoliteSession, decode_html
from .registry import lang_versions, load_registry
from .render import BrowserRenderer, detect_render
from .state import SyncState, content_hash, now_iso

log = logging.getLogger(__name__)


# ── naming ───────────────────────────────────────────────────────────────────────────
def slugify(text: str, max_len: int = 60) -> str:
    text = normalize(text)
    text = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE).strip("-")
    return text[:max_len].rstrip("-") or "page"


def url_segments(url: str) -> list[str]:
    """URL path (+ query) as safe folder names: the corpus mirrors each site's structure."""
    parts = urllib.parse.urlsplit(url)
    segs = [s for s in urllib.parse.unquote(parts.path).split("/") if s]
    segs = [re.sub(r'[<>:"\\|?*\s]+', "_", s)[:80] for s in segs]
    if parts.query:
        segs.append("q_" + re.sub(r"[^\w=-]+", "_", parts.query)[:60])
    return segs or ["home"]


def doc_identity(site_key: str, base_url: str, section: str | None) -> tuple[str, list[str]]:
    """Path segments for the corpus file. A whole-page document (no section split) uses
    the URL's own last segment as the filename directly — no extra 'index' layer, no extra
    folder nesting. A split page (section set) genuinely needs the folder/filename split,
    since several section documents share the same URL folder."""
    segs = url_segments(base_url)
    if section:
        leaf = slugify(section)
        path_segs = segs + [leaf]
    else:
        leaf = "index"
        path_segs = segs
    return f"{site_key}:/{'/'.join(segs)}#{leaf}", path_segs


# ── run ──────────────────────────────────────────────────────────────────────────────
class Crawler:
    def __init__(self, root: Path, registry_path: Path, force: bool = False, dry_run: bool = False,
                 workers: int = 4):
        self.root = root
        self.reg = load_registry(registry_path)
        self.taxonomy = yaml.safe_load((root / "kb" / "taxonomy.yaml").read_text(encoding="utf-8"))
        self.classifier = Classifier(self.taxonomy)
        self.raw_dir = root / "data" / "raw"
        self.corpus_dir = root / "data" / "corpus"
        self.state = SyncState(root / "data" / "state")
        self.force = force
        self.dry_run = dry_run
        self.workers = workers
        self.defaults = self.reg["defaults"]
        self.lock = threading.Lock()
        self.page_overrides = {canonical(p["url"]): p for p in self.reg.get("pages", [])}

    def new_session(self) -> PoliteSession:
        d = self.defaults
        return PoliteSession(d["user_agent"], d["delay_seconds"], d["timeout_seconds"],
                             respect_robots=d.get("respect_robots_txt", True))

    # 2. fetch ------------------------------------------------------------------------
    def raw_path(self, site_key: str, url: str) -> Path:
        return self.raw_dir.joinpath(site_key, *url_segments(url), "content.bin")

    def fetch(self, ctx: dict, url: str, lastmod: str | None) -> tuple[bytes, str, bool] | None:
        """Return (raw, render_mode, fresh). fresh=False: the source failed, last snapshot reused."""
        site_key, site, http = ctx["key"], ctx["site"], ctx["http"]
        path = self.raw_path(site_key, url)
        prev = self.state.urls.get(url, {})
        snapshot = path.exists()

        if snapshot and not self.force and lastmod and prev.get("sitemap_lastmod") == lastmod:
            return path.read_bytes(), prev.get("render", "html"), True        # sitemap says unchanged

        resp = ctx["cache"].pop(url, None) or http.fetch(
            url, etag=None if self.force or not snapshot else prev.get("etag"),
            last_modified=None if self.force or not snapshot else prev.get("last_modified"))
        if resp.not_modified and snapshot:
            with self.lock:
                prev.update(fetched_at=now_iso(), http_status=304, last_error=None)
            return path.read_bytes(), prev.get("render", "html"), True
        if not resp.ok:
            reason = resp.error or f"HTTP {resp.status}"
            with self.lock:
                self.state.urls.setdefault(url, {})["last_error"] = f"{now_iso()} {reason}"
            if snapshot:
                log.warning("FAIL %s (%s): last snapshot reused", url, reason)
                return path.read_bytes(), prev.get("render", "html"), False
            log.error("FAIL %s (%s): no snapshot", url, reason)
            return None

        raw = resp.content
        mode, reason = detect_render(raw, resp.content_type)
        forced = site.get("render", "auto")
        if forced != "auto" and mode != "pdf":
            mode = forced
        # Small redirect pages (<meta http-equiv="refresh">) point to the real page.
        refresh = re.search(rb'http-equiv="?refresh"?[^>]*url=[\'"]?([^\'">]+)', raw[:3000], re.I)
        if refresh and len(raw) < 3000:
            target = urllib.parse.urljoin(resp.url, refresh.group(1).decode("latin-1").strip())
            log.info("meta refresh %s -> %s", url, target)
            return self.fetch(ctx, target, None)
        if mode == "js":
            renderer = ctx.setdefault("renderer", BrowserRenderer(self.defaults["user_agent"]))
            if not renderer.available():
                log.error("JS page %s (%s): install the browser renderer: "
                          "pip install -r requirements-js.txt && playwright install chromium", url, reason)
                return None
            raw = renderer.render(url)
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        with self.lock:
            self.state.urls[url] = {
                "raw_path": path.relative_to(self.root).as_posix(), "render": mode,
                "etag": resp.etag, "last_modified": resp.last_modified, "sitemap_lastmod": lastmod,
                "fetched_at": now_iso(), "http_status": resp.status,
                "insecure_tls": resp.insecure_tls, "last_error": None,
            }
        return raw, mode, True

    # 3-5. extract, split, classify ------------------------------------------------------
    def documents(self, ctx: dict, url: str, raw: bytes, mode: str) -> list[dict]:
        site = ctx["site"]
        override = self.page_overrides.get(canonical(url), {})
        if mode == "pdf":
            ex = pdf_to_markdown(raw)
        else:
            cfg = dict(site.get("extract", {}))
            split = override.get("split", site.get("split", "none"))
            if split == "accordion":
                cfg["accordion"] = True
            ex = html_to_markdown(raw, url, cfg)
        split = override.get("split", site.get("split", "none"))
        if mode == "pdf" or split == "none":
            parts = split_document(ex.markdown, "none")
        elif split == "titles":
            parts = split_document(ex.markdown, "titles", titles=override.get("titles", []))
        else:   # headings / accordion
            parts = split_document(ex.markdown, "headings", level=override.get("level", site.get("split_level", 2)))

        docs = []
        min_words = site.get("min_words", self.defaults.get("min_words", 30))
        for part in parts:
            # Count only visible text: link *targets* (sharer.php?u=...&text=Title+Encoded)
            # otherwise get counted as prose words, letting link-only stub pages (a title +
            # a couple of social-share buttons) slip past the min_words filter.
            visible = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", part.markdown)
            words = len(re.findall(r"\w*[^\W\d_]\w*", visible))   # words with letters only
            if words < min_words:
                continue
            title_m = re.match(r"^# (.+)$", part.markdown, re.M)
            title = override.get("title") or (title_m.group(1) if title_m else ex.title) or url
            hint = self.classifier.classify(title, [ex.title, *ex.breadcrumb], part.markdown, site.get("domain"))
            links = sorted(set(re.findall(r"\]\((https?://[^)\s]+(?:\.pdf|get_file[^)\s]*))\)", part.markdown, re.I)))
            docs.append({
                "section": part.section, "markdown": part.markdown, "title": title,
                "breadcrumb": ex.breadcrumb + ([part.section] if part.section and part.section != "intro" else []),
                "source_updated": ex.source_updated, "words": words, "forms": links,
                "doc_type": override.get("doc_type") or doc_type(url, title, mode == "pdf", site.get("doc_types")),
                "domain_hint": override.get("domain") or hint.domain,
                "subtopic_hint": override.get("subtopic") or hint.subtopic,
                "hint_score": hint.score, "hint_keywords": hint.matched[:6],
            })
        return docs

    # 6. sync -------------------------------------------------------------------------
    def sync_doc(self, ctx: dict, base_url: str, url: str, lang: str | None, doc: dict,
                 mode: str, fresh: bool, found: Found) -> str:
        site_key, site = ctx["key"], ctx["site"]
        lang = lang or detect_lang(doc["markdown"])
        doc_id, segs = doc_identity(site_key, base_url, doc["section"])
        new_hash = content_hash(doc["markdown"])
        # The same title can appear twice on a page (services.gov.tn repeats services across
        # tabs): an identical copy is skipped, a different text gets a numbered id.
        n = 1
        while f"{doc_id}:{lang}" in ctx["run_hashes"]:
            if ctx["run_hashes"][f"{doc_id}:{lang}"] == new_hash:
                return "duplicate"
            n += 1
            doc_id, segs = doc_identity(site_key, base_url, f"{doc['section']} {n}")
        key = f"{doc_id}:{lang}"
        ctx["run_hashes"][key] = new_hash
        out = self.corpus_dir.joinpath(site_key, *segs[:-1], f"{segs[-1]}.{lang}.md")
        with self.lock:
            prev = dict(self.state.pages.get(key, {}))
        ctx["seen"].add(key)

        if prev.get("hash") == new_hash and out.exists():
            status = "unchanged" if fresh else "stale"
            if fresh and prev.get("status") in ("stale", "removed", "error"):
                self._set_status(out, "active")
                self.state.log_change(key, "recovered", url=url)
            elif not fresh and prev.get("status") != "stale":
                self._set_status(out, "stale")
                self.state.log_change(key, "stale", reason="source unreachable, last snapshot reused")
            meta = prev.get("meta", {})
        else:
            status = "changed" if prev.get("hash") else "new"
            u = self.state.urls.get(url, {})
            meta = {
                "doc_id": doc_id, "site": site_key, "org": site["org"], "trust": site["trust"],
                "url": url, "lang": lang, "title": doc["title"], "section": doc["section"],
                "breadcrumb": doc["breadcrumb"], "doc_type": doc["doc_type"],
                "domain_hint": doc["domain_hint"], "subtopic_hint": doc["subtopic_hint"],
                "hint_score": doc["hint_score"], "hint_keywords": doc["hint_keywords"],
                "forms": doc["forms"], "render": mode, "discovered_via": found.via,
                "source_updated": doc["source_updated"], "http_last_modified": u.get("last_modified"),
                "sitemap_lastmod": found.lastmod, "fetched_at": u.get("fetched_at"),
                "first_seen": prev.get("meta", {}).get("first_seen") or now_iso(),
                "last_changed": now_iso(), "content_hash": new_hash, "words": doc["words"],
                "status": "active" if fresh else "stale",
            }
            if not self.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()
                out.write_text(f"---\n{front}\n---\n\n{doc['markdown']}", encoding="utf-8")
            self.state.log_change(key, status, url=url, hash=new_hash, previous_hash=prev.get("hash"))
            if not fresh:
                status = "stale"
        with self.lock:
            self.state.pages[key] = {"site": site_key, "path": out.relative_to(self.root).as_posix(),
                                     "url": url, "hash": new_hash, "status": status,
                                     "last_checked": now_iso(), "meta": meta}
        return status

    @staticmethod
    def _set_status(path: Path, status: str):
        if path.exists():
            text = path.read_text(encoding="utf-8")
            path.write_text(re.sub(r"^status: \w+$", f"status: {status}", text, count=1, flags=re.M),
                            encoding="utf-8")

    # one site ------------------------------------------------------------------------
    def run_site(self, site_key: str) -> Counter:
        site = self.reg["sites"][site_key]
        ctx = {"key": site_key, "site": site, "http": self.new_session(), "seen": set(), "cache": {}, "run_hashes": {}}
        counts: Counter = Counter()
        try:
            found = Discoverer(ctx["http"], site).run(ctx["cache"])
            for page in self.reg.get("pages", []):          # pinned pages are always included
                if page["site"] == site_key and canonical(page["url"]) not in {f.url for f in found}:
                    found.append(Found(canonical(page["url"]), "pinned"))
            log.info("%s: %d URLs discovered", site_key, len(found))
            if not found:
                log.error("%s: nothing discovered (site down?) - existing documents kept as stale", site_key)
                return self._mark_site(site_key, "stale", counts)

            by_base: dict[str, Found] = {}
            for f in found:   # /web/x, /fr/web/x and /ar/web/x are the same page
                by_base.setdefault(lang_versions(site, f.url)[0], f)
            for base, f in by_base.items():
                variants = lang_versions(site, base)[1]
                targets = variants.items() if variants else [(None, f.url)]
                for lang, url in targets:
                    try:
                        got = self.fetch(ctx, url, f.lastmod)
                        if got is None:
                            counts["error"] += 1
                            continue
                        raw, mode, fresh = got
                        for doc in self.documents(ctx, url, raw, mode):
                            counts[self.sync_doc(ctx, base, url, lang, doc, mode, fresh, f)] += 1
                    except Exception:   # one broken page never stops the site
                        log.exception("crash on %s", url)
                        counts["error"] += 1
            self._mark_site(site_key, "removed", counts, only_unseen=ctx["seen"])
        finally:
            if "renderer" in ctx:
                ctx["renderer"].close()
        log.info("%s done: %s", site_key, dict(counts))
        return counts

    def _mark_site(self, site_key: str, status: str, counts: Counter, only_unseen: set | None = None) -> Counter:
        """status='stale': whole site down. status='removed': documents not found this run;
        only declared removed after 2 consecutive misses (a seed page may just be down)."""
        with self.lock:
            keys = [k for k, p in self.state.pages.items() if p.get("site") == site_key
                    and (only_unseen is None or k not in only_unseen) and p.get("status") != status]
            if only_unseen is not None:
                for k, p in self.state.pages.items():
                    if p.get("site") == site_key and k in only_unseen:
                        p["missed_runs"] = 0
        for key in keys:
            page = self.state.pages[key]
            if status == "removed":
                page["missed_runs"] = page.get("missed_runs", 0) + 1
                if page["missed_runs"] < 2:
                    counts["missing"] += 1
                    continue
            self._set_status(self.root / page["path"], status)
            page["status"] = status
            self.state.log_change(key, status, url=page.get("url"))
            counts[status] += 1
        return counts

    # all sites -------------------------------------------------------------------------
    def run(self, sites: list[str] | None = None) -> Counter:
        active = [k for k, s in self.reg["sites"].items() if s["status"] == "active"]
        selected = [s for s in (sites or active) if s in self.reg["sites"]]
        total: Counter = Counter()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for counts in pool.map(self.run_site, selected):
                total.update(counts)
        if not self.dry_run:
            self.state.save()
            self.write_catalog()
            (self.state.dir / "last_run.json").write_text(json.dumps(
                {"finished_at": now_iso(), "sites": selected, "counts": dict(total)}, indent=2),
                encoding="utf-8")
        return total

    def write_catalog(self):
        """One JSON line per document: the single entry point for the cleaning stage."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        with (self.corpus_dir / "catalog.jsonl").open("w", encoding="utf-8") as f:
            for key, page in sorted(self.state.pages.items()):
                row = {**page.get("meta", {}), "path": page["path"], "status": page["status"],
                       "last_checked": page.get("last_checked")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
