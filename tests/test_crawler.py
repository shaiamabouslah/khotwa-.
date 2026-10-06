"""Offline tests for the crawler (no network): python -m unittest discover -s tests -t ."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from crawler.classify import Classifier, detect_lang, doc_type
from crawler.discover import Discoverer
from crawler.extract import extract_section, html_to_markdown, normalize, split_document
from crawler.http import Response, decode_html
from crawler.pipeline import Crawler
from crawler.registry import lang_versions, validate_registry
from crawler.render import detect_render

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY = yaml.safe_load((ROOT / "kb" / "taxonomy.yaml").read_text(encoding="utf-8"))


class FakeHTTP:
    """Stands in for PoliteSession: serves whatever the test puts in `pages`."""
    def __init__(self, pages=None):
        self.pages: dict[str, bytes | None] = pages or {}

    def fetch(self, url, etag=None, last_modified=None):
        body = self.pages.get(url)
        if body is None:
            return Response(url=url, status=None, error="ConnectionError")
        ctype = "application/pdf" if body[:5] == b"%PDF-" else "text/html"
        return Response(url=url, status=200, content=body, content_type=ctype)


# ── extraction ───────────────────────────────────────────────────────────────────────
class ExtractTests(unittest.TestCase):
    def test_mixed_encoding_page(self):
        raw = "<p>Salarié</p>".encode("utf-8") + "<p>Pièces à fournir</p>".encode("cp1252")
        self.assertEqual(decode_html(raw), "<p>Salarié</p><p>Pièces à fournir</p>")

    def test_hierarchy_links_and_noise(self):
        html = b"""<html><body><nav>Menu</nav><div class="main">
            <h2>Pi\xc3\xa8ces</h2><ul><li><a href="/f.pdf">Formulaire</a></li></ul>
            <script>x()</script></div><footer>Footer</footer></body></html>"""
        ex = html_to_markdown(html, "https://site.tn/page", {"content": ["div.main"]})
        self.assertIn("## Pièces", ex.markdown)
        self.assertIn("[Formulaire](https://site.tn/f.pdf)", ex.markdown)
        self.assertNotIn("Menu", ex.markdown)
        self.assertNotIn("Footer", ex.markdown)

    def test_accordion_split_one_document_per_service(self):
        html = """<div data-elementor-type="wp-page">
          <details class="e-n-accordion-item"><summary><div>Service A</div></summary>
            <ul class="eael-feature-list-items"><li class="eael-feature-list-item">
            <h6>Documents demandés</h6><p>Aucun</p></li></ul></details>
          <details class="e-n-accordion-item"><summary><div>Service B</div></summary><p>B text here</p></details>
        </div>""".encode()
        ex = html_to_markdown(html, "https://x.tn/", {"content": ['div[data-elementor-type="wp-page"]'],
                                                      "accordion": True})
        parts = split_document(ex.markdown, "headings", level=2)
        self.assertEqual([p.section for p in parts], ["Service A", "Service B"])
        self.assertIn("### Documents demandés", parts[0].markdown)
        self.assertNotIn("Service B", parts[0].markdown)

    def test_section_stops_at_next_sibling(self):
        md = "## A\ntext a\n### A.1\nsub\n## B\ntext b\n"
        self.assertEqual(extract_section(md, "A"), "# A\ntext a\n### A.1\nsub\n")

    def test_split_by_known_titles(self):
        md = "intro\n\nإن كنت طالبًا جديدًا\n\nشرط 1 للحصول\n\nمنح الدكتوراه\n\nشرط 2 للحصول\n"
        parts = split_document(md, "titles", titles=["إن كنت طالبًا جديدًا", "منح الدكتوراه"])
        self.assertEqual(len(parts), 2)
        self.assertIn("شرط 1", parts[0].markdown)
        self.assertNotIn("شرط 2", parts[0].markdown)

    def test_normalize_ignores_spacing_and_tatweel(self):
        self.assertEqual(normalize("Vous  êtes\xa0nouveau"), normalize("vous êtes nouveau"))
        self.assertEqual(normalize("معادلـــة"), normalize("معادلة"))


# ── detection & classification ───────────────────────────────────────────────────────
class DetectTests(unittest.TestCase):
    def test_render_detection(self):
        spa = b'<html><body><app-root></app-root><script src="main.js"></script></body></html>'
        static = ("<html><body><main>" + "mot " * 200 + "</main></body></html>").encode()
        self.assertEqual(detect_render(spa)[0], "js")
        self.assertEqual(detect_render(static)[0], "html")
        self.assertEqual(detect_render(b"%PDF-1.7 ...")[0], "pdf")

    def test_language(self):
        self.assertEqual(detect_lang("الوثائق المطلوبة للحصول على المنحة"), "ar")
        self.assertEqual(detect_lang("Les pièces à fournir pour la demande de bourse"), "fr")
        self.assertEqual(detect_lang("The step by step process to register your company"), "en")

    def test_classifier(self):
        clf = Classifier(TAXONOMY)
        h = clf.classify("Affiliation des employés de maison", [], "Les gens de maison doivent être déclarés")
        self.assertEqual((h.domain, h.subtopic), ("employment", "employes-de-maison"))
        h = clf.classify("طلب منحة جامعية", [], "الوثائق المطلوبة لمطلب المنح")
        self.assertEqual((h.domain, h.subtopic), ("studies", "bourses"))
        self.assertIsNone(clf.classify("Paiement des factures d'eau", [], "SONEDE").subtopic)

    def test_doc_type(self):
        self.assertEqual(doc_type("https://x.tn/a", "بلاغ حول المنح", False, []), "announcement")
        self.assertEqual(doc_type("https://x.tn/f.pdf", "Demande d'affiliation", True, []), "form")
        self.assertEqual(doc_type("https://x.tn/a", "x", False, [{"match": "/a$", "type": "procedure"}]),
                         "procedure")


# ── registry & discovery ─────────────────────────────────────────────────────────────
class RegistryTests(unittest.TestCase):
    def test_real_registry_is_valid(self):
        reg = yaml.safe_load((ROOT / "sources.yaml").read_text(encoding="utf-8"))
        self.assertEqual(validate_registry(reg, TAXONOMY), [])

    def test_lang_variants(self):
        site = {"lang_variants": {
            "base": [r"^(https://www\.cnss\.tn)/(?:fr/|ar/)?web/", r"\1/web/"],
            "fr": [r"^(https://www\.cnss\.tn)/web/", r"\1/fr/web/"],
            "ar": [r"^(https://www\.cnss\.tn)/web/", r"\1/ar/web/"]}}
        base, v = lang_versions(site, "https://www.cnss.tn/fr/web/x")
        self.assertEqual(base, "https://www.cnss.tn/web/x")
        self.assertEqual(v["ar"], "https://www.cnss.tn/ar/web/x")

    def test_discovery_follow_keep_exclude(self):
        hub = b"""<a href="/proc/1">p1</a><a href="/proc/2;jsessionid=AB">p2</a>
                  <a href="/forms/f.pdf">form</a><a href="/news/9">news</a><a href="https://other.tn/x">ext</a>"""
        site = {"base_url": "https://s.tn", "discover": {
            "seeds": ["https://s.tn/hub"], "follow": ["/hub"], "keep": ["/proc/", r"\.pdf$"],
            "exclude": ["/news/"]}}
        http = FakeHTTP({"https://s.tn/hub": hub})
        urls = sorted(f.url for f in Discoverer(http, site).run({}))
        self.assertEqual(urls, ["https://s.tn/forms/f.pdf", "https://s.tn/proc/1", "https://s.tn/proc/2"])


# ── full sync lifecycle ──────────────────────────────────────────────────────────────
class SyncTests(unittest.TestCase):
    URL = "https://site.tn/p"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "kb").mkdir()
        shutil.copy(ROOT / "kb" / "taxonomy.yaml", self.root / "kb" / "taxonomy.yaml")
        self.write_registry([self.URL])
        self.http = FakeHTTP()

    def tearDown(self):
        shutil.rmtree(self.root)

    def write_registry(self, seeds):
        reg = {"defaults": {"user_agent": "test", "delay_seconds": 0, "timeout_seconds": 1, "min_words": 3},
               "sites": {"s": {"status": "active", "trust": "official", "org": "Org", "domain": "studies",
                               "base_url": "https://site.tn", "discover": {"seeds": seeds},
                               "extract": {"content": ["main"]}}}}
        (self.root / "sources.yaml").write_text(yaml.safe_dump(reg), encoding="utf-8")

    def run_once(self, body):
        crawler = Crawler(self.root, self.root / "sources.yaml")
        crawler.new_session = lambda: self.http
        self.http.pages[self.URL] = None if body is None else \
            f"<main><h1>Bourse</h1><p>{body} demande de bourse universitaire</p></main>".encode()
        return crawler.run()

    def doc(self) -> Path:
        files = list((self.root / "data" / "corpus").rglob("*.md"))
        self.assertEqual(len(files), 1)
        return files[0]

    def events(self):
        lines = (self.root / "data/state/changelog.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["event"] for line in lines]

    def test_full_lifecycle(self):
        self.assertEqual(self.run_once("v1")["new"], 1)
        meta = yaml.safe_load(self.doc().read_text(encoding="utf-8").split("---")[1])
        self.assertEqual((meta["lang"], meta["domain_hint"], meta["subtopic_hint"]), ("fr", "studies", "bourses"))
        self.assertEqual(meta["status"], "active")
        first_write = self.doc().stat().st_mtime_ns

        self.assertEqual(self.run_once("v1")["unchanged"], 1)
        self.assertEqual(self.doc().stat().st_mtime_ns, first_write)    # untouched: clean git diff

        self.assertEqual(self.run_once("v2")["changed"], 1)
        self.assertEqual(self.run_once(None)["stale"], 1)               # source down
        text = self.doc().read_text(encoding="utf-8")
        self.assertIn("v2", text)                                       # last good version kept
        self.assertIn("status: stale", text)

        self.assertEqual(self.run_once("v2")["unchanged"], 1)           # source back
        self.assertIn("status: active", self.doc().read_text(encoding="utf-8"))
        self.assertEqual(self.events(), ["new", "changed", "stale", "recovered"])

        catalog = (self.root / "data/corpus/catalog.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(catalog), 1)

    def test_removed_only_after_two_missed_runs(self):
        self.run_once("v1")
        self.write_registry(["https://site.tn/other"])                  # page no longer discovered
        self.assertEqual(self.run_once("v1")["missing"], 1)
        self.assertIn("status: active", self.doc().read_text(encoding="utf-8"))
        self.assertEqual(self.run_once("v1")["removed"], 1)
        self.assertIn("status: removed", self.doc().read_text(encoding="utf-8"))

    def test_same_title_twice_on_a_page(self):
        reg = yaml.safe_load((self.root / "sources.yaml").read_text(encoding="utf-8"))
        reg["sites"]["s"]["split"] = "headings"
        (self.root / "sources.yaml").write_text(yaml.safe_dump(reg), encoding="utf-8")
        crawler = Crawler(self.root, self.root / "sources.yaml")
        crawler.new_session = lambda: self.http
        self.http.pages[self.URL] = ("<main><h2>Bourse</h2><p>texte un demande de bourse</p>"
                                     "<h2>Bourse</h2><p>texte un demande de bourse</p>"      # exact copy
                                     "<h2>Bourse</h2><p>texte deux autre bourse</p></main>").encode()
        counts = crawler.run()
        self.assertEqual((counts["new"], counts["duplicate"]), (2, 1))   # nothing overwritten

    def test_error_without_snapshot(self):
        self.assertEqual(self.run_once(None)["error"], 1)
        self.assertFalse((self.root / "data" / "corpus").exists() and
                         list((self.root / "data" / "corpus").rglob("*.md")))


if __name__ == "__main__":
    unittest.main()
