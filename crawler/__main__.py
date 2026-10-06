"""Khotwa crawler command line.

  python -m crawler validate                    check sources.yaml (offline)
  python -m crawler probe <url> [<url>...]      inspect a site before adding it (robots, sitemap, static/JS)
  python -m crawler probe --all                 re-probe every site of the registry
  python -m crawler discover --site cnss        list the pages a site's rules would crawl (no extraction)
  python -m crawler crawl                       discover + crawl + sync all active sites
  python -m crawler crawl --site cnss --site ooun   only these sites (candidates allowed here)
  python -m crawler status                      summary of the corpus and of the last run
  python -m crawler sections <url>              headings/services found on one page

crawl options: --force (ignore ETag/Last-Modified/sitemap lastmod), --dry-run (write nothing),
               --strict (exit 1 on errors, for CI), --workers N (sites in parallel, default 4)
"""
import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from .discover import Discoverer
from .extract import html_to_markdown
from .http import PoliteSession
from .pipeline import Crawler
from .probe import as_row, probe
from .registry import load_registry, validate_registry

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "sources.yaml"
STATE = ROOT / "data" / "state"


def setup_logging(verbose: bool):
    STATE.mkdir(parents=True, exist_ok=True)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    logfile = logging.FileHandler(STATE / "crawl.log", encoding="utf-8")
    logfile.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", handlers=[console, logfile])


def session(reg) -> PoliteSession:
    d = reg["defaults"]
    return PoliteSession(d["user_agent"], d["delay_seconds"], d["timeout_seconds"],
                         respect_robots=d.get("respect_robots_txt", True))


def cmd_validate(reg) -> int:
    taxonomy = yaml.safe_load((ROOT / "kb" / "taxonomy.yaml").read_text(encoding="utf-8"))
    errors = validate_registry(reg, taxonomy)
    by_status = Counter(s["status"] for s in reg["sites"].values())
    for e in errors:
        print("ERROR", e)
    print(f"{len(reg['sites'])} sites {dict(by_status)}, {len(reg['pages'])} page overrides, {len(errors)} error(s)")
    return 1 if errors else 0


def cmd_probe(reg, urls: list[str], all_sites: bool) -> int:
    if all_sites:
        urls = [s["base_url"] for s in reg["sites"].values() if s["status"] != "excluded"]
    if not urls:
        sys.exit("give at least one URL, or --all")

    def one(url):
        return as_row(probe(session(reg), url))

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, urls))
    for r in rows:
        robots = {True: "allowed", False: "DISALLOWED", None: "-"}[r["robots_allows_us"]]
        print(f"{r['url']}\n  status={r['status']}  render={r['render'] or '-'} ({r['render_reason']})\n"
              f"  robots={robots}  ai_bots_blocked={r['blocks_ai_bots'] or 'none'}  sitemap={r['sitemap'] or '-'}\n"
              f"  tls_valid={r['tls_valid']}  cms={r['cms'] or '-'}  title={r['title']!r}")
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "probe.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_discover(reg, sites: list[str]) -> int:
    keys = sites or [k for k, s in reg["sites"].items() if s["status"] == "active"]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda k: Discoverer(session(reg), reg["sites"][k]).run({}), keys))
    for key, found in zip(keys, results):
        print(f"\n## {key}: {len(found)} URL(s)")
        for f in found:
            print(f"  [{f.via:7}] {f.url}" + (f"  (lastmod {f.lastmod})" if f.lastmod else ""))
    return 0


def cmd_crawl(reg, args) -> int:
    for key in args.site:
        if key not in reg["sites"]:
            sys.exit(f"unknown site {key!r}")
        if reg["sites"][key]["status"] == "excluded":
            sys.exit(f"site {key!r} is excluded: {reg['sites'][key].get('reason')}")
    crawler = Crawler(ROOT, REGISTRY, force=args.force, dry_run=args.dry_run, workers=args.workers)
    counts = crawler.run(args.site or None)
    print("\nSummary:", dict(counts))
    print("Corpus: data/corpus/ (catalog.jsonl)  History: data/state/changelog.jsonl  Log: data/state/crawl.log")
    return 1 if args.strict and (counts["error"] or counts["stale"]) else 0


def cmd_status() -> int:
    path = STATE / "sync_state.json"
    if not path.exists():
        sys.exit("no crawl yet: python -m crawler crawl")
    state = json.loads(path.read_text(encoding="utf-8"))
    pages = state["pages"]
    print(f"last update: {state['updated_at']}   documents: {len(pages)}")
    print("by status :", dict(Counter(p["status"] for p in pages.values())))
    print("by site   :", dict(Counter(p["site"] for p in pages.values())))
    metas = [p.get("meta", {}) for p in pages.values()]
    print("by lang   :", dict(Counter(m.get("lang") for m in metas)))
    print("by domain :", dict(Counter(m.get("domain_hint") for m in metas)))
    print("by type   :", dict(Counter(m.get("doc_type") for m in metas)))
    for key, p in sorted(pages.items()):
        if p["status"] in ("error", "stale", "removed"):
            print(f"  {p['status']:8} {key}")
    last = STATE / "last_run.json"
    if last.exists():
        print("last run  :", last.read_text(encoding="utf-8").replace("\n", " "))
    return 0


def cmd_sections(reg, url: str) -> int:
    site = next((s for s in reg["sites"].values() if url.startswith(s["base_url"])), {})
    resp = session(reg).fetch(url)
    if not resp.ok:
        sys.exit(f"cannot fetch {url}: {resp.error or resp.status}")
    cfg = dict(site.get("extract", {}))
    if site.get("split") == "accordion":
        cfg["accordion"] = True
    ex = html_to_markdown(resp.content, url, cfg)
    print(f"title: {ex.title}\nbreadcrumb: {ex.breadcrumb}\nsource_updated: {ex.source_updated}\n")
    for s in ex.sections:
        print(" -", s)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="python -m crawler", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("probe")
    p.add_argument("urls", nargs="*")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("discover")
    p.add_argument("--site", action="append", default=[])
    p = sub.add_parser("crawl")
    p.add_argument("--site", action="append", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("-q", "--quiet", action="store_true")
    sub.add_parser("status")
    p = sub.add_parser("sections")
    p.add_argument("url")
    args = ap.parse_args()

    reg = load_registry(REGISTRY)
    if args.cmd == "validate":
        sys.exit(cmd_validate(reg))
    if args.cmd == "status":
        sys.exit(cmd_status())
    setup_logging(verbose=not getattr(args, "quiet", False) and args.cmd == "crawl")
    if args.cmd == "probe":
        sys.exit(cmd_probe(reg, args.urls, args.all))
    if args.cmd == "discover":
        sys.exit(cmd_discover(reg, args.site))
    if args.cmd == "crawl":
        sys.exit(cmd_crawl(reg, args))
    if args.cmd == "sections":
        sys.exit(cmd_sections(reg, args.url))


if __name__ == "__main__":
    main()
