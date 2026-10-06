"""Load and validate the source registry (sources.yaml)."""
import re
from pathlib import Path

import yaml

TRUST_LEVELS = {"official", "institutional", "program"}
STATUSES = {"active", "candidate", "excluded"}
RENDER_MODES = {"auto", "html", "js", "pdf"}
SPLIT_MODES = {"none", "headings", "titles", "accordion"}


def load_registry(path: Path) -> dict:
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))
    reg.setdefault("pages", [])
    return reg


def validate_registry(reg: dict, taxonomy: dict) -> list[str]:
    errors = []
    domains = taxonomy["domains"]
    for key, site in reg["sites"].items():
        where = f"sites.{key}"
        for field in ("base_url", "org", "trust", "status"):
            if not site.get(field):
                errors.append(f"{where}: missing '{field}'")
        if site.get("trust") and site["trust"] not in TRUST_LEVELS:
            errors.append(f"{where}: trust must be one of {sorted(TRUST_LEVELS)}")
        if site.get("status") and site["status"] not in STATUSES:
            errors.append(f"{where}: status must be one of {sorted(STATUSES)}")
        if site.get("render", "auto") not in RENDER_MODES:
            errors.append(f"{where}: render must be one of {sorted(RENDER_MODES)}")
        if site.get("split", "none") not in SPLIT_MODES:
            errors.append(f"{where}: split must be one of {sorted(SPLIT_MODES)}")
        if site.get("domain") and site["domain"] not in domains:
            errors.append(f"{where}: unknown domain {site['domain']}")
        disc = site.get("discover", {})
        if site.get("status") == "active" and not (disc.get("seeds") or disc.get("sitemap")):
            errors.append(f"{where}: active site needs discover.seeds or discover.sitemap")
        for field in ("follow", "keep", "exclude"):
            for pattern in disc.get(field, []) or []:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{where}.discover.{field}: bad regex {pattern!r} ({exc})")
        for rule in disc.get("rewrite", []) or []:
            if not (isinstance(rule, list) and len(rule) == 2):
                errors.append(f"{where}.discover.rewrite: expected [pattern, replacement]")
        for lang, rule in (site.get("lang_variants") or {}).items():
            if lang != "only" and not (isinstance(rule, list) and len(rule) == 2):
                errors.append(f"{where}.lang_variants.{lang}: expected [pattern, replacement]")
    for i, page in enumerate(reg.get("pages", [])):
        if page.get("site") not in reg["sites"]:
            errors.append(f"pages[{i}]: unknown site {page.get('site')}")
        if page.get("split", "none") not in SPLIT_MODES:
            errors.append(f"pages[{i}]: bad split mode")
        if page.get("split") == "titles" and not page.get("titles"):
            errors.append(f"pages[{i}]: split 'titles' needs a 'titles' list")
    return errors


def lang_versions(site: dict, url: str) -> tuple[str, dict[str, str]]:
    """Return (base_url_without_language, {lang: url}) using `lang_variants`.

    Example (cnss.tn keeps the language in a cookie unless the path is prefixed):
      lang_variants:
        base: ['^(https://www\\.cnss\\.tn)/(?:fr/|ar/)?web/', '\\1/web/']
        fr:   ['^(https://www\\.cnss\\.tn)/web/', '\\1/fr/web/']
        ar:   ['^(https://www\\.cnss\\.tn)/web/', '\\1/ar/web/']
    Without `lang_variants` the page is fetched once and its language detected."""
    rules = site.get("lang_variants") or {}
    if not rules or (rules.get("only") and not re.search(rules["only"], url)):
        return url, {}
    base = re.sub(rules["base"][0], rules["base"][1], url) if "base" in rules else url
    return base, {lang: re.sub(p, r, base) for lang, (p, r) in rules.items() if lang not in ("base", "only")}
