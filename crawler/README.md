# Crawler & sync

Builds the **unified corpus** of Khotwa from the official sources declared in
[`sources.yaml`](../sources.yaml), and keeps it in sync.

```
sources.yaml (sites) ─▶ discover ─▶ fetch (HTML / JS / PDF, auto-detected) ─▶ extract ─▶ split
                                                                                         │
           data/corpus/<site>/<url path>/<doc>.<lang>.md  +  catalog.jsonl  ◀─ sync ◀─ classify
```

## Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Only if you activate a JavaScript site (`render: js`, e.g. `utm`, `rne_app`):

```bash
pip install -r requirements-js.txt
playwright install chromium
```

## Commands

```bash
python -m crawler validate                 # check sources.yaml (offline, instant)
python -m crawler discover --site cnss     # which pages would be crawled for a site (no extraction)
python -m crawler crawl --site ooun        # small test first
python -m crawler crawl                    # all active sites (= the sync: run it again any time)
python -m crawler status                   # corpus summary: by site, language, domain, type, status
```

| Command / option | Use |
|---|---|
| `probe <url>...` / `probe --all` | inspect a site before adding it: robots.txt, AI-bot blocking, sitemap, TLS, CMS, **static HTML or JS app** |
| `discover [--site K]` | test a site's discovery rules |
| `crawl [--site K ...]` | discover + crawl + sync (a `candidate` site can be crawled explicitly with `--site`) |
| `crawl --force` | ignore ETag / Last-Modified / sitemap `lastmod`, re-download everything |
| `crawl --dry-run` | do everything but write nothing |
| `crawl --strict` | exit code 1 on errors or stale documents (for CI / GitHub Actions) |
| `crawl --workers N` | sites crawled in parallel (default 4; each site stays at 1 request / 2 s) |
| `sections <url>` | headings / services of a page (to write `pages:` overrides) |
| `status` | summary of the corpus and of the last run |

## Adding a source

1. `python -m crawler probe https://new-site.tn/` → reachable? robots OK? sitemap? static or JS?
2. Add a block under `sites:` in `sources.yaml` (copy a similar site; start with `status: candidate`).
3. `python -m crawler validate` then `python -m crawler discover --site <key>` until the URL list is right.
4. `python -m crawler crawl --site <key>`, look at `data/corpus/<key>/`, then set `status: active`.

## Output: the unified corpus

```
data/
├── corpus/
│   ├── catalog.jsonl                         # one line per document: THE input of the cleaning stage
│   └── <site>/<url path>/<doc>.<lang>.md     # mirrors each site's own structure
├── raw/<site>/<url path>/content.bin         # exact bytes fetched (HTML or PDF), for reprocessing
└── state/
    ├── sync_state.json                       # hash / status of every document and URL
    ├── changelog.jsonl                       # every event: new, changed, stale, recovered, removed
    ├── last_run.json, probe.json
    └── crawl.log
```

`<doc>` is `index` for a whole page, or the slug of the section when one page holds several
documents (CNSS article = 1 document; services.gov.tn page = 1 document per service;
OOUN page = 1 document per student profile).

Every document has the **same front matter**, whatever its source:

```yaml
doc_id: cnss:/web/non-salarie/.../non_sal-secteur_-non_agr1_affiliation#index
site: cnss
org: Caisse Nationale de Sécurité Sociale (CNSS)
trust: official                 # official | institutional | program
url: https://www.cnss.tn/fr/web/non-salarie/...
lang: fr                        # fr | ar | en (declared variant or detected)
title: ...
section: null                   # title of the block when the page was split
breadcrumb: [CNSS, Non Salarié, ...]
doc_type: procedure             # procedure | announcement | form | guide | document | page
domain_hint: employment         # rule-based hints from kb/taxonomy.yaml keywords,
subtopic_hint: non-salaries     #   to be confirmed by the cleaning stage
hint_score: 7
hint_keywords: [non salarie, affiliation]
forms: [https://www.cnss.tn/c/document_library/get_file?uuid=...]
render: html                    # html | js | pdf (how it was fetched)
discovered_via: link            # sitemap | seed | link | pinned
source_updated: 2024-06-27      # date displayed by the source itself, when it has one
http_last_modified: null
sitemap_lastmod: null
fetched_at: 2026-10-03T...
first_seen: 2026-10-03T...
last_changed: 2026-10-03T...
content_hash: sha256:...
words: 542
status: active                  # active | stale | removed
```

## Sync rules

| Situation | Status | Document file |
|---|---|---|
| First time seen | `new` | written |
| Same content hash | `unchanged` | **not touched** (clean git / DVC diff) |
| Different hash | `changed` | rewritten, `last_changed` updated |
| Source down, previous version exists | `stale` | **kept**, `status: stale` |
| Source down, never fetched | `error` | nothing written |
| Source back | `unchanged` + event `recovered` | `status: active` |
| Page no longer discovered, 1st run | `missing` | untouched (a seed may just be down) |
| Page no longer discovered, 2nd run in a row | `removed` | kept, `status: removed` |

Cost savings: sitemap `lastmod` unchanged → no request at all; ETag / Last-Modified → `304`
without body; pages fetched during discovery are reused by the crawl.

The hash covers the extracted content only: menus, news boxes or session ids changing
around a procedure are not counted as changes.

## Tests

```bash
python -m unittest discover -s tests -t .
```

Offline (no network): encodings, hierarchy, accordions, splitting, render detection,
language, classifier, registry, discovery rules, language variants, and the full sync
lifecycle (new → unchanged → changed → stale → recovered, missing → removed).
