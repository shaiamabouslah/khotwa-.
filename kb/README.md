# Khotwa knowledge base

## Where this fits: the three data layers

| Layer | Folder | Produced by | Content |
|---|---|---|---|
| 0. Raw | `data/raw/` | crawler | exact bytes fetched (HTML / PDF), for reprocessing |
| 1. Corpus | `data/corpus/` + `catalog.jsonl` | crawler (`python -m crawler crawl`) | every document from every source, **same front matter**, site structure preserved, domain/subtopic as *hints* |
| 2. Knowledge base | `kb/` (this folder) | cleaning pipeline (next stage) | de-duplicated, classified, merged FR/AR, one procedure per file in the 8-section template below |

Sources and crawling rules: [`sources.yaml`](../sources.yaml) and [`crawler/README.md`](../crawler/README.md).

The knowledge base (KB) holds every procedure Khotwa can answer about. It is split into two
**fully separate domains**, which become the two RAG indexes:

| Domain | Folder | Index |
|---|---|---|
| Études & Mobilité Académique | `kb/studies/` | studies index |
| Emploi & Vie Professionnelle | `kb/employment/` | employment index |

The structure is defined once in [`taxonomy.yaml`](taxonomy.yaml) and enforced by
`python scripts/validate_kb.py`.

## Layout

```
kb/
├── README.md              # this contract
├── taxonomy.yaml          # domains, subtopics, section headings (source of truth)
├── _templates/            # procedure.fr.md, procedure.ar.md
├── studies/<subtopic>/<id>.<lang>.md
└── employment/<subtopic>/<id>.<lang>.md
```

**Subtopics**

- `studies/`: `equivalence`, `bourses`, `inscription`, `releves-attestations`, `legalisation`, `stages-pfe`, `mobilite`
- `employment/`: `affiliation-cnss`, `employes-de-maison`, `non-salaries`, `patente-fiscalite`, `attestations`, `creation-entreprise`

## Rules

1. **One file = one procedure.** A source page that describes several procedures
   (e.g. OOUN's scholarship page: new bachelor, renewal, doctoral student…) becomes several
   files, each pointing to the same URL with a different `sources[].section`.
2. **File name = `<id>.<lang>.md`**, placed in `kb/<domain>/<subtopic>/`.
   The `id` starts with the subtopic, is kebab-case, and **never changes** once created
   (the chunk index and evaluation set refer to it).
3. **Languages:** `fr` and `ar`. The two language versions of a procedure share the same `id`
   and list each other in `translations`. A missing translation is allowed.
4. **Fixed body:** the 8 `##` sections below, always present, always in this order.
   No other `##` heading, no `#` heading. Use `###`, bold labels or lists inside a section.

   | key | fr | ar |
   |---|---|---|
   | resume | Résumé | ملخص |
   | conditions | Conditions | الشروط |
   | documents | Pièces à fournir | الوثائق المطلوبة |
   | etapes | Étapes | المراحل |
   | delais | Délais et coûts | الآجال والمعاليم |
   | lieu | Où s'adresser | الجهة المعنية |
   | formulaires | Formulaires | المطبوعات |
   | references | Références juridiques | المراجع القانونية |

5. **A section the source does not cover is not deleted.** Write
   `_Non précisé par la source._` (fr) or `_غير مذكور في المصدر._` (ar).
   The chunker skips these sections.
6. **Nothing invented.** Every fact must come from a URL in `sources`. The `Résumé` and the
   numbered `Étapes` may rephrase the source, but must not add information.
   Figures that look outdated are kept as written by the source and flagged in `notes`.
7. **Official sources only.** No aggregator (idaraty.tn, demarches.tn…) in `sources`:
   they may be used by hand to *discover* procedures, never as the cited source.

## Front matter (metadata)

| Field | Required | Description |
|---|---|---|
| `id` | ✅ | Stable identifier, `<subtopic>-<slug>` |
| `title` | ✅ | Official name of the procedure |
| `domain` | ✅ | `studies` or `employment` |
| `subtopic` | ✅ | Key from `taxonomy.yaml` (must match the folder) |
| `lang` | ✅ | `fr` or `ar` |
| `translations` | | Other languages available for this `id` |
| `org` | ✅ | Administration in charge |
| `audience` | | Who it concerns, in plain words |
| `keywords` | ✅ | Synonyms, acronyms, Tunisian/Arabic terms users actually type |
| `related` | | `id`s of linked procedures, **including across domains** |
| `sources` | ✅ | List of `{url, breadcrumb, section?, source_updated?}` |
| `last_checked` | ✅ | Last date the source was fetched and compared |
| `last_changed` | ✅ | Last date the content of this procedure actually changed |
| `status` | ✅ | `active`, `stale` (source unreachable, last good version kept), `deprecated` |
| `notes` | | Caveats: outdated figures, contradictions between sources |

`breadcrumb` preserves where the procedure sits in the source site's own structure
(e.g. `[CNSS, Non-salariés, Indépendants, Secteur non agricole, Affiliation]`), so
answers can cite it precisely. `source_updated` is the date the source itself displays,
when it displays one (CNSS does; most sites do not).

## How the KB is used downstream

- **Chunking:** one chunk per non-empty section; each chunk carries `id`, `domain`,
  `subtopic`, `lang`, section key, `title`, `breadcrumb`, `url`, `last_changed`.
- **Retrieval:** search over chunks, return the whole parent procedure (all its sections).
- **Sync:** when a source changes, the procedure file is regenerated, `last_changed` is
  updated, and the change is visible as a git diff on the `.md` file.

## Reference examples

- [`employment/non-salaries/non-salaries-affiliation-non-agricole.fr.md`](employment/non-salaries/non-salaries-affiliation-non-agricole.fr.md): a full page, all sections covered, legal references.
- [`studies/bourses/bourses-nouveau-bachelier.fr.md`](studies/bourses/bourses-nouveau-bachelier.fr.md): one section of a multi-procedure page, with outdated figures flagged in `notes`.
