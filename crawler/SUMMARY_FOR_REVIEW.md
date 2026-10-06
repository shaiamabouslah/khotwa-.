# Khotwa — project explanation (Part 1: the crawler)

## The project
**Khotwa** ("step" in Arabic) is a university MLOps project (TEK-UP University). It's a
platform that answers, in French or Arabic, questions about **Tunisian administrative
procedures** — things like "how do I register as a freelancer with social security?"
or "how do I get my foreign degree recognized?".

It covers two domains:
- **Studies** — degree equivalence, scholarships, university enrollment, academic
  mobility (Erasmus), internship conventions...
- **Employment** — social security affiliation, freelance/patente registration,
  business creation, employment certificates...

The architecture has three parts:
1. **Crawler** (this part, built) — collect official procedure pages from ~17
   Tunisian government/university websites into one unified corpus.
2. **Cleaning pipeline** (next, not built yet) — deduplicate, merge, and turn that raw
   corpus into a clean knowledge base (one Markdown file per procedure, fixed template).
3. **RAG + router** (later) — a router classifies a user's question into one of the two
   domains, then a retrieval system answers from the matching domain's knowledge base.

Constraint: everything must be free/open-source (university module requirement, no
paid APIs/tools).

## What the crawler (Part 1) does
Given a list of official websites (ministries, the social security fund CNSS,
university offices, national government portals), it automatically:
1. **Finds the relevant pages** on each site (via its sitemap, a few starting pages,
   and link-following rules) — no one maintains a manual list of URLs.
2. **Detects how each page is built**: plain HTML, a PDF, or a JavaScript app — and
   fetches it accordingly.
3. **Extracts the content as clean Markdown**, keeping headings, lists, tables, links,
   and discarding menus/ads/footers.
4. **Splits a page into several procedures** when it bundles more than one (e.g. a
   government portal page listing 5 different services).
5. **Tags each procedure** with a guessed domain/topic (Studies vs Employment, and a
   sub-topic) using keyword matching, and detects its language (French/Arabic/English).
6. **Keeps everything in sync**: re-running the crawl later only re-downloads pages
   that changed, keeps the last good version if a site goes down temporarily, and logs
   every change in a history file.

One specific site, a popular private procedure aggregator (idaraty.tn), is deliberately
**excluded**: its robots.txt explicitly forbids AI crawlers and its terms forbid
reproducing content, so it's only browsed manually (by a human) to find candidate
official sources, never crawled automatically.

## Where it stands
The crawler is built and tested (17 offline unit tests). A first full run against the
real 17 sites produced about 1,200 procedure documents in French/Arabic/English.
Two bugs were found and fixed during that run (duplicate requests to nonexistent URL
variants, and a rare case where two same-titled sections on one page could overwrite
each other). The next step is Part 2: cleaning and deduplicating that raw corpus into
the final knowledge base.
