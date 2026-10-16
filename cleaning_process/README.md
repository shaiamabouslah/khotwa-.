# cleaning_process

Unzip at the repo root (files land in scripts/). Commit first: git is your backup.

## Order (PowerShell)
1. python scripts\fixes\check_quality.py            # baseline, expected to FAIL on raw data
2. python scripts\fixes\run_all.py                  # dry run: counts + examples
3. python scripts\fixes\run_all.py --apply          # repeats passes until stable
4. python scripts\fixes\check_quality.py            # must exit 0 now
5. patch scripts\clean_pipeline.py (line ~273) so flagged docs are skipped:
     if meta.get("status") != "active" or meta.get("rag_exclude") or not body.strip():
   $env:PYTHONPATH="."; python scripts\clean_pipeline.py --apply     # rebuilds kb/
6. python scripts\rag_prepare.py                    # kb/ -> data\rag\chunks.jsonl (+ quarantine, report)
7. python scripts\fixes\eval_retrieval.py --auto    # sanity floor (BM25)
   python scripts\fixes\eval_retrieval.py --file questions.jsonl   # real questions: {"q","doc_id","lang"}

## Review by hand
- data\state\fixes.jsonl        every change made, per file
- docs with rag_exclude: true   (kept on disk, skipped by the RAG)
- data\rag\quarantine.jsonl and review_dupes.json

## Re-crawl
The crawler rewrites changed pages from raw extraction: run steps 3-6 again after each crawl.
