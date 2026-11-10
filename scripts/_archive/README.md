# Archive

One-off scripts, each already run once against the corpus to fix a specific past issue
(a path-naming migration, two passes of the same language-detection bug, a metadata
retrofit, a hand-reviewed stub list). None of them are part of the active pipeline
(`scripts/clean_pipeline.py`) and none should be re-run against the current corpus --
they assume a corpus shape from when they were written, and may do nothing or something
wrong against today's data.

Kept for history, not for reuse. See git log for when/why each one ran.
