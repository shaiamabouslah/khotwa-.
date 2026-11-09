"""Unit tests for Retriever's confidence gate (Retriever._apply_min_score).

Deliberately does NOT instantiate Retriever() or import sentence_transformers/torch: this
static method is the one piece of retriever.py that is pure data-in/data-out logic (no model,
no I/O), so it is tested as such -- fast, no model download, safe to run on every push. The
model's own retrieval quality (hit@1/3/5, MRR) is a separate, slower evaluation
(scripts/fixes/eval_dense.py, scripts/compare_chunking.py), not a unit test.

Run: python -m unittest discover -s tests -t .
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from retriever import BI_SCORE_OVERRIDE, DEFAULT_MIN_RERANK_SCORE, Hit, Retriever  # noqa: E402


def make_hit(score: float, bi_score: float | None = None) -> Hit:
    return Hit(score=score, chunk_id="c1", doc_id="d1", lang="fr", domain="studies",
               subtopic="bourses", title="t", org="o", section="s", url="u", text="txt",
               n_chunks=1, bi_score=bi_score)


class TestApplyMinScore(unittest.TestCase):
    def test_passes_through_above_threshold(self):
        hits = [make_hit(DEFAULT_MIN_RERANK_SCORE + 0.01)]
        self.assertEqual(Retriever._apply_min_score(hits, True, DEFAULT_MIN_RERANK_SCORE), hits)

    def test_abstains_below_threshold_without_override(self):
        hits = [make_hit(0.05, bi_score=0.3)]
        self.assertEqual(Retriever._apply_min_score(hits, True, DEFAULT_MIN_RERANK_SCORE), [])

    def test_override_rescues_low_rerank_high_bi_score(self):
        hits = [make_hit(0.02, bi_score=BI_SCORE_OVERRIDE)]
        self.assertEqual(Retriever._apply_min_score(hits, True, DEFAULT_MIN_RERANK_SCORE), hits)

    def test_override_does_not_fire_just_below_cutoff(self):
        hits = [make_hit(0.02, bi_score=BI_SCORE_OVERRIDE - 0.01)]
        self.assertEqual(Retriever._apply_min_score(hits, True, DEFAULT_MIN_RERANK_SCORE), [])

    def test_override_ignored_when_bi_score_missing(self):
        hits = [make_hit(0.02, bi_score=None)]
        self.assertEqual(Retriever._apply_min_score(hits, True, DEFAULT_MIN_RERANK_SCORE), [])

    def test_empty_hits_returns_empty(self):
        self.assertEqual(Retriever._apply_min_score([], True, DEFAULT_MIN_RERANK_SCORE), [])

    def test_rerank_false_bypasses_threshold_entirely(self):
        hits = [make_hit(0.0)]
        self.assertEqual(Retriever._apply_min_score(hits, False, DEFAULT_MIN_RERANK_SCORE), hits)

    def test_min_score_none_bypasses_threshold(self):
        hits = [make_hit(0.0)]
        self.assertEqual(Retriever._apply_min_score(hits, True, None), hits)


if __name__ == "__main__":
    unittest.main()
