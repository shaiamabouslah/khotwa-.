"""Unit tests for the retrieval quality gate (scripts/check_metrics.py).

Run: python -m unittest discover -s tests -t .
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_metrics import violations  # noqa: E402


def _metrics(hit1=0.75, hit5=0.93, mrr=0.82, ar_hit5=0.90):
    group = lambda h1, h5, m: {"n": 10, "hit@1": h1, "hit@3": h5, "hit@5": h5, "mrr": m}
    return {"dense": {"all": group(hit1, hit5, mrr), "ar": group(hit1, ar_hit5, mrr),
                      "negatives_blocked_by_floor": 0.2, "seconds_per_question": 0.1}}


class TestQualityGate(unittest.TestCase):
    def test_current_level_passes(self):
        self.assertEqual(violations(_metrics()), [])

    def test_overall_regression_fails(self):
        self.assertEqual(len(violations(_metrics(hit5=0.80))), 1)

    def test_one_language_collapsing_fails_even_if_overall_is_fine(self):
        self.assertEqual(violations(_metrics(ar_hit5=0.50)), ["ar hit@5 = 0.50 < 0.80"])


if __name__ == "__main__":
    unittest.main()
