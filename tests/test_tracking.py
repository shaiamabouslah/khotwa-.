"""Unit tests for scripts/tracking.py and the pure parts of scripts/eval_generation.py.

No network, no MLflow server, no Groq call. Run: python -m unittest discover -s tests -t .
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import tracking  # noqa: E402


class TestTracking(unittest.TestCase):
    def test_metric_names_are_valid_mlflow_keys(self):
        self.assertEqual(tracking.metric_name("dense", "all", "hit@1"), "dense/all/hit_at_1")

    def test_nested_metrics_are_flattened_and_non_numbers_skipped(self):
        mlflow = MagicMock()
        tracking.log_metrics_tree(mlflow, {"dense": {"all": {"hit@5": 0.99, "n": 94}, "ok": True, "note": "x"}})
        self.assertEqual(mlflow.log_metrics.call_args.args[0], {"dense/all/hit_at_5": 0.99, "dense/all/n": 94.0})

    def test_credentials_are_read_from_the_dagshub_dvc_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".dvc").mkdir()
            (Path(tmp) / ".dvc" / "config.local").write_text(
                "['remote \"other\"']\n    user = someone\n    password = nope\n"
                "['remote \"dagshub\"']\n    user = shaimabouslah\n    password = abc123\n", encoding="utf-8")
            with patch.object(tracking, "ROOT", Path(tmp)):
                self.assertEqual(tracking._dvc_credentials(), ("shaimabouslah", "abc123"))

    def test_disabled_without_tracking_uri(self):
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": " "}):
            self.assertFalse(tracking.enabled())


class TestGenerationEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import eval_generation
        cls.ev = eval_generation

    def test_sample_covers_every_language(self):
        positives = [{"q": f"{lang}{i}", "lang": lang} for lang in ("fr", "ar", "en", "derja") for i in range(10)]
        picked = self.ev.sample(positives, [{"q": "neg"}], n=8, n_neg=1, seed=0)
        self.assertEqual(sorted(q["lang"] for q in picked if q["kind"] == "in_scope"),
                         ["ar", "ar", "derja", "derja", "en", "en", "fr", "fr"])
        self.assertEqual(picked[-1]["kind"], "out_of_scope")

    def test_summary_rates(self):
        row = lambda kind, out, gold=None, grounded=None: {"kind": kind, "outcome": out, "gold_cited": gold,
                                                          "grounded": grounded, "latency_s": 2.0}
        m = self.ev.summarize([row("in_scope", "answer", True, True), row("in_scope", "answer", False, False),
                               row("in_scope", "clarification"), row("out_of_scope", "uncited"),
                               row("out_of_scope", "answer", None, True)])
        self.assertEqual((m["answer_rate"], m["gold_document_cited_rate"], m["clarification_rate"]),
                         (0.6667, 0.3333, 0.3333))
        self.assertEqual((m["grounded_answer_rate"], m["out_of_scope_refused_rate"]), (0.6667, 0.5))


if __name__ == "__main__":
    unittest.main()
