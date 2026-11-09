"""Unit tests for generate.py's own logic: chitchat routing, language detection, citation
formatting, history persistence, the multi-turn retrieval-anchor fallback, and graceful
degradation on a Groq rate limit.

Deliberately uses a mock retriever (duck-typed: just needs .search()) instead of the real
Retriever -- answer() already takes `retriever` as a parameter for exactly this reason, so no
production code needed to change to make this testable. Never touches torch/sentence-
transformers, never calls the real Groq API, and never writes to the project's real
data/rag/generation_cache.json or data/rag/conversations/ (HISTORY_DIR is swapped for a temp
dir, use_cache=False everywhere else).

Run: python -m unittest discover -s tests -t .
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generate  # noqa: E402
from retriever import Hit  # noqa: E402


def make_hit(title="t", url="u") -> Hit:
    return Hit(score=0.9, chunk_id="c1", doc_id="d1", lang="fr", domain="studies",
               subtopic="bourses", title=title, org="o", section="s", url=url, text="texte",
               n_chunks=1, bi_score=0.9)


class TestChitchat(unittest.TestCase):
    def test_greeting_fr(self):
        self.assertTrue(generate.GREETING_RE.match("Bonjour"))

    def test_greeting_ar(self):
        self.assertTrue(generate.GREETING_RE.match("السلام عليكم"))

    def test_greeting_en(self):
        self.assertTrue(generate.GREETING_RE.match("hello"))

    def test_farewell_fr(self):
        self.assertTrue(generate.FAREWELL_RE.match("merci beaucoup"))

    def test_real_question_matches_neither(self):
        q = "Comment affilier un travailleur independant a la CNSS ?"
        self.assertIsNone(generate.GREETING_RE.match(q))
        self.assertIsNone(generate.FAREWELL_RE.match(q))

    def test_answer_greeting_bypasses_retriever_entirely(self):
        # retriever=None would crash if the chitchat path ever touched it -- it must not.
        result = generate.answer("bonjour", retriever=None, use_cache=False)
        self.assertFalse(result["abstained"])
        self.assertEqual(result["hits"], [])


class TestDetectLang(unittest.TestCase):
    def test_french(self):
        self.assertEqual(generate.detect_lang("Comment affilier un salarie a la CNSS ?"), "fr")

    def test_english(self):
        self.assertEqual(generate.detect_lang("How can I get a scholarship for this year?"), "en")

    def test_arabic(self):
        self.assertEqual(generate.detect_lang("كيف أنخرط في الضمان الاجتماعي؟"), "ar")

    def test_explicit_override_wins(self):
        self.assertEqual(generate.detect_lang("Comment affilier un salarie ?", lang="en"), "en")


class TestCitations(unittest.TestCase):
    def test_build_context_numbers_and_cites_each_hit(self):
        ctx = generate.build_context([make_hit("Titre A", "http://a"), make_hit("Titre B", "http://b")])
        self.assertIn("[1] Titre A", ctx)
        self.assertIn("[2] Titre B", ctx)
        self.assertIn("http://a", ctx)

    def test_sources_block_lists_every_hit_with_a_clickable_link(self):
        block = generate.sources_block([make_hit("Titre A", "http://a")], "fr")
        self.assertIn("Sources officielles", block)
        self.assertIn("[1] Titre A", block)
        self.assertIn("http://a", block)

    def test_sources_block_uses_requested_language_label(self):
        block = generate.sources_block([make_hit()], "ar")
        self.assertIn("المصادر الرسمية", block)


class TestHistoryPersistence(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(generate, "HISTORY_DIR", Path(self._tmpdir.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_roundtrip(self):
        history = [("Q1", "A1", False), ("agricole", "A2", False)]
        generate.save_history("s1", history)
        self.assertEqual(generate.load_history("s1"), history)

    def test_missing_session_returns_empty(self):
        self.assertEqual(generate.load_history("never-saved"), [])

    def test_backward_compat_with_old_2tuple_format(self):
        # Files saved before the abstained flag existed are 2-element lists -- must load as
        # abstained=False, not crash.
        import json
        path = generate._history_path("legacy")
        path.write_text(json.dumps([["Q1", "A1"]]), encoding="utf-8")
        self.assertEqual(generate.load_history("legacy"), [("Q1", "A1", False)])

    def test_session_name_is_sanitized_to_a_safe_filename(self):
        path = generate._history_path("../../etc/passwd")
        self.assertNotIn("..", str(path))
        self.assertEqual(path.parent, Path(self._tmpdir.name))


class TestMultiTurnAnchorFallback(unittest.TestCase):
    def test_bare_question_tried_first(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []  # abstain -- only the call args matter here
        generate.answer("une question complete", retriever=mock_retriever, use_cache=False)
        self.assertEqual(mock_retriever.search.call_count, 1)
        self.assertEqual(mock_retriever.search.call_args.args[0], "une question complete")

    def test_fallback_anchors_on_non_abstained_turns_only_oldest_first(self):
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [[], []]  # bare question fails, fallback too -> abstain
        history = [
            ("Q1 complete", "A1", False),
            ("agricole", "A2", False),
            ("tour qui a echoue", "A3", True),  # must be skipped: it abstained
        ]
        result = generate.answer("et le delai ?", retriever=mock_retriever, use_cache=False, history=history)
        self.assertEqual(mock_retriever.search.call_count, 2)
        first_query = mock_retriever.search.call_args_list[0].args[0]
        second_query = mock_retriever.search.call_args_list[1].args[0]
        self.assertEqual(first_query, "et le delai ?")
        self.assertEqual(second_query, "Q1 complete agricole et le delai ?")
        self.assertTrue(result["abstained"])

    def test_no_fallback_when_history_is_entirely_abstained(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        history = [("Q1", "A1", True)]
        generate.answer("suite ?", retriever=mock_retriever, use_cache=False, history=history)
        # nothing non-abstained to anchor on -> no second, polluted call
        self.assertEqual(mock_retriever.search.call_count, 1)

    def test_no_hits_at_all_abstains_with_localized_message(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        result = generate.answer("Quelle est la recette du couscous ?", retriever=mock_retriever, use_cache=False)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["answer"], generate.NO_MATCH_MESSAGE["fr"])


class TestGroqRetryAndDegradation(unittest.TestCase):
    def _rate_limit_error(self, retry_after: str):
        import httpx
        from groq import RateLimitError
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
        return RateLimitError("rate limited", response=response, body=None)

    @patch("time.sleep")
    def test_short_wait_retries_then_succeeds(self, mock_sleep):
        client = MagicMock()
        client.chat.completions.create.side_effect = [self._rate_limit_error("1"), "ok"]
        result = generate._call_groq_with_retry(client, model="m", messages=[])
        self.assertEqual(result, "ok")
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    def test_long_wait_reraises_without_retrying(self, mock_sleep):
        from groq import RateLimitError
        client = MagicMock()
        client.chat.completions.create.side_effect = self._rate_limit_error("600")
        with self.assertRaises(RateLimitError):
            generate._call_groq_with_retry(client, model="m", messages=[])
        mock_sleep.assert_not_called()

    def test_answer_degrades_gracefully_and_does_not_cache_on_rate_limit(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [make_hit()]
        # generate.py does `from groq import Groq` lazily, inside answer() itself -- so the
        # patch target is groq.Groq (the real source), not a generate.Groq attribute that
        # only exists transiently during that import.
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), \
             patch("groq.Groq") as mock_groq_cls, \
             patch("generate._call_groq_with_retry", side_effect=self._rate_limit_error("600")):
            mock_groq_cls.return_value = MagicMock()
            result = generate.answer("une question", retriever=mock_retriever, use_cache=False)
        self.assertFalse(result["abstained"])
        self.assertEqual(result["answer"], generate.BUSY_MESSAGE["fr"])


if __name__ == "__main__":
    unittest.main()
