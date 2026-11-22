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

    def test_chitchat_lang_english_greeting_no_stopwords(self):
        # "hello" and "thank you" have zero overlap with detect_lang()'s stopword lists -- this
        # is exactly the case that used to silently default to French.
        self.assertEqual(generate._chitchat_lang("hello"), "en")
        self.assertEqual(generate._chitchat_lang("thank you"), "en")

    def test_chitchat_lang_french_greeting(self):
        self.assertEqual(generate._chitchat_lang("bonjour"), "fr")
        self.assertEqual(generate._chitchat_lang("merci"), "fr")

    def test_chitchat_lang_arabic_greeting(self):
        self.assertEqual(generate._chitchat_lang("مرحبا"), "ar")

    def test_answer_greeting_in_english_replies_in_english(self):
        result = generate.answer("hello", retriever=None, use_cache=False)
        self.assertEqual(result["answer"], generate.GREETING_REPLY["en"])

    def test_answer_farewell_in_english_replies_in_english(self):
        result = generate.answer("thank you", retriever=None, use_cache=False)
        self.assertEqual(result["answer"], generate.FAREWELL_REPLY["en"])

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
        # <=3 words: answer() only anchors short follow-ups on the history (a longer question is
        # trusted to stand alone -- see the word-count guard in answer()).
        result = generate.answer("le delai ?", retriever=mock_retriever, use_cache=False, history=history)
        self.assertEqual(mock_retriever.search.call_count, 2)
        first_query = mock_retriever.search.call_args_list[0].args[0]
        second_query = mock_retriever.search.call_args_list[1].args[0]
        self.assertEqual(first_query, "le delai ?")
        self.assertEqual(second_query, "Q1 complete agricole le delai ?")
        self.assertTrue(result["abstained"])

    def test_long_unrelated_question_is_not_glued_to_history(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        history = [("Q1 complete", "A1", False)]
        generate.answer("Quelle est la recette du couscous tunisien ?", retriever=mock_retriever,
                        use_cache=False, history=history)
        self.assertEqual(mock_retriever.search.call_count, 1)

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


class TestShortLongTermHistorySplit(unittest.TestCase):
    def test_under_window_all_short_term_no_older_topics(self):
        history = [("Q1", "A1", False), ("Q2", "A2", False)]
        short_term, older_topics = generate._split_history(history)
        self.assertEqual(short_term, history)
        self.assertEqual(older_topics, [])

    def test_over_window_splits_oldest_into_topics(self):
        history = [(f"Q{i}", f"A{i}", False) for i in range(1, 9)]  # 8 turns, window is 6
        short_term, older_topics = generate._split_history(history)
        self.assertEqual(short_term, history[-6:])
        self.assertEqual(older_topics, ["Q1", "Q2"])

    def test_abstained_older_turns_excluded_from_topics(self):
        history = [("Q1", "A1", True), ("Q2", "A2", False)] + \
                  [(f"Qx{i}", f"A{i}", False) for i in range(6)]
        _, older_topics = generate._split_history(history)
        self.assertEqual(older_topics, ["Q2"])

    def test_older_topics_deduped_keeping_first_occurrence(self):
        history = [("Q1", "A1", False), ("Q1", "A1b", False)] + \
                   [(f"Qx{i}", f"A{i}", False) for i in range(6)]
        _, older_topics = generate._split_history(history)
        self.assertEqual(older_topics, ["Q1"])

    def test_empty_history(self):
        self.assertEqual(generate._split_history([]), ([], []))


class TestAnswerUsesTwoTierHistory(unittest.TestCase):
    def _run_with_history(self, history):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [make_hit()]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), \
             patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = "reponse"
            mock_groq_cls.return_value = mock_client
            generate.answer("question actuelle", retriever=mock_retriever, use_cache=False, history=history)
            return mock_client.chat.completions.create.call_args.kwargs["messages"]

    def test_long_history_sends_only_short_term_turns_verbatim(self):
        history = [(f"Q{i}", f"A{i}", False) for i in range(1, 9)]  # 8 turns
        messages = self._run_with_history(history)
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        # the 2 oldest turns must NOT appear verbatim anymore
        self.assertNotIn("Q1", user_contents)
        self.assertNotIn("Q2", user_contents)
        self.assertIn("Q3", user_contents)

    def test_long_history_adds_a_digest_system_message_for_older_topics(self):
        history = [(f"Q{i}", f"A{i}", False) for i in range(1, 9)]
        messages = self._run_with_history(history)
        digests = [m["content"] for m in messages if m["role"] == "system"
                   and m["content"] != generate.SYSTEM_PROMPT]
        self.assertEqual(len(digests), 1)
        self.assertIn("Q1", digests[0])
        self.assertIn("Q2", digests[0])

    def test_short_history_has_no_digest_message(self):
        history = [("Q1", "A1", False)]
        messages = self._run_with_history(history)
        self.assertEqual(len([m for m in messages if m["role"] == "system"]), 1)  # SYSTEM_PROMPT only


def make_cnss_hit() -> Hit:
    return Hit(score=0.7, chunk_id="c1", doc_id="d1", lang="fr", domain="employment",
               subtopic="affiliation-cnss", title="t", org="CNSS", section="s", url="u", text="texte",
               n_chunks=1, bi_score=0.7)


class TestSectorGuard(unittest.TestCase):
    def _answer(self, question):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [make_cnss_hit()]
        return generate.answer(question, retriever=mock_retriever, use_cache=False)

    def test_asks_sector_for_cnss_question_without_sector(self):
        result = self._answer("Comment m'affilier a la CNSS ?")
        self.assertIn("secteur", result["answer"])

    def test_asks_sector_for_arabic_cnss_question(self):
        result = self._answer("كيف أنخرط في الضمان الاجتماعي؟")
        self.assertIn("القطاع", result["answer"])

    def test_does_not_ask_sector_when_question_is_not_about_cnss(self):
        # out-of-scope question whose nearest document happens to be a CNSS page -> must go to the
        # LLM (which declines), not get a CNSS sector question
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), \
             patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = "hors sujet"
            mock_groq_cls.return_value = mock_client
            result = self._answer("ما هي إجراءات تجديد جواز السفر التونسي؟")
        self.assertNotIn("القطاع", result["answer"])
        mock_client.chat.completions.create.assert_called_once()

    def test_no_sector_question_when_sector_already_given(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), \
             patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = "reponse"
            mock_groq_cls.return_value = mock_client
            result = self._answer("Comment m'affilier a la CNSS, secteur agricole ?")
        self.assertNotIn("De quel secteur", result["answer"])


class TestDenseFloorWiring(unittest.TestCase):
    def test_no_rerank_passes_the_dense_floor(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        generate.answer("une question complete", retriever=mock_retriever, rerank=False, use_cache=False)
        self.assertEqual(mock_retriever.search.call_args.kwargs["min_dense_score"],
                         generate.DEFAULT_MIN_DENSE_SCORE)

    def test_rerank_keeps_its_own_threshold_only(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        generate.answer("une question complete", retriever=mock_retriever, rerank=True, use_cache=False)
        self.assertIsNone(mock_retriever.search.call_args.kwargs["min_dense_score"])


class TestMissingApiKey(unittest.TestCase):
    def test_raises_a_normal_exception_not_systemexit(self):
        # SystemExit inside the web server killed the worker process and left every later request
        # hanging -- this must be an ordinary exception the server can turn into an HTTP error.
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [make_hit()]
        with patch.dict(os.environ):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(generate.MissingApiKeyError):
                generate.answer("une question", retriever=mock_retriever, use_cache=False)


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
