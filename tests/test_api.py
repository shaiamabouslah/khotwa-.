"""Unit tests for scripts/api.py -- auth, conversations, and the /chat shell around
generate.answer().

Patches api.generate.answer directly and swaps generate.HISTORY_DIR / auth.USERS_PATH for temp
locations, so this never loads the embedding model, never calls the real Groq API, and never
touches the project's real data/rag/conversations/ or data/users.json. api._retriever is
pre-seeded with a MagicMock in setUp, before get_retriever() could lazily build a real one.

Run: python -m unittest discover -s tests -t .
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import api  # noqa: E402
import auth  # noqa: E402
import generate  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


def _answer_stub(answer_text="reponse", abstained=False):
    return {"answer": answer_text, "hits": [], "abstained": abstained}


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._history_patcher = patch.object(generate, "HISTORY_DIR", Path(self._tmpdir.name))
        self._history_patcher.start()
        self._users_patcher = patch.object(auth, "USERS_PATH", Path(self._tmpdir.name) / "users.json")
        self._users_patcher.start()
        # api.TITLES_PATH is computed once from generate.HISTORY_DIR at import time -- patching
        # HISTORY_DIR above doesn't retroactively move it, so it's patched separately here.
        self._titles_patcher = patch.object(api, "TITLES_PATH", Path(self._tmpdir.name) / "_titles.json")
        self._titles_patcher.start()
        self._events_patcher = patch.object(api.monitoring, "EVENTS_PATH", Path(self._tmpdir.name) / "events.jsonl")
        self._events_patcher.start()
        api._retriever = MagicMock()
        # Each TestClient instance keeps its own cookie jar -- a fresh one per registered user
        # simulates two different browsers/people.
        self.client = TestClient(api.app)

    def tearDown(self):
        self._history_patcher.stop()
        self._users_patcher.stop()
        self._titles_patcher.stop()
        self._events_patcher.stop()
        api._retriever = None
        self._tmpdir.cleanup()

    def _register(self, client, username, password="motdepasse123"):
        return client.post("/auth/register", json={"username": username, "password": password})

    def _new_conversation(self, client):
        return client.post("/conversations").json()["id"]


class TestAuthEndpoints(ApiTestBase):
    def test_register_logs_in_and_me_reflects_it(self):
        resp = self._register(self.client, "etudiant1")
        self.assertEqual(resp.status_code, 201)
        me = self.client.get("/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json(), {"username": "etudiant1"})

    def test_duplicate_register_rejected(self):
        self._register(self.client, "etudiant1")
        resp = self._register(self.client, "etudiant1")
        self.assertEqual(resp.status_code, 409)

    def test_me_without_session_is_401(self):
        resp = self.client.get("/auth/me")
        self.assertEqual(resp.status_code, 401)

    def test_login_with_correct_password(self):
        self._register(self.client, "etudiant1")
        self.client.post("/auth/logout")
        resp = self.client.post("/auth/login", json={"username": "etudiant1", "password": "motdepasse123"})
        self.assertEqual(resp.status_code, 200)

    def test_login_with_wrong_password_is_401(self):
        self._register(self.client, "etudiant1")
        self.client.post("/auth/logout")
        resp = self.client.post("/auth/login", json={"username": "etudiant1", "password": "faux"})
        self.assertEqual(resp.status_code, 401)

    def test_logout_then_me_is_401(self):
        self._register(self.client, "etudiant1")
        self.client.post("/auth/logout")
        resp = self.client.get("/auth/me")
        self.assertEqual(resp.status_code, 401)


class TestConversations(ApiTestBase):
    def test_create_then_list(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        listed = self.client.get("/conversations").json()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], conv_id)
        self.assertEqual(listed[0]["title"], "Nouvelle conversation")

    def test_list_requires_auth(self):
        resp = self.client.get("/conversations")
        self.assertEqual(resp.status_code, 401)

    def test_chat_updates_title_to_first_question(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=_answer_stub("A1")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "Q1"})
        listed = self.client.get("/conversations").json()
        self.assertEqual(listed[0]["title"], "Q1")

    def test_get_conversation_returns_turns(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=_answer_stub("A1")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "Q1"})
        resp = self.client.get(f"/conversations/{conv_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["turns"],
                         [{"question": "Q1", "answer": "A1", "abstained": False, "sources": [], "options": [],
                           "turn": 0}])

    def test_get_unknown_conversation_is_404(self):
        self._register(self.client, "etudiant1")
        resp = self.client.get("/conversations/doesnotexist")
        self.assertEqual(resp.status_code, 404)

    def test_delete_removes_conversation(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        resp = self.client.delete(f"/conversations/{conv_id}")
        self.assertEqual(resp.status_code, 204)
        self.assertEqual(self.client.get("/conversations").json(), [])

    def test_delete_unknown_conversation_is_404(self):
        self._register(self.client, "etudiant1")
        resp = self.client.delete("/conversations/doesnotexist")
        self.assertEqual(resp.status_code, 404)

    def test_rename_changes_title_in_list(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        resp = self.client.patch(f"/conversations/{conv_id}", json={"title": "Mon suivi CNSS"})
        self.assertEqual(resp.status_code, 200)
        listed = self.client.get("/conversations").json()
        self.assertEqual(listed[0]["title"], "Mon suivi CNSS")

    def test_rename_survives_first_message_title_autoupdate(self):
        # Once renamed, sending a message must NOT silently revert the title back to the
        # question text -- only a still-default "Nouvelle conversation" gets auto-titled.
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        self.client.patch(f"/conversations/{conv_id}", json={"title": "Mon suivi CNSS"})
        with patch.object(api.generate, "answer", return_value=_answer_stub("A1")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "Q1"})
        listed = self.client.get("/conversations").json()
        self.assertEqual(listed[0]["title"], "Mon suivi CNSS")

    def test_rename_blank_title_rejected(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        resp = self.client.patch(f"/conversations/{conv_id}", json={"title": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_rename_unknown_conversation_is_404(self):
        self._register(self.client, "etudiant1")
        resp = self.client.patch("/conversations/doesnotexist", json={"title": "x"})
        self.assertEqual(resp.status_code, 404)

    def test_rename_requires_auth(self):
        resp = self.client.patch("/conversations/x", json={"title": "y"})
        self.assertEqual(resp.status_code, 401)

    def test_user_cannot_rename_another_users_conversation(self):
        self._register(self.client, "alice")
        conv_id = self._new_conversation(self.client)

        other_client = TestClient(api.app)
        self._register(other_client, "bob")
        resp = other_client.patch(f"/conversations/{conv_id}", json={"title": "hijack"})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(self.client.get("/conversations").json()[0]["title"], "Nouvelle conversation")

    def test_delete_clears_its_custom_title(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        self.client.patch(f"/conversations/{conv_id}", json={"title": "Mon suivi CNSS"})
        self.client.delete(f"/conversations/{conv_id}")
        titles = api._load_titles()
        self.assertNotIn(api._conversation_key("etudiant1", conv_id), titles)

    def test_user_cannot_see_or_delete_another_users_conversation(self):
        self._register(self.client, "alice")
        conv_id = self._new_conversation(self.client)

        other_client = TestClient(api.app)
        self._register(other_client, "bob")
        self.assertEqual(other_client.get("/conversations").json(), [])
        resp = other_client.delete(f"/conversations/{conv_id}")
        self.assertEqual(resp.status_code, 404)
        # alice's conversation must survive bob's attempt
        self.assertEqual(len(self.client.get("/conversations").json()), 1)


class TestChatEndpoint(ApiTestBase):
    def test_chat_requires_auth(self):
        resp = self.client.post("/chat", json={"conversation_id": "x", "question": "q"})
        self.assertEqual(resp.status_code, 401)

    def test_chat_against_unknown_conversation_is_404(self):
        self._register(self.client, "etudiant1")
        resp = self.client.post("/chat", json={"conversation_id": "doesnotexist", "question": "q"})
        self.assertEqual(resp.status_code, 404)

    def test_chat_returns_answer_and_abstained(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=_answer_stub("reponse", False)):
            resp = self.client.post("/chat", json={"conversation_id": conv_id, "question": "une question"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"answer": "reponse", "abstained": False, "sources": [], "options": [],
                                       "turn": 0})

    def test_blank_question_rejected(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        resp = self.client.post("/chat", json={"conversation_id": conv_id, "question": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_history_round_trips_across_two_calls(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=_answer_stub("A1")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "Q1"})

        captured = {}

        def stub(*args, **kwargs):
            captured["history"] = list(kwargs["history"])
            return _answer_stub("A2")

        with patch.object(api.generate, "answer", side_effect=stub):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "Q2"})

        self.assertEqual(captured["history"], [("Q1", "A1", False)])
        key = api._conversation_key("etudiant1", conv_id)
        self.assertEqual(generate.load_history(key), [("Q1", "A1", False), ("Q2", "A2", False)])

    def test_greeting_does_not_build_the_retriever(self):
        # Regression test: get_retriever() loads the embedding model + reranker, which can take
        # a long time -- a bare "bonjour" must stay instant, same as generate.answer() itself
        # already guarantees for chitchat (generate.GREETING_RE / FAREWELL_RE).
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api, "get_retriever") as mock_get_retriever, \
             patch.object(api.generate, "answer", return_value=_answer_stub("Bonjour !")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "bonjour"})
        mock_get_retriever.assert_not_called()

    def test_real_question_builds_the_retriever(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api, "get_retriever") as mock_get_retriever, \
             patch.object(api.generate, "answer", return_value=_answer_stub("reponse")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "une vraie question"})
        mock_get_retriever.assert_called_once()

    def _answer_with_sources(self, body):
        from retriever import Hit
        hits = [Hit(score=0.9, chunk_id=f"c{i}", doc_id=f"d{i}", lang="fr", domain="studies",
                    subtopic="bourses", title=f"Titre {i}", org="Caisse Nationale de Sécurité Sociale (CNSS)",
                    section="s", url=f"https://www.cnss.tn/page{i}", text="t", n_chunks=1)
                for i in (1, 2, 3)]
        return _answer_stub(body + generate.sources_block(hits, "fr"))

    def test_chat_returns_sources_separately_and_only_cited_ones(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        stub = self._answer_with_sources("Il faut un **dossier** [1] et une copie [3].")
        with patch.object(api.generate, "answer", return_value=stub):
            body = self.client.post("/chat", json={"conversation_id": conv_id, "question": "q"}).json()
        self.assertEqual(body["answer"], "Il faut un **dossier** [1] et une copie [3].")
        self.assertEqual([s["n"] for s in body["sources"]], [1, 3])
        self.assertEqual(body["sources"][0]["url"], "https://www.cnss.tn/page1")
        self.assertEqual(body["sources"][0]["org"], "Caisse Nationale de Sécurité Sociale (CNSS)")

    def test_reopened_conversation_also_gets_sources_separately(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=self._answer_with_sources("Voir [2].")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "q"})
        turn = self.client.get(f"/conversations/{conv_id}").json()["turns"][0]
        self.assertEqual(turn["answer"], "Voir [2].")
        self.assertEqual([s["n"] for s in turn["sources"]], [2])

    def test_uncited_answer_shows_no_sources(self):
        # e.g. "the excerpts don't cover this" -- showing source buttons would imply they back it
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        stub = self._answer_with_sources("Les extraits ne couvrent pas cette question.")
        with patch.object(api.generate, "answer", return_value=stub):
            body = self.client.post("/chat", json={"conversation_id": conv_id, "question": "q"}).json()
        self.assertEqual(body["answer"], "Les extraits ne couvrent pas cette question.")
        self.assertEqual(body["sources"], [])

    def _clarification(self):
        labels = [generate.SECTOR_LABELS[s]["fr"] for s in generate.SECTOR_SUBTOPICS["non-salaries"]]
        return generate.SECTOR_CLARIFY_MESSAGE["fr"].format(options=", ".join(labels)), labels

    def test_clarification_comes_with_clickable_options(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        msg, labels = self._clarification()
        with patch.object(api.generate, "answer", return_value=_answer_stub(msg)):
            body = self.client.post("/chat", json={"conversation_id": conv_id, "question": "q"}).json()
        self.assertEqual(body["options"], labels)

    def test_reopened_conversation_offers_options_only_on_the_open_question(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        msg, labels = self._clarification()
        for reply in (msg, "Reponse finale.", msg):
            with patch.object(api.generate, "answer", return_value=_answer_stub(reply)):
                self.client.post("/chat", json={"conversation_id": conv_id, "question": "q"})
        turns = self.client.get(f"/conversations/{conv_id}").json()["turns"]
        self.assertEqual([t["options"] for t in turns], [[], [], labels])

    def _events(self):
        return api.monitoring.read_events()

    def test_each_question_is_logged_with_its_outcome(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=self._answer_with_sources("Voir [1].")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "une vraie question"})
        with patch.object(api.generate, "answer", return_value=_answer_stub("Je n'ai pas trouvé.")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "une autre question"})
        with patch.object(api.generate, "answer", return_value=_answer_stub("pas d'info", abstained=True)):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "couscous ?"})
        with patch.object(api.generate, "answer", return_value=_answer_stub("Bonjour !")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "bonjour"})
        events = self._events()
        self.assertEqual([e["outcome"] for e in events], ["answer", "uncited", "no_match", "chitchat"])
        self.assertTrue(all("latency_ms" in e for e in events))
        self.assertNotIn("etudiant1", str(events))  # usernames are stored hashed

    def test_failed_question_is_logged_as_error(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", side_effect=ValueError("boom")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "une question"})
        self.assertEqual(self._events()[0]["outcome"], "error")

    def test_busy_reply_is_logged_but_not_kept_in_history_nor_rateable(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        busy = dict(_answer_stub(generate.BUSY_MESSAGE["ar"]), busy=True)
        with patch.object(api.generate, "answer", return_value=busy):
            data = self.client.post("/chat", json={"conversation_id": conv_id, "question": "سؤال"}).json()
        self.assertIsNone(data["turn"])
        self.assertEqual(self._events()[0]["outcome"], "busy")
        self.assertEqual(self.client.get(f"/conversations/{conv_id}").json()["turns"], [])
        seen = []
        def answer(*args, history, **kwargs):
            seen.append(list(history))
            return self._answer_with_sources("Voir [1].")
        with patch.object(api.generate, "answer", side_effect=answer):
            data = self.client.post("/chat", json={"conversation_id": conv_id, "question": "سؤال"}).json()
        self.assertEqual(seen, [[]])
        self.assertEqual(data["turn"], 0)

    def test_empty_reply_is_logged_as_empty_and_not_kept(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        empty = dict(_answer_stub(generate.EMPTY_ANSWER_MESSAGE["fr"]), busy=True, empty=True)
        with patch.object(api.generate, "answer", return_value=empty):
            data = self.client.post("/chat", json={"conversation_id": conv_id, "question": "une question"}).json()
        self.assertEqual(data["answer"], generate.EMPTY_ANSWER_MESSAGE["fr"])
        self.assertIsNone(data["turn"])
        self.assertEqual(self._events()[0]["outcome"], "empty")
        self.assertEqual(self.client.get(f"/conversations/{conv_id}").json()["turns"], [])
        self.assertEqual(self.client.get("/metrics").json()["error_rate"], 1.0)

    def test_feedback_counts_only_the_latest_vote(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=self._answer_with_sources("Voir [1].")):
            turn = self.client.post("/chat", json={"conversation_id": conv_id, "question": "q"}).json()["turn"]
        for value in ("down", "up"):
            r = self.client.post("/feedback", json={"conversation_id": conv_id, "turn": turn, "value": value})
            self.assertEqual(r.status_code, 200)
        m = self.client.get("/metrics").json()
        self.assertEqual((m["feedback"]["up"], m["feedback"]["down"]), (1, 0))
        self.assertEqual(m["recent_downvoted"], [])

    def test_feedback_validation(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        self.assertEqual(self.client.post("/feedback", json={"conversation_id": conv_id, "turn": 0,
                                                             "value": "meh"}).status_code, 400)
        self.assertEqual(self.client.post("/feedback", json={"conversation_id": conv_id, "turn": 5,
                                                             "value": "up"}).status_code, 404)

    def test_metrics_aggregate_and_require_login(self):
        self.assertEqual(self.client.get("/metrics").status_code, 401)
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=self._answer_with_sources("Voir [1].")):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "q1"})
        with patch.object(api.generate, "answer", return_value=_answer_stub("pas d'info", abstained=True)):
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "recette du couscous ?"})
        m = self.client.get("/metrics").json()
        self.assertEqual(m["questions"], 2)
        self.assertEqual(m["answer_rate"], 0.5)
        self.assertEqual(m["recent_unanswered"][0]["question"], "recette du couscous ?")

    def test_answer_with_a_figure_in_no_source_is_flagged_on_metrics(self):
        from retriever import Hit
        hit = Hit(score=0.9, chunk_id="c1", doc_id="d1", lang="fr", domain="studies", subtopic="bourses",
                  title="Bourse", org="OOUN", section="s", url="https://www.ooun.rnu.tn/bourse",
                  text="Le revenu ne doit pas dépasser 1200 dinars.", n_chunks=1)
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        for body in ("Plafond : 1200 dinars [1].", "Plafond : 1500 dinars [1]."):
            stub = {"answer": body + generate.sources_block([hit], "fr"), "hits": [hit], "abstained": False}
            with patch.object(api.generate, "answer", return_value=stub):
                self.client.post("/chat", json={"conversation_id": conv_id, "question": "plafond de revenu ?"})
        m = self.client.get("/metrics").json()
        self.assertEqual(m["grounding"], {"checked": 2, "flagged": 1, "flagged_rate": 0.5})
        self.assertEqual(m["recent_flagged"][0]["unsupported"], ["1500"])

    def test_health_reports_key_and_retriever_state(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("groq_key", body)
        self.assertIn("retriever_ready", body)

    def test_missing_api_key_returns_503_and_server_stays_up(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", side_effect=generate.MissingApiKeyError("no key")):
            resp = self.client.post("/chat", json={"conversation_id": conv_id, "question": "une question"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("GROQ_API_KEY", resp.json()["detail"])
        self.assertEqual(self.client.get("/conversations").status_code, 200)

    def test_chat_uses_the_web_rerank_setting(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=_answer_stub("A")) as mock_answer:
            self.client.post("/chat", json={"conversation_id": conv_id, "question": "une vraie question"})
        self.assertEqual(mock_answer.call_args.kwargs["rerank"], api.WEB_RERANK)

    def test_unexpected_error_returns_500_not_a_hang(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", side_effect=ValueError("boom")):
            resp = self.client.post("/chat", json={"conversation_id": conv_id, "question": "une question"})
        self.assertEqual(resp.status_code, 500)

    def test_abstained_flag_passes_through(self):
        self._register(self.client, "etudiant1")
        conv_id = self._new_conversation(self.client)
        with patch.object(api.generate, "answer", return_value=_answer_stub("pas d'info", True)):
            resp = self.client.post("/chat", json={"conversation_id": conv_id, "question": "couscous ?"})
        self.assertTrue(resp.json()["abstained"])


if __name__ == "__main__":
    unittest.main()
