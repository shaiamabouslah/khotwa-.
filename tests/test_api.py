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
        api._retriever = MagicMock()
        # Each TestClient instance keeps its own cookie jar -- a fresh one per registered user
        # simulates two different browsers/people.
        self.client = TestClient(api.app)

    def tearDown(self):
        self._history_patcher.stop()
        self._users_patcher.stop()
        self._titles_patcher.stop()
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
                         [{"question": "Q1", "answer": "A1", "abstained": False, "sources": []}])

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
        self.assertEqual(resp.json(), {"answer": "reponse", "abstained": False, "sources": []})

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
