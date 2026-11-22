"""Unit tests for scripts/auth.py -- local account store (data/users.json).

Run: python -m unittest discover -s tests -t .
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import auth  # noqa: E402


class TestCreateUser(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(auth, "USERS_PATH", Path(self._tmpdir.name) / "users.json")
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_create_then_verify_succeeds(self):
        auth.create_user("etudiant1", "motdepasse123")
        self.assertTrue(auth.verify_user("etudiant1", "motdepasse123"))

    def test_wrong_password_rejected(self):
        auth.create_user("etudiant1", "motdepasse123")
        self.assertFalse(auth.verify_user("etudiant1", "mauvais"))

    def test_unknown_username_rejected(self):
        self.assertFalse(auth.verify_user("jamais-cree", "x"))

    def test_duplicate_username_rejected(self):
        auth.create_user("etudiant1", "motdepasse123")
        with self.assertRaises(auth.UserError):
            auth.create_user("etudiant1", "autre-mdp")

    def test_empty_password_rejected(self):
        with self.assertRaises(auth.UserError):
            auth.create_user("etudiant1", "")

    def test_invalid_username_format_rejected(self):
        for bad in ("ab", "a" * 33, "a b", "a/b", "../etc"):
            with self.assertRaises(auth.UserError):
                auth.create_user(bad, "motdepasse123")

    def test_password_not_stored_in_plain_text(self):
        auth.create_user("etudiant1", "motdepasse123")
        raw = auth.USERS_PATH.read_text(encoding="utf-8")
        self.assertNotIn("motdepasse123", raw)


if __name__ == "__main__":
    unittest.main()
