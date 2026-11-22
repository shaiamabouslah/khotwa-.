#!/usr/bin/env python3
"""Local account store for the web UI (scripts/api.py) -- usernames + password hashes in
data/users.json, same flat-JSON-file convention this project already uses everywhere else
(generation_cache.json, query_cache.json, data/rag/conversations/*.json). Not meant to scale
past a handful of demo accounts; a real deployment would swap this for a proper user database
without touching api.py's calls into this module.

Passwords are never stored in plain text: PBKDF2-HMAC-SHA256 with a random per-user salt
(stdlib hashlib, no bcrypt/passlib dependency needed for this scale).
"""
import hashlib
import hmac
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USERS_PATH = ROOT / "data" / "users.json"

USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
_PBKDF2_ITERATIONS = 310_000


class UserError(Exception):
    """Invalid username/password, or a username that's already taken."""


def _load_users() -> dict:
    if not USERS_PATH.exists():
        return {}
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict):
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


def create_user(username: str, password: str):
    if not USERNAME_RE.match(username):
        raise UserError("le nom d'utilisateur doit faire 3 a 32 caracteres (lettres, chiffres, _ ou -)")
    if not password:
        raise UserError("le mot de passe ne peut pas etre vide")

    users = _load_users()
    if username in users:
        raise UserError("ce nom d'utilisateur est deja pris")

    salt = os.urandom(16)
    users[username] = {
        "salt": salt.hex(),
        "hash": _hash_password(password, salt),
    }
    _save_users(users)


def verify_user(username: str, password: str) -> bool:
    users = _load_users()
    entry = users.get(username)
    if not entry:
        return False
    salt = bytes.fromhex(entry["salt"])
    # Constant-time compare -- this is a login check, not worth a timing side-channel over.
    return hmac.compare_digest(_hash_password(password, salt), entry["hash"])
