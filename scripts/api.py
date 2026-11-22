#!/usr/bin/env python3
"""Web backend for Khotwa -- login + per-user, multi-conversation chat over generate.answer(),
and serves the static UI in ../static/.

All the actual assistant logic (chitchat routing, abstention, sector clarification, short/
long-term history) already lives in generate.answer()/load_history()/save_history() -- this
module is just an HTTP shell around them (plus accounts, in auth.py), so the CLI
(`generate.py --chat`) and the web UI stay in sync with zero duplicated assistant logic.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe -m uvicorn scripts.api:app --reload
    # then open http://127.0.0.1:8000/ (served at /, see StaticFiles mount below)

Needs GROQ_API_KEY (see generate.py's own docstring) and the same embeddings cache generate.py
needs. SESSION_SECRET should be set in .env for login sessions to survive a server restart; if
unset, a random one is generated at startup (fine for a quick demo, just means every restart logs
everyone out).

The retriever (embedding model + reranker) only loads on the first /chat call that actually needs
it (see get_retriever() below) -- it can take anywhere from several seconds to a few minutes
depending on the machine. It is deliberately NOT warmed up eagerly at server startup: doing that
once starved unrelated requests (plain conversation list/create/delete/rename, which never touch
the retriever) while the heavy CPU-bound load was running in the background.

Conversations: each is stored via generate.load_history/save_history under the composite key
"{username}__{conversation_id}" -- reuses that storage mechanism completely as-is (it already
sanitizes to a safe filename), so a user's conversations are just every data/rag/conversations/
file whose name starts with "{username}__". No separate index to keep in sync.
"""
import json
import os
import secrets
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import auth  # noqa: E402
import generate  # noqa: E402
from retriever import Retriever  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Lazy singleton, built at most once, only when the first real question actually needs it (NOT
# warmed up eagerly at server startup -- tried that, but loading the embedding model + reranker
# is heavy enough on CPU that doing it in a background thread right as the server starts starved
# unrelated requests, including plain conversation list/create/delete/rename calls that never
# touch the retriever at all. A lock still guards against two concurrent /chat requests (e.g. two
# different new conversations) each starting their own expensive build before the first finishes.
_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


app = FastAPI(title="Khotwa API")

_session_secret = os.environ.get("SESSION_SECRET")
if not _session_secret:
    print("SESSION_SECRET not set in .env -- using a one-off key for this run; "
          "every restart will log all users out. Set SESSION_SECRET in .env to avoid that.")
    _session_secret = secrets.token_hex(32)
app.add_middleware(SessionMiddleware, secret_key=_session_secret, same_site="lax", https_only=False)


def require_user(request: Request) -> str:
    username = request.session.get("username")
    if not username:
        raise HTTPException(401, "not authenticated")
    return username


class Credentials(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    conversation_id: str
    question: str
    lang: str | None = None


class ChatResponse(BaseModel):
    answer: str
    abstained: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(creds: Credentials, request: Request):
    try:
        auth.create_user(creds.username, creds.password)
    except auth.UserError as e:
        raise HTTPException(409, str(e))
    request.session["username"] = creds.username
    return {"username": creds.username}


@app.post("/auth/login")
def login(creds: Credentials, request: Request):
    if not auth.verify_user(creds.username, creds.password):
        raise HTTPException(401, "nom d'utilisateur ou mot de passe incorrect")
    request.session["username"] = creds.username
    return {"username": creds.username}


@app.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


@app.get("/auth/me")
def me(username: str = Depends(require_user)):
    return {"username": username}


def _conversation_key(username: str, conversation_id: str) -> str:
    return f"{username}__{conversation_id}"


# Custom conversation titles (set via rename) live in a small sidecar file next to the history
# files themselves, keyed by the same "{username}__{conversation_id}" string -- kept separate from
# generate.py's own history JSON so a rename is pure UI metadata, not conversation content, and
# generate.py needs zero changes to support it.
TITLES_PATH = generate.HISTORY_DIR / "_titles.json"


def _load_titles() -> dict:
    if not TITLES_PATH.exists():
        return {}
    try:
        return json.loads(TITLES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_titles(titles: dict):
    TITLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    TITLES_PATH.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")


class RenameRequest(BaseModel):
    title: str


@app.get("/conversations")
def list_conversations(username: str = Depends(require_user)):
    titles = _load_titles()
    out = []
    for path in generate.HISTORY_DIR.glob(f"{username}__*.json"):
        conv_id = path.stem[len(username) + 2:]
        key = _conversation_key(username, conv_id)
        if key in titles:
            title = titles[key]
        else:
            history = generate.load_history(key)
            title = history[0][0][:40] if history else "Nouvelle conversation"
        out.append({"id": conv_id, "title": title, "updated_at": path.stat().st_mtime})
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out


@app.post("/conversations", status_code=201)
def create_conversation(username: str = Depends(require_user)):
    conv_id = secrets.token_hex(4)
    generate.save_history(_conversation_key(username, conv_id), [])
    return {"id": conv_id}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, username: str = Depends(require_user)):
    key = _conversation_key(username, conversation_id)
    path = generate._history_path(key)
    if not path.exists():
        raise HTTPException(404, "conversation introuvable")
    history = generate.load_history(key)
    return {"id": conversation_id,
            "turns": [{"question": q, "answer": a, "abstained": ab} for q, a, ab in history]}


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, body: RenameRequest, username: str = Depends(require_user)):
    key = _conversation_key(username, conversation_id)
    path = generate._history_path(key)
    if not path.exists():
        raise HTTPException(404, "conversation introuvable")
    title = body.title.strip()[:60]
    if not title:
        raise HTTPException(400, "le titre ne peut pas etre vide")
    titles = _load_titles()
    titles[key] = title
    _save_titles(titles)
    return {"id": conversation_id, "title": title}


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, username: str = Depends(require_user)):
    key = _conversation_key(username, conversation_id)
    path = generate._history_path(key)
    if not path.exists():
        raise HTTPException(404, "conversation introuvable")
    path.unlink()
    titles = _load_titles()
    if titles.pop(key, None) is not None:
        _save_titles(titles)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, username: str = Depends(require_user)):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    key = _conversation_key(username, req.conversation_id)
    path = generate._history_path(key)
    if not path.exists():
        raise HTTPException(404, "conversation introuvable -- cree-la via POST /conversations")

    # Chitchat (greeting/farewell) is answered inside generate.answer() without ever touching
    # the retriever -- but Python evaluates call arguments eagerly, so passing get_retriever()
    # directly would load the whole embedding model + reranker on EVERY first message, including
    # a bare "bonjour", before answer() even reaches the point where it would decide it doesn't
    # need one. Skip building it for exactly the same chitchat patterns answer() itself checks
    # first (generate.GREETING_RE / generate.FAREWELL_RE -- reused, not re-implemented here).
    needs_retriever = not (generate.GREETING_RE.match(question) or generate.FAREWELL_RE.match(question))
    retriever = get_retriever() if needs_retriever else None

    history = generate.load_history(key)
    result = generate.answer(question, retriever, lang=req.lang, use_cache=False, history=history)
    history.append((question, result["answer"], result["abstained"]))
    generate.save_history(key, history)

    return ChatResponse(answer=result["answer"], abstained=result["abstained"])


# Mounted last: StaticFiles with html=True serves static/index.html at "/" and would otherwise
# shadow routes declared after it.
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
