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

The retriever starts loading in a background thread at server startup (~15s: imports + BGE-M3 +
the embeddings cache), so the first question doesn't pay for it. The web app answers without the
cross-encoder reranker -- see WEB_RERANK below for the measurements behind that.

Conversations: each is stored via generate.load_history/save_history under the composite key
"{username}__{conversation_id}" -- reuses that storage mechanism completely as-is (it already
sanitizes to a safe filename), so a user's conversations are just every data/rag/conversations/
file whose name starts with "{username}__". No separate index to keep in sync.
"""
import json
import os
import re
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import auth  # noqa: E402
import generate  # noqa: E402
import grounding  # noqa: E402
import monitoring  # noqa: E402
from retriever import Retriever  # noqa: E402

if not os.environ.get("GROQ_API_KEY"):
    print("\n" + "!" * 78 + "\n"
          f"  GROQ_API_KEY is not set ({ROOT / '.env'} missing or has no GROQ_API_KEY line).\n"
          "  Greetings will still work, but every real question will fail until it is set.\n"
          + "!" * 78 + "\n", flush=True)

# Measured 2026-11-22 on this project's laptop CPU (6 cores): with the cross-encoder reranker
# (bge-reranker-v2-m3, 568M params, 20 candidates) a single question took ~57s to retrieve --
# intrinsic compute, truncating inputs to 512 tokens didn't fix it. Dense-only retrieval takes
# ~0.16s with hit@5 = 0.95 on the 41 real questions (the reranker only reorders inside the top 5,
# which the LLM reads in full anyway). What the reranker uniquely gave was a calibrated
# out-of-scope threshold; without it, abstention uses retriever.DEFAULT_MIN_DENSE_SCORE plus the
# generation prompt's own "say so if the excerpts don't answer" rule.
WEB_RERANK = False

# Singleton, built once: a lock guards against two concurrent first /chat requests each starting
# their own build. Warmed up in the background at startup (lifespan below).
_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


def _warm_up():
    try:
        get_retriever()
        print("Retriever ready.", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app):
    # Background thread, so the page and auth endpoints answer immediately while it loads. (An
    # earlier attempt at this was reverted on a wrong diagnosis: what actually froze the server
    # back then was generate.answer() calling sys.exit() when GROQ_API_KEY was missing.)
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Khotwa API", lifespan=lifespan)

_session_secret = os.environ.get("SESSION_SECRET")
if not _session_secret:
    print("SESSION_SECRET not set in .env -- using a one-off key for this run; "
          "every restart will log all users out. Set SESSION_SECRET in .env to avoid that.")
    _session_secret = secrets.token_hex(32)
# KHOTWA_HTTPS=1 when served behind HTTPS: the session cookie is then never sent over plain HTTP.
app.add_middleware(SessionMiddleware, secret_key=_session_secret, same_site="lax",
                   https_only=os.environ.get("KHOTWA_HTTPS") == "1")


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


class Source(BaseModel):
    n: int
    title: str
    org: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source] = []
    options: list[str] = []  # clickable replies when the answer is a clarification question
    turn: int | None = None  # index of this exchange in the conversation (for feedback)


def _present(text: str) -> tuple[str, list[dict]]:
    """Answer body + the sources the UI shows as links: only the ones the answer actually cites
    ([1], [3]...). A source button claims "this answer comes from here" -- an answer that cites
    nothing (typically "the excerpts don't cover this") gets none. History keeps the full text."""
    body, sources = generate.split_sources(text)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    return body, [s for s in sources if s["n"] in cited]


@app.get("/health")
def health():
    return {"status": "ok",
            "groq_key": bool(os.environ.get("GROQ_API_KEY")),
            "retriever_ready": _retriever is not None}


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
    turns = []
    for i, (q, a, ab) in enumerate(history):
        body, sources = _present(a)
        # only the last turn's clarification is still open; earlier ones were already answered
        options = generate.clarify_options(body) if i == len(history) - 1 else []
        turns.append({"question": q, "answer": body, "abstained": ab, "sources": sources, "options": options,
                      "turn": i})
    return {"id": conversation_id, "turns": turns}


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
    lang = req.lang or (generate.detect_lang(question) if needs_retriever else generate._chitchat_lang(question))
    event = {"event": "question", "user": monitoring.user_hash(username), "conversation_id": req.conversation_id,
             "lang": lang, "question": question}
    started = time.perf_counter()

    history = generate.load_history(key)
    try:
        retriever = get_retriever() if needs_retriever else None
        result = generate.answer(question, retriever, lang=req.lang, rerank=WEB_RERANK,
                                 use_cache=False, history=history)
    except generate.MissingApiKeyError:
        monitoring.log_event(**event, outcome="error", error="MissingApiKeyError",
                             latency_ms=round((time.perf_counter() - started) * 1000))
        raise HTTPException(503, "Le serveur n'a pas de cle GROQ_API_KEY configuree (fichier .env) -- "
                                 "impossible de generer une reponse.")
    except Exception as e:
        # Logged in full for the developer; the browser gets a short, readable reason instead of
        # a request that never resolves.
        import traceback
        traceback.print_exc()
        monitoring.log_event(**event, outcome="error", error=type(e).__name__,
                             latency_ms=round((time.perf_counter() - started) * 1000))
        raise HTTPException(500, f"Erreur interne pendant la generation de la reponse : {type(e).__name__}")
    body, sources = _present(result["answer"])
    options = generate.clarify_options(body)
    if not needs_retriever:
        outcome = "chitchat"
    elif options:
        outcome = "clarification"
    elif result["abstained"]:
        outcome = "no_match"
    elif result.get("busy"):
        outcome = "busy"
    else:
        outcome = "answer" if sources else "uncited"
    extra = {}
    if outcome in ("answer", "uncited") and result.get("hits"):
        g = grounding.check(body, grounding.source_texts(result["hits"]), question)
        extra["grounding"] = {k: v for k, v in g.items() if v}
    monitoring.log_event(**event, outcome=outcome, n_sources=len(sources),
                         latency_ms=round((time.perf_counter() - started) * 1000), **extra)

    # A "service busy, retry in a minute" reply is not an exchange: kept out of the history, or
    # the model would read it as its own previous answer on the next turn, and not rateable.
    if outcome == "busy":
        return ChatResponse(answer=body, abstained=False)
    history.append((question, result["answer"], result["abstained"]))
    generate.save_history(key, history)
    return ChatResponse(answer=body, abstained=result["abstained"], sources=sources, options=options,
                        turn=len(history) - 1)


class Feedback(BaseModel):
    conversation_id: str
    turn: int
    value: str  # "up" | "down"


@app.post("/feedback")
def feedback(fb: Feedback, username: str = Depends(require_user)):
    if fb.value not in ("up", "down"):
        raise HTTPException(400, "value must be 'up' or 'down'")
    key = _conversation_key(username, fb.conversation_id)
    if not generate._history_path(key).exists():
        raise HTTPException(404, "conversation introuvable")
    history = generate.load_history(key)
    if not 0 <= fb.turn < len(history):
        raise HTTPException(404, "message introuvable")
    monitoring.log_event(event="feedback", user=monitoring.user_hash(username), conversation_id=fb.conversation_id,
                         turn=fb.turn, value=fb.value, question=history[fb.turn][0])
    return {"status": "ok"}


@app.get("/metrics")
def metrics(username: str = Depends(require_user)):
    return monitoring.compute_metrics(monitoring.read_events())


# Mounted last: StaticFiles with html=True serves static/index.html at "/" and would otherwise
# shadow routes declared after it.
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
