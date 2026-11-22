"""Usage monitoring for the web app: one JSON line per question and per feedback click in
events.jsonl, and the aggregates served by GET /metrics.

The log stays local (gitignored, kept out of the Docker image; under KHOTWA_STATE_DIR when set, so
it lands in the container's persistent volume). Usernames are stored hashed. Question texts are
kept on purpose: for a RAG assistant the most useful thing monitoring shows is which questions it
could not answer -- that is what tells you which documents the knowledge base is missing.
"""
import datetime as dt
import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = (Path(os.environ["KHOTWA_STATE_DIR"]) / "events.jsonl" if os.environ.get("KHOTWA_STATE_DIR")
               else ROOT / "data" / "monitoring" / "events.jsonl")

# Outcomes of a question, as logged by api.chat():
#   answer         grounded LLM answer that cites at least one source
#   uncited        LLM answer citing nothing -- in practice "I didn't find this in the sources"
#   no_match       nothing relevant retrieved, answered "I don't know" without calling the LLM
#   clarification  asked for the CNSS sector (with clickable options)
#   chitchat       greeting / thanks, canned reply
#   busy           Groq rate limit reached, "try again in a minute"
#   error          the request failed (missing API key, crash...)
UNANSWERED = ("no_match", "uncited")

_lock = threading.Lock()


def user_hash(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:12]


def log_event(**fields):
    record = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **fields}
    with _lock:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_events() -> list[dict]:
    if not EVENTS_PATH.exists():
        return []
    events = []
    with EVENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut by a crash mid-write shouldn't take the whole page down
    return events


def _percentile(values: list[float], p: float):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def compute_metrics(events: list[dict]) -> dict:
    questions = [e for e in events if e.get("event") == "question"]
    outcomes = Counter(e.get("outcome") for e in questions)
    substantive = sum(n for o, n in outcomes.items() if o not in ("chitchat",))
    latency = [e["latency_ms"] for e in questions if e.get("outcome") == "answer" and "latency_ms" in e]

    # A user can change their mind: only their latest vote on a given answer counts.
    votes = {}
    for e in events:
        if e.get("event") == "feedback":
            votes[(e.get("user"), e.get("conversation_id"), e.get("turn"))] = e
    up = sum(1 for e in votes.values() if e.get("value") == "up")
    down = sum(1 for e in votes.values() if e.get("value") == "down")
    downvoted = sorted((e for e in votes.values() if e.get("value") == "down"), key=lambda e: e["ts"], reverse=True)

    def rate(n):
        return round(n / substantive, 3) if substantive else None

    return {
        "questions": len(questions),
        "by_outcome": dict(outcomes),
        "by_language": dict(Counter(e.get("lang") for e in questions)),
        "answer_rate": rate(outcomes["answer"]),
        "unanswered_rate": rate(sum(outcomes[o] for o in UNANSWERED)),
        "error_rate": rate(outcomes["error"] + outcomes["busy"]),
        "latency_ms": {"p50": _percentile(latency, 50), "p95": _percentile(latency, 95),
                       "count": len(latency)},
        "feedback": {"up": up, "down": down,
                     "satisfaction": round(up / (up + down), 3) if up + down else None},
        "recent_unanswered": [{"ts": e["ts"], "lang": e.get("lang"), "question": e.get("question")}
                              for e in reversed(questions) if e.get("outcome") in UNANSWERED][:15],
        "recent_downvoted": [{"ts": e["ts"], "question": e.get("question")} for e in downvoted][:10],
    }
