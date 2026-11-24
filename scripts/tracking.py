"""Optional experiment tracking on the MLflow server DagsHub hosts for this repository.

Off unless MLFLOW_TRACKING_URI is set (in .env or the CI environment), so the scripts run the same
without network access:

  MLFLOW_TRACKING_URI=https://dagshub.com/shaimabouslah/khotwa-..mlflow

Credentials: MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD when set (CI), otherwise the
DagsHub user and token already stored for DVC in .dvc/config.local -- the token is not copied
into a second file.

Also (all no-ops while tracking is off, and fail-safe: an MLflow error never breaks an answer):
  - the web app's traces (setup_app): one trace per /chat question -- retrieval, the Groq call,
    outcome, grounding, prompt and data version -- in the experiment "production", with the
    users' thumbs up/down attached as "user_feedback" assessments (log_user_feedback)
  - the prompt registry (system_prompt / register_prompt, scripts/register_prompt.py): the system
    prompt the app answers with is the version behind the alias KHOTWA_PROMPT_ALIAS ("production"
    by default), falling back to generate.SYSTEM_PROMPT
  - data_version(): git commit + md5 of the chunks + what dvc.lock expects + embeddings fingerprint,
    as tags for any run or trace
"""
import functools
import hashlib
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# MLFLOW_TRACKING_URI usually lives in .env: load it here so every script that imports this module
# (benchmark_retrieval, eval_generation, register_prompt...) sees it, not only the web app. Values
# already in the environment (CI) win; nothing happens without python-dotenv or a .env file.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def _dvc_credentials() -> tuple[str, str] | None:
    path = ROOT / ".dvc" / "config.local"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    section = re.search(r"\['remote \"dagshub\"'\](.*?)(?=^\[|\Z)", text, re.S | re.M)
    if not section:
        return None
    user = re.search(r"^\s*user\s*=\s*(\S+)", section.group(1), re.M)
    password = re.search(r"^\s*password\s*=\s*(\S+)", section.group(1), re.M)
    return (user.group(1), password.group(1)) if user and password else None


def metric_name(*parts: str) -> str:
    """MLflow keys allow letters, digits, _ - . / and spaces: hit@1 -> hit_at_1."""
    return re.sub(r"[^\w\-./ ]", "_", "/".join(parts).replace("@", "_at_"))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def start(experiment: str):
    """Configure MLflow for `experiment` and return the mlflow module (call only when enabled())."""
    for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        if key in os.environ:
            os.environ[key] = os.environ[key].strip()
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        creds = _dvc_credentials()
        if creds:
            os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"] = creds
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"].strip())
    mlflow.set_experiment(experiment)
    global _mlflow
    _mlflow = mlflow  # from here on, the span/prompt helpers below are live
    return mlflow


def log_metrics_tree(mlflow, tree: dict, prefix: str = "") -> None:
    """Log every number of a nested metrics dict ({"dense": {"all": {"hit@1": ...}}})."""
    flat = {}

    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, path + [k])
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[metric_name(*path, k)] = float(v)

    walk(tree, [prefix] if prefix else [])
    mlflow.log_metrics(flat)


# ================================================================== tracing helpers (app + eval)
# Live only once start() has run in this process (scripts) or setup_app() succeeded (web app);
# otherwise every helper below is a no-op, so importing this module never costs anything.
_mlflow = None
_app_lock = threading.Lock()
_app_state = None  # None = not tried yet, True/False = result


def _log(msg: str):
    print(f"[mlflow] {msg}", flush=True)


def setup_app(experiment: str = "production") -> bool:
    """Called once by the web app at startup. Returns True when its traces are being recorded."""
    global _app_state
    if _app_state is not None:
        return _app_state
    with _app_lock:
        if _app_state is not None:
            return _app_state
        if not enabled() or os.environ.get("KHOTWA_MLFLOW") == "0":
            _app_state = False
            return False
        try:
            mlflow = start(os.environ.get("KHOTWA_MLFLOW_EXPERIMENT", experiment))
            try:
                mlflow.groq.autolog()  # each Groq call becomes a CHAT_MODEL span (prompt, reply, tokens)
            except Exception as e:
                _log(f"groq autolog unavailable ({type(e).__name__}): traces without the LLM span")
            _app_state = True
            _log(f"web app traces -> {os.environ['MLFLOW_TRACKING_URI'].strip()} "
                 f"(experiment {os.environ.get('KHOTWA_MLFLOW_EXPERIMENT', experiment)})")
        except Exception as e:
            _log(f"tracing off: {type(e).__name__}: {e}")
            _app_state = False
        return _app_state


def active() -> bool:
    return _mlflow is not None


class _NoSpan:
    trace_id = None

    def set_inputs(self, *_a, **_k): pass
    def set_outputs(self, *_a, **_k): pass
    def set_attributes(self, *_a, **_k): pass
    def set_attribute(self, *_a, **_k): pass


@contextmanager
def span(name: str, span_type: str = "UNKNOWN", inputs: dict | None = None):
    """mlflow.start_span when tracking is live, a stand-in otherwise. MLflow's own errors are
    swallowed; errors raised by the traced code propagate unchanged (and mark the span as failed)."""
    if _mlflow is None:
        yield _NoSpan()
        return
    try:
        cm = _mlflow.start_span(name=name, span_type=span_type)
        s = cm.__enter__()
    except Exception as e:
        _log(f"span '{name}' not recorded ({type(e).__name__})")
        yield _NoSpan()
        return
    try:
        if inputs is not None:
            s.set_inputs(inputs)
    except Exception:
        pass
    try:
        yield s
    except BaseException as e:
        try:
            cm.__exit__(type(e), e, e.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def trace(name: str | None = None, span_type: str = "CHAIN"):
    """Decorator: records the call's arguments and return value as a span (when live)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _mlflow is None:
                return fn(*args, **kwargs)
            return _mlflow.trace(fn, name=name or fn.__name__, span_type=span_type)(*args, **kwargs)
        return wrapper
    return deco


def retrieval_span(fn):
    """For generate.retrieve(): a RETRIEVER span whose output is the retrieved chunks as MLflow
    documents (text + doc_id/title/score/url), so a trace shows exactly what the LLM read."""
    @functools.wraps(fn)
    def wrapper(retriever, query, *args, **kwargs):
        if _mlflow is None:
            return fn(retriever, query, *args, **kwargs)
        simple = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None)))}
        with span("retrieve", "RETRIEVER", {"query": query, **simple}) as s:
            hits = fn(retriever, query, *args, **kwargs)
            try:
                s.set_outputs([{"id": h.chunk_id, "page_content": h.text,
                                "metadata": {"doc_id": h.doc_id, "title": h.title, "org": h.org, "lang": h.lang,
                                             "domain": h.domain, "subtopic": h.subtopic,
                                             "score": round(float(h.score), 4), "url": h.url}} for h in hits])
                s.set_attributes({"n_hits": len(hits),
                                  "top_score": round(float(hits[0].score), 4) if hits else None})
            except Exception:
                pass
            return hits
    return wrapper


def tag_current_trace(tags: dict | None = None, session_id: str | None = None, user: str | None = None):
    if _mlflow is None:
        return
    clean = {k: str(v) for k, v in (tags or {}).items() if v is not None}
    try:
        kwargs = {"tags": clean}
        if session_id:
            kwargs["session_id"] = session_id
        if user:
            kwargs["user"] = user
        _mlflow.update_current_trace(**kwargs)
    except TypeError:  # an MLflow without session_id/user: keep them as tags
        try:
            _mlflow.update_current_trace(tags={**clean, "session": session_id or "", "user": user or ""})
        except Exception:
            pass
    except Exception:
        pass


def log_user_feedback(trace_id: str | None, value: str, user: str, question: str | None = None) -> bool:
    """Attach a thumbs up/down from the web UI to the trace of the answer it rates. Traces are
    exported asynchronously, so a very fast click can arrive before its trace exists: on failure,
    retry for a few seconds in a background thread (the HTTP request never waits)."""
    if not trace_id or _mlflow is None:
        return False

    def send():
        from mlflow.entities import AssessmentSource, AssessmentSourceType
        _mlflow.log_feedback(trace_id=trace_id, name="user_feedback", value=(value == "up"),
                             source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=user),
                             rationale="thumbs up" if value == "up" else "thumbs down",
                             metadata={"question": (question or "")[:300]})

    try:
        send()
        return True
    except Exception as first:
        def retry():
            for _ in range(6):
                time.sleep(2)
                try:
                    send()
                    return
                except Exception:
                    continue
            _log(f"feedback not recorded for {trace_id} ({type(first).__name__}: {first})")
        threading.Thread(target=retry, daemon=True).start()
        return False


# ================================================================================ data version
@functools.lru_cache(maxsize=4)
def data_version(rag_dir: str | None = None) -> dict:
    """What a run/trace was produced with, as tags:
    git_commit, git_dirty            the code
    chunks_md5                       the chunks actually used (<rag_dir>/chunks.jsonl)
    chunks_md5_dvc_lock              what dvc.lock says data/rag/chunks.jsonl should be
    chunks_in_sync                   false = this run did NOT use the committed data version
    embeddings_sig                   start of the embeddings fingerprint (chunks.<model>.sig)"""
    rag = Path(rag_dir) if rag_dir else ROOT / "data" / "rag"
    tags = {"git_commit": git_commit()}
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "scripts", "kb"],
                               cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        tags["git_dirty"] = str(bool(dirty)).lower()
    except Exception:
        pass
    chunks = rag / "chunks.jsonl"
    if chunks.exists():
        h = hashlib.md5()
        with open(chunks, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        tags["chunks_md5"] = h.hexdigest()
    lock = ROOT / "dvc.lock"
    if lock.exists():
        m = re.search(r"path: data/rag/chunks\.jsonl\s+hash: md5\s+md5: (\w+)", lock.read_text(encoding="utf-8"))
        if m:
            tags["chunks_md5_dvc_lock"] = m.group(1)
            if "chunks_md5" in tags:
                tags["chunks_in_sync"] = str(tags["chunks_md5"] == m.group(1)).lower()
    sigs = sorted(rag.glob("chunks.*.sig"))
    if sigs:
        tags["embeddings_sig"] = sigs[0].read_text(encoding="utf-8").strip()[:16]
    return tags


# ============================================================================= prompt registry
PROMPT_NAME = "khotwa-system-prompt"
_PROMPT_TTL = 60.0  # seconds: an alias moved on the server reaches the running app within a minute
_prompt_cache: dict[str, tuple[float, str, str]] = {}


def system_prompt(default: str) -> tuple[str, str]:
    """(text, version label) of the system prompt to answer with: the registered version behind
    KHOTWA_PROMPT_ALIAS ("production") when tracking is live and one exists, else `default`
    labelled "code". A missing registry, an unreachable server or a fresh clone all keep working."""
    if _mlflow is None:
        return default, "code"
    alias = os.environ.get("KHOTWA_PROMPT_ALIAS", "production")
    now = time.monotonic()
    hit = _prompt_cache.get(alias)
    if hit and now - hit[0] < _PROMPT_TTL:
        return hit[1], hit[2]
    try:
        pv = _mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{alias}", allow_missing=True)
        text, label = (pv.template, f"v{pv.version}") if pv is not None else (default, "code")
    except Exception as e:
        _log(f"prompt registry unavailable, using the prompt in the code ({type(e).__name__})")
        text, label = default, "code"
    _prompt_cache[alias] = (now, text, label)
    return text, label


def register_prompt(template: str, message: str, alias: str | None = None) -> str:
    """Register `template` as a new version unless it equals the latest one; optionally point
    `alias` at it. Returns "vN". Needs start() first (scripts/register_prompt.py does it)."""
    if _mlflow is None:
        raise RuntimeError("tracking is off: set MLFLOW_TRACKING_URI")
    try:
        latest = _mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@latest", allow_missing=True)
    except Exception:
        latest = None
    if latest is not None and latest.template == template:
        version = int(latest.version)
        _log(f"unchanged -- {PROMPT_NAME} stays at v{version}")
    else:
        pv = _mlflow.genai.register_prompt(
            name=PROMPT_NAME, template=template, commit_message=message,
            tags={"git_commit": git_commit(), "sha1": hashlib.sha1(template.encode("utf-8")).hexdigest()[:12]})
        version = int(pv.version)
        _log(f"registered {PROMPT_NAME} v{version}")
    if alias:
        _mlflow.genai.set_prompt_alias(PROMPT_NAME, alias=alias, version=version)
        _log(f"@{alias} -> v{version}")
    _prompt_cache.clear()
    return f"v{version}"