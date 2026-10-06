"""Sync state: what was fetched, what changed, and the change log.

data/state/sync_state.json   current state, rewritten at each run
data/state/changelog.jsonl   append-only history of every change detected
"""
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(markdown: str) -> str:
    """Hash of the extracted content only, so menus, session ids or news boxes
    changing around the procedure do not count as a change."""
    norm = re.sub(r"\s+", " ", markdown).strip()
    return "sha256:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()


class SyncState:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "sync_state.json"
        self.changelog = self.dir / "changelog.jsonl"
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.urls: dict = data.get("urls", {})     # url -> {etag, last_modified, raw_path, render, ...}
        self.pages: dict = data.get("pages", {})   # "doc_id:lang" -> {hash, status, path, meta, ...}
        self._lock = threading.Lock()              # sites are crawled in parallel threads

    def log_change(self, key: str, event: str, **details):
        entry = {"time": now_iso(), "doc": key, "event": event, **details}
        with self._lock, self.changelog.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save(self):
        data = {"updated_at": now_iso(), "urls": self.urls, "pages": self.pages}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)   # atomic: a crash never leaves a half-written state
