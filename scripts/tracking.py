"""Optional experiment tracking on the MLflow server DagsHub hosts for this repository.

Off unless MLFLOW_TRACKING_URI is set (in .env or the CI environment), so the scripts run the same
without network access:

  MLFLOW_TRACKING_URI=https://dagshub.com/shaimabouslah/khotwa-..mlflow

Credentials: MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD when set (CI), otherwise the
DagsHub user and token already stored for DVC in .dvc/config.local -- the token is not copied
into a second file.
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
