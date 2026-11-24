#!/usr/bin/env python3
"""Register Khotwa's system prompt (generate.SYSTEM_PROMPT) in the MLflow prompt registry.

    python scripts/register_prompt.py -m "Add rule 8: table formatting"            # new version only
    python scripts/register_prompt.py -m "Add rule 8: table formatting" --promote  # + move @production to it
    python scripts/register_prompt.py --promote-version 3                          # roll back/forward
    python scripts/register_prompt.py --list

Needs MLFLOW_TRACKING_URI (same server as the rest of the project, DagsHub by default -- see
scripts/tracking.py for credentials). If that server has no prompt registry, registration fails
with its error and the app simply keeps answering with SYSTEM_PROMPT from generate.py.

Workflow:
  1. edit SYSTEM_PROMPT in generate.py
  2. register it (without --promote): it becomes a new version, NOT yet used by the web app
  3. evaluate it before it goes live: KHOTWA_PROMPT_ALIAS=latest python scripts/eval_generation.py
     (the run records prompt_version, so runs on different versions compare side by side)
  4. if it's better: --promote-version N   (the running server picks it up within a minute;
     traces are tagged prompt_version=vN, so you can compare live behaviour across versions)

A version identical to the latest one is never registered twice.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import tracking  # noqa: E402


def _all_versions(mlflow) -> list:
    """Every registered version. search_prompt_versions returns a plain list or an object with
    .prompt_versions depending on the MLflow version, and some servers don't implement it at
    all -- then load the versions one by one until the first missing one."""
    try:
        res = mlflow.MlflowClient().search_prompt_versions(tracking.PROMPT_NAME)
        found = list(getattr(res, "prompt_versions", res) or [])
        if found:
            return found
    except Exception:
        pass
    out, n = [], 1
    while True:
        try:
            pv = mlflow.genai.load_prompt(tracking.PROMPT_NAME, version=n, allow_missing=True)
        except Exception:
            pv = None
        if pv is None:
            return out
        out.append(pv)
        n += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--message", help="what changed in this version (commit message)")
    ap.add_argument("--promote", action="store_true", help="point the @production alias at the registered version")
    ap.add_argument("--alias", default="production", help="alias to move with --promote / --promote-version")
    ap.add_argument("--promote-version", type=int, help="only move the alias to an existing version")
    ap.add_argument("--list", action="store_true", help="list registered versions")
    a = ap.parse_args()

    if not tracking.enabled():
        sys.exit("MLFLOW_TRACKING_URI is not set (in .env) -- the prompt registry lives on the MLflow server.")
    mlflow = tracking.start("prompts")

    if a.list:
        versions = _all_versions(mlflow)
        prod = None
        try:
            prod = mlflow.genai.load_prompt(f"prompts:/{tracking.PROMPT_NAME}@{a.alias}", allow_missing=True)
        except Exception:
            pass
        for pv in sorted(versions, key=lambda v: int(v.version)):
            mark = f"  <- @{a.alias}" if prod is not None and prod.version == pv.version else ""
            print(f"v{pv.version}  {pv.commit_message or ''}{mark}")
        return

    if a.promote_version:
        mlflow.genai.set_prompt_alias(tracking.PROMPT_NAME, alias=a.alias, version=a.promote_version)
        print(f"@{a.alias} -> v{a.promote_version}")
        return

    if not a.message:
        ap.error("-m/--message is required to register a version")
    import generate  # imported late: loads only what's needed to read SYSTEM_PROMPT
    label = tracking.register_prompt(generate.SYSTEM_PROMPT, a.message, alias=a.alias if a.promote else None)
    print(label + ("" if a.promote else f"  (not promoted -- run with --promote-version {label[1:]} when ready)"))


if __name__ == "__main__":
    main()
