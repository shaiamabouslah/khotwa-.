"""Check every KB procedure file against kb/taxonomy.yaml.

Usage: python scripts/validate_kb.py
Exit code 1 if any error is found (warnings do not fail).
"""
import datetime as dt
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "kb"
TAXONOMY = yaml.safe_load((KB / "taxonomy.yaml").read_text(encoding="utf-8"))

REQUIRED = ["id", "title", "domain", "subtopic", "lang", "org", "keywords",
            "sources", "last_checked", "last_changed", "status"]
LIST_FIELDS = ["translations", "audience", "keywords", "related", "sources", "notes"]
DATE_FIELDS = ["last_checked", "last_changed"]
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)


def is_date(value):
    if isinstance(value, dt.date):
        return True
    try:
        dt.date.fromisoformat(str(value))
        return True
    except ValueError:
        return False


def procedure_files():
    for path in sorted(KB.rglob("*.md")):
        rel = path.relative_to(KB)
        if rel.parts[0].startswith("_") or rel.name == "README.md":
            continue
        yield path


def check_file(path, errors, warnings):
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = FRONT_MATTER.match(text)
    if not match:
        errors.append(f"{rel}: missing or malformed YAML front matter")
        return None
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        errors.append(f"{rel}: invalid YAML ({exc.__class__.__name__})")
        return None
    body = match.group(2)

    for field in REQUIRED:
        if meta.get(field) in (None, "", []):
            errors.append(f"{rel}: missing required field '{field}'")
    for field in LIST_FIELDS:
        if field in meta and not isinstance(meta[field], list):
            errors.append(f"{rel}: '{field}' must be a list")
    for field in DATE_FIELDS:
        if field in meta and not is_date(meta[field]):
            errors.append(f"{rel}: '{field}' is not an ISO date (YYYY-MM-DD)")

    domain, subtopic, lang = meta.get("domain"), meta.get("subtopic"), meta.get("lang")
    domains = TAXONOMY["domains"]
    if domain not in domains:
        errors.append(f"{rel}: unknown domain '{domain}'")
    elif subtopic not in domains[domain]["subtopics"]:
        errors.append(f"{rel}: unknown subtopic '{subtopic}' for domain '{domain}'")
    if lang not in TAXONOMY["languages"]:
        errors.append(f"{rel}: unknown lang '{lang}'")
    if meta.get("status") not in TAXONOMY["statuses"]:
        errors.append(f"{rel}: unknown status '{meta.get('status')}'")

    # Location and file name must agree with the metadata.
    expected = KB / str(domain) / str(subtopic) / f"{meta.get('id')}.{lang}.md"
    if path != expected:
        errors.append(f"{rel}: should be at {expected.relative_to(ROOT).as_posix()}")

    for i, src in enumerate(meta.get("sources") or []):
        if not isinstance(src, dict) or not str(src.get("url", "")).startswith("http"):
            errors.append(f"{rel}: sources[{i}] needs an http(s) 'url'")
            continue
        if not src.get("breadcrumb"):
            warnings.append(f"{rel}: sources[{i}] has no breadcrumb")
        if "source_updated" in src and not is_date(src["source_updated"]):
            errors.append(f"{rel}: sources[{i}].source_updated is not an ISO date")

    # Body: real content required. No forced section template -- a doc keeps whatever
    # structure its source actually has (own headings, bold labels, lists); the 8-key list
    # in taxonomy.yaml is advisory only (for the chunker to recognize a natural heading when
    # one happens to match), never enforced.
    if not body.strip():
        errors.append(f"{rel}: body is empty")
    if re.search(r"^# ", body, re.M):
        errors.append(f"{rel}: top-level '# ' headings are not allowed in the body")
    if "<" in body and re.search(r"<[A-Za-z][^>]*>", body):
        warnings.append(f"{rel}: body contains HTML tags")

    return meta


def main():
    errors, warnings = [], []
    metas = {}
    for path in procedure_files():
        meta = check_file(path, errors, warnings)
        if meta and meta.get("id"):
            key = (meta["id"], meta.get("lang"))
            if key in metas:
                errors.append(f"duplicate id/lang {key}: {path} and {metas[key][0]}")
            metas[key] = (path, meta)

    ids = {key[0] for key in metas}
    for (pid, lang), (path, meta) in metas.items():
        rel = path.relative_to(ROOT).as_posix()
        for other in meta.get("translations") or []:
            if (pid, other) not in metas:
                errors.append(f"{rel}: declares translation '{other}' but {pid}.{other}.md is missing")
        for rid in meta.get("related") or []:
            if rid not in ids:
                warnings.append(f"{rel}: related id '{rid}' does not exist yet")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    n_files = len(metas)
    print(f"\n{n_files} procedure file(s), {len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
