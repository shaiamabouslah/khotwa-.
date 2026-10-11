"""LLM-based domain/subtopic classifier using a local Ollama model (qwen2.5:7b by default).

Replaces the brittle keyword-matching Classifier in crawler/classify.py, whose scores can
collide across subtopics when a document mentions terms from more than one (e.g. "immatriculation"
matching both CNSS affiliation and RNE business registration -- found and patched by hand once,
but that class of bug doesn't scale as the taxonomy grows).

Two modes:
  --sample N      classify N docs (a mix of known-tricky + random) and print old vs new side by
                   side, WITHOUT writing anything -- use this to judge quality before a full run.
  --apply         classify the whole corpus and patch front matter + sync_state.json + catalog.jsonl,
                   same pattern as scripts/fix_subtopic_labels.py.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/llm_classify.py --sample 20
    PYTHONPATH=. .venv/Scripts/python.exe scripts/llm_classify.py --apply
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
STATE_PATH = ROOT / "data" / "state" / "sync_state.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def build_taxonomy_prompt(taxonomy: dict) -> str:
    lines = []
    for domain, d in taxonomy["domains"].items():
        for sub, s in d["subtopics"].items():
            lines.append(f'- domain="{domain}" subtopic="{sub}": {s["fr"]} / {s["ar"]}')
    return "\n".join(lines)


def valid_pairs(taxonomy: dict) -> set:
    return {(domain, sub) for domain, d in taxonomy["domains"].items() for sub in d["subtopics"]}


def classify_one(title: str, breadcrumb: list, body: str, taxonomy_prompt: str, valid: set) -> dict:
    text = " ".join(body.split()[:400])
    prompt = f"""Tu classes un document administratif tunisien (français ou arabe) dans une taxonomie fixe.

Catégories possibles (domain/subtopic) :
{taxonomy_prompt}

Si le document ne correspond clairement à AUCUNE de ces catégories (ex: annuaire d'adresses,
menu de navigation, page d'accueil générique, actualité sans procédure), réponds avec domain
et subtopic à null -- n'invente jamais une catégorie approximative.

Titre: {title}
Fil d'Ariane: {' > '.join(breadcrumb) if breadcrumb else '(aucun)'}
Contenu (début): {text[:2500]}

Réponds UNIQUEMENT en JSON, sur une seule ligne, format exact:
{{"domain": "studies"|"employment"|null, "subtopic": "<clé exacte ci-dessus>"|null, "confidence": "high"|"medium"|"low"}}"""

    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    try:
        parsed = json.loads(out["response"])
    except (json.JSONDecodeError, KeyError):
        return {"domain": None, "subtopic": None, "confidence": "low", "_raw": out.get("response")}
    # The model sometimes invents a subtopic key instead of picking one verbatim from the
    # list (seen: "remise-penalites", "bourses" under employment instead of studies, typos
    # like "cotations") -- reject anything not in the real taxonomy rather than let a
    # hallucinated label into the corpus. domain alone (subtopic=null) is still accepted.
    pair = (parsed.get("domain"), parsed.get("subtopic"))
    if parsed.get("subtopic") is not None and pair not in valid:
        return {"domain": None, "subtopic": None, "confidence": "low",
                "_rejected": f"{pair} not in taxonomy"}
    return parsed


def iter_corpus_docs():
    for md_path in CORPUS.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        meta = yaml.safe_load(front)
        yield md_path, meta, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="classify N docs, print only, write nothing")
    ap.add_argument("--apply", action="store_true", help="classify the whole corpus and patch it")
    ap.add_argument("--force", action="store_true", help="with --apply, re-classify even docs already done")
    ap.add_argument("--seed-tricky", action="store_true", default=True)
    args = ap.parse_args()

    taxonomy = yaml.safe_load((ROOT / "kb" / "taxonomy.yaml").read_text(encoding="utf-8"))
    taxonomy_prompt = build_taxonomy_prompt(taxonomy)
    valid = valid_pairs(taxonomy)

    docs = list(iter_corpus_docs())

    if args.sample:
        tricky_markers = ["equivalence_rnu", "rne_app", "apii_portal", "cnss/fr/notre_reseau",
                           "annuaire_administration"]
        tricky = [d for d in docs if any(m in str(d[0]) for m in tricky_markers)]
        random.seed(0)
        rest = [d for d in docs if d not in tricky]
        sample = (tricky[: args.sample // 2] + random.sample(rest, min(len(rest), args.sample - len(tricky[: args.sample // 2]))))
        t0 = time.time()
        for path, meta, body in sample:
            rel = path.relative_to(ROOT).as_posix()
            old = (meta.get("domain_hint"), meta.get("subtopic_hint"))
            new = classify_one(meta.get("title", ""), meta.get("breadcrumb") or [], body, taxonomy_prompt, valid)
            flag = "  <-- CHANGED" if (new.get("domain"), new.get("subtopic")) != old else ""
            print(f"{rel}")
            print(f"  old (rules): {old}")
            print(f"  new (llm)  : ({new.get('domain')}, {new.get('subtopic')}) conf={new.get('confidence')}{flag}")
        print(f"\n{len(sample)} docs classified in {time.time()-t0:.1f}s ({(time.time()-t0)/max(len(sample),1):.1f}s/doc)")
        return

    if args.apply:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        path_to_key = {p["path"]: k for k, p in state["pages"].items()}

        def sync_catalog_and_state():
            STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            with (CORPUS / "catalog.jsonl").open("w", encoding="utf-8") as f:
                for key, page in sorted(state["pages"].items()):
                    row = {**page.get("meta", {}), "path": page["path"], "status": page["status"],
                           "last_checked": page.get("last_checked")}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        changed = skipped = 0
        t0 = time.time()
        for i, (path, meta, body) in enumerate(docs):
            rel = path.relative_to(ROOT).as_posix()
            key = path_to_key.get(rel)
            # Resumable: a doc already LLM-classified in a prior (possibly interrupted) run
            # is skipped -- its current front-matter values are just re-synced to state/catalog,
            # no new Ollama call. --force re-classifies everything regardless.
            if meta.get("hint_source") == "llm" and not args.force:
                skipped += 1
            else:
                old = (meta.get("domain_hint"), meta.get("subtopic_hint"))
                new = classify_one(meta.get("title", ""), meta.get("breadcrumb") or [], body, taxonomy_prompt, valid)
                new_pair = (new.get("domain"), new.get("subtopic"))
                meta["domain_hint"] = new.get("domain")
                meta["subtopic_hint"] = new.get("subtopic")
                meta["hint_score"] = {"high": 10, "medium": 5, "low": 1}.get(new.get("confidence"), 0)
                meta["hint_keywords"] = []
                meta["hint_source"] = "llm"
                new_front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()
                path.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")
                if new_pair != old:
                    changed += 1
            if key and "meta" in state["pages"][key]:
                state["pages"][key]["meta"]["domain_hint"] = meta.get("domain_hint")
                state["pages"][key]["meta"]["subtopic_hint"] = meta.get("subtopic_hint")
                state["pages"][key]["meta"]["hint_score"] = meta.get("hint_score")
                state["pages"][key]["meta"]["hint_keywords"] = []
                state["pages"][key]["meta"]["hint_source"] = meta.get("hint_source")
            if (i + 1) % 25 == 0:
                sync_catalog_and_state()   # checkpoint: survives a kill between here and the end
                elapsed = time.time() - t0
                done = i + 1 - skipped
                rate = elapsed / done if done else 0
                print(f"  {i+1}/{len(docs)} ({skipped} skipped, resumed), "
                      f"~{rate*(len(docs)-i-1):.0f}s remaining")

        sync_catalog_and_state()
        print(f"\n{changed} changed, {skipped} skipped (already done), "
              f"{len(docs)-skipped} newly classified in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
