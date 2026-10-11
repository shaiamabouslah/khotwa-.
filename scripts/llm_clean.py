"""LLM-based content cleaning pass (local Qwen2.5 via Ollama) -- the heavier sibling of
llm_classify.py. Where the rule-based extractor (crawler/extract.py) fights each site's HTML
quirks one CSS selector at a time, this asks the model to do what a human would: read the raw
extracted markdown and keep only the real, useful content.

Scope: only documents flagged by scripts/audit_noise.py (table_noise / link_only / login_wall)
-- running this on the whole corpus would waste hours re-cleaning pages that are already fine
(see session notes: classification alone took ~4.6h over 826 docs; this is a heavier rewrite
task, slower per doc).

For each flagged doc the model returns:
  title           a short, specific, content-reflecting title (not the generic site-wide title
                   every aneti page currently shares)
  content         the cleaned markdown: real prose/lists/tables only, nav/footer/cookie banner/
                   breadcrumb/"back to top"/social-share noise stripped out
  disposition     "keep" | "login_required" | "no_real_content"
                   -- the latter two mean: don't rewrite the body, just mark status=removed
                   (same convention as scripts/mark_nav_stubs.py)

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/llm_clean.py --sample 10   # print only
    PYTHONPATH=. .venv/Scripts/python.exe scripts/llm_clean.py --apply       # patch files
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_noise import score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
STATUS_RE = re.compile(r"^status: \w+$", re.M)
TITLE_RE = re.compile(r"^title: .*$", re.M)


def clean_one(raw_markdown: str) -> dict:
    prompt = f"""Tu nettoies le contenu brut d'une page web administrative tunisienne (extraite automatiquement, en français ou arabe).

Le texte brut ci-dessous contient le vrai contenu MÉLANGÉ avec du bruit : menu de navigation
répété, pied de page, bannière cookies, fil d'Ariane, boutons de partage, widgets de recherche,
tableaux de mise en page vides. Ta tâche : extraire UNIQUEMENT le contenu réel et utile.

Règles :
- Supprime entièrement : menus de navigation principaux, pieds de page, bannières cookies/RGPD,
  liens "retour en haut"/"imprimer"/"plan du site", widgets de partage social, barres de
  recherche, fil d'Ariane (breadcrumb).
- Sois intelligent sur les liens, ne les supprime JAMAIS aveuglément : un lien fait partie du
  CONTENU (donc à garder) s'il mène vers un formulaire à remplir, un document/PDF à télécharger,
  une étape concrète d'une procédure ("déposez votre demande ici", "téléchargez le formulaire"),
  ou toute ressource que l'utilisateur doit utiliser pour accomplir la démarche décrite par la
  page. Un lien fait partie de la NAVIGATION (donc à supprimer) seulement s'il sert juste à se
  déplacer ailleurs sur le site (menu, "voir aussi", liens vers d'autres pages non liées à la
  procédure en cours). Dans le doute, garde le lien.
- Garde : tout texte informatif réel, listes de conditions/documents/étapes, tableaux de
  données réels (prix, délais, adresses), liens qui font partie du contenu (pas de la nav).
- Si la page ENTIÈRE nécessite une connexion/authentification pour voir le vrai contenu
  (ex: "veuillez vous connecter pour continuer" est le seul contenu), disposition="login_required".
- Si après nettoyage il ne reste aucun contenu réel utilisable (juste des liens, un menu vide),
  disposition="no_real_content".
- Sinon disposition="keep", et donne un titre court et SPÉCIFIQUE à ce que fait cette page
  précise (pas un titre générique du site).
- Ne traduis jamais, ne résume jamais le contenu réel -- garde le texte original tel quel,
  juste débarrassé du bruit.

TEXTE BRUT :
{raw_markdown[:6000]}

Réponds UNIQUEMENT en JSON sur une seule ligne:
{{"disposition": "keep"|"login_required"|"no_real_content", "title": "...", "content": "... (markdown nettoyé, vide si disposition != keep)"}}"""

    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    out = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read())
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                return {"disposition": "keep", "title": None, "content": None, "_error": str(e)}
            time.sleep(5 * (attempt + 1))
    try:
        result = json.loads(out["response"])
    except (json.JSONDecodeError, KeyError):
        return {"disposition": "keep", "title": None, "content": None, "_raw": out.get("response")}

    # Caught once in practice: the model silently translated an Arabic contact-info page into
    # Chinese, against an explicit "never translate" instruction. CJK appearing where the source
    # had none is an unambiguous tell (this corpus is only ever French/Arabic/English) -- revert
    # that document to "keep nothing changed" rather than risk corrupting real content silently.
    cjk = re.compile(r"[一-鿿]")
    if cjk.search((result.get("content") or "") + (result.get("title") or "")) and not cjk.search(raw_markdown):
        return {"disposition": "keep", "title": None, "content": None,
                "_rejected": "CJK output from non-CJK input"}
    return result


def all_docs(threshold: float, force: bool):
    """Every active doc, lowest-noise-first isn't needed -- but skip ones already cleaned by
    this script's v2 prompt (the `cleaned_v2: true` marker) unless --force, so a run spanning
    many hours survives being killed and resumed (this session's background jobs keep getting
    cut by session boundaries -- resumability isn't optional here)."""
    for p in CORPUS.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        front, body = m.groups()
        if "status: active" not in front:
            continue
        if not force and "cleaned_v2: true" in front:
            continue
        sc = score(body)
        if sc["link_only"] or sc["login_wall"] or sc["table_noise"] > threshold:
            yield p, front, body, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--all", action="store_true",
                     help="scan the whole corpus with a lower noise threshold (0.15 vs 0.4), "
                          "not just the extreme cases -- resumable via cleaned_v2 marker")
    ap.add_argument("--force", action="store_true", help="with --all, re-clean docs already marked cleaned_v2")
    args = ap.parse_args()

    threshold = 0.15 if args.all else 0.4
    docs = list(all_docs(threshold, args.force))
    print(f"{len(docs)} doc(s) flagged for cleaning (threshold={threshold})")
    if args.sample:
        docs = docs[: args.sample]

    def mark_done(fm: str) -> str:
        return fm if "cleaned_v2:" in fm else fm + "\ncleaned_v2: true"

    t0 = time.time()
    changed = removed = unchanged = 0
    for i, (path, front, body, sc) in enumerate(docs):
        rel = path.relative_to(ROOT).as_posix()
        result = clean_one(body)
        disp = result.get("disposition", "keep")
        print(f"\n{rel}  (flags: noise={sc['table_noise']:.2f} link_only={sc['link_only']} "
              f"login={sc['login_wall']})")
        print(f"  -> {disp}" + (f": {result.get('title')}" if disp == "keep" else "")
              + (f"  [{result['_error']}]" if result.get("_error") else "")
              + (f"  [{result['_rejected']}]" if result.get("_rejected") else ""))

        if not args.apply:
            if disp == "keep" and result.get("content"):
                preview = result["content"][:300].replace("\n", " ")
                print(f"  preview: {preview}...")
            continue

        if disp in ("login_required", "no_real_content"):
            new_front = STATUS_RE.sub("status: removed", front, count=1)
            new_front = mark_done(new_front)
            path.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")
            removed += 1
        elif disp == "keep" and result.get("content"):
            new_front = front
            if result.get("title"):
                title_line = f'title: {result["title"]}'
                new_front = (TITLE_RE.sub(title_line, new_front, count=1)
                             if TITLE_RE.search(new_front) else new_front + f"\n{title_line}")
            new_front = mark_done(new_front)
            path.write_text(f"---\n{new_front}\n---\n\n{result['content'].strip()}\n", encoding="utf-8")
            changed += 1
        else:
            # Model failed / was rejected (CJK guard, bad JSON, server error): still mark it
            # done so a resumed --all run doesn't retry the same failing doc forever. The doc
            # keeps its original (uncleaned but uncorrupted) content.
            new_front = mark_done(front)
            path.write_text(f"---\n{new_front}\n---\n{body}", encoding="utf-8")
            unchanged += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"\n  [{i+1}/{len(docs)}, {elapsed:.0f}s elapsed, "
                  f"~{elapsed/(i+1)*(len(docs)-i-1):.0f}s remaining]")

    if args.apply:
        print(f"\n{changed} cleaned, {removed} marked removed, {unchanged} failed/rejected, "
              f"{len(docs)} processed in {time.time()-t0:.0f}s")
        print("Run scripts/resync_catalog.py afterwards to refresh catalog.jsonl/sync_state.json.")


if __name__ == "__main__":
    main()
