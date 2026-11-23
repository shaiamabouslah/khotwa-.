#!/usr/bin/env python3
"""Readable titles for CNSS pages, in the page's own language.

The CNSS site gives no useful <title>, so fixes/fix_titles.py fell back to the last breadcrumb
label -- which on that site is an internal CMS code: "Emp Secteur pêche3 Immatriculation
(Employeur)". 122 of 311 KB documents ended up titled like that, and users saw it on the source
buttons (and it is embedded in every chunk's text). This turns the code into
"<leading heading or procedure> — <role>, <audience>", e.g. "Affiliation — employeur, secteur de
la pêche" / "الانخراط — المؤجر، قطاع الصيد البحري". The " — <role>" suffix is what
cnss_rules.CNSS_ROLES looks for to tell employer / employee / self-employed pages apart.

Used by clean_pipeline.populate_kb (so a rebuild keeps readable titles); run directly to migrate
existing kb/ files in place (the code is kept as `original_title`):

  python scripts/cnss_titles.py            # dry run: show what would change
  python scripts/cnss_titles.py --apply
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "kb"

CODE_RE = re.compile(r"^(?P<role>Non Sal|Emp|Sal)\s+(?P<rest>.+?)\s*\([^()]*\)\s*$")

ROLES = {"Emp": {"fr": "employeur", "ar": "المؤجر"},
         "Sal": {"fr": "salarié", "ar": "الأجير"},
         "Non Sal": {"fr": "non-salarié", "ar": "غير الأجير"}}

# Order matters: "non agr" before "agr", "regime complementaire" before "affiliation".
AUDIENCES = [("non agr", {"fr": "secteur non agricole", "ar": "القطاع غير الفلاحي"}),
             ("agr", {"fr": "secteur agricole", "ar": "القطاع الفلاحي"}),
             ("peche", {"fr": "secteur de la pêche", "ar": "قطاع الصيد البحري"}),
             ("gens maison", {"fr": "gens de maison", "ar": "عملة المنازل"}),
             ("chantiers", {"fr": "travailleurs des chantiers", "ar": "عملة الحضائر"}),
             ("etudiants", {"fr": "étudiants", "ar": "الطلبة"}),
             ("artistes", {"fr": "artistes", "ar": "الفنانون"}),
             ("tte", {"fr": "Tunisiens à l'étranger", "ar": "التونسيون بالخارج"})]
PROCEDURES = [("regime complementaire", {"fr": "Affiliation au régime complémentaire", "ar": "الانخراط في النظام التكميلي"}),
              ("affiliation", {"fr": "Affiliation", "ar": "الانخراط"}),
              ("immatriculation", {"fr": "Immatriculation", "ar": "التسجيل"}),
              ("assiette", {"fr": "Assiette des cotisations", "ar": "قاعدة احتساب الاشتراكات"}),
              ("cotisation", {"fr": "Cotisations", "ar": "الاشتراكات"}),
              ("avantage", {"fr": "Avantages", "ar": "الامتيازات"}),
              ("attestation", {"fr": "Attestations", "ar": "الشهادات"}),
              ("remise penalites", {"fr": "Remise des pénalités de retard", "ar": "الإعفاء من خطايا التأخير"}),
              ("licenciement", {"fr": "Procédures en cas de licenciement", "ar": "الإجراءات في حالة الطرد"})]
MAX_HEADING = 80


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if not unicodedata.combining(c))
    return re.sub(r"[\s_-]+", " ", s)


def _leading_heading(body: str) -> str | None:
    first = next((l.strip() for l in body.splitlines() if l.strip()), "")
    m = re.match(r"^#{1,6}\s+(.+)$", first)
    if not m:
        return None
    h = re.sub(r"[*_`]+", "", m.group(1)).strip().rstrip(".:")
    if len(h) > MAX_HEADING:
        h = h[:MAX_HEADING].rsplit(" ", 1)[0] + "…"
    return h if len(h) >= 8 else None


def readable_title(title: str, lang: str, body: str = "") -> str | None:
    """None when `title` isn't a CNSS code title (leave it as it is)."""
    m = CODE_RE.match(title or "")
    if not m:
        return None
    rest = _norm(m.group("rest"))
    lang = lang if lang in ("fr", "ar") else "fr"
    audience = next((a[lang] for key, a in AUDIENCES if key in rest), None)
    procedure = next((p[lang] for key, p in PROCEDURES if key in rest), None)
    if not audience or not procedure:
        return None
    core = _leading_heading(body) or procedure
    sep = "، " if lang == "ar" else ", "
    return f"{core} — {ROLES[m.group('role')][lang]}{sep}{audience}"


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


def migrate(apply: bool) -> int:
    changed = 0
    for path in sorted(KB.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        if "original_title" in fm:
            continue
        new = readable_title(fm.get("title", ""), fm.get("lang", "fr"), text[m.end():])
        if not new:
            continue
        changed += 1
        print(f"{path.relative_to(ROOT)}\n    {fm['title']!r}\n -> {new!r}")
        if apply:
            def dump(key, value):
                return yaml.safe_dump({key: value}, allow_unicode=True, width=1000).strip()
            lines = m.group(1).split("\n")
            i = next(j for j, l in enumerate(lines) if l.startswith("title:"))
            lines[i:i + 1] = [dump("title", new), dump("original_title", fm["title"])]
            # newline="\n": plain write_text turns every "\n" into "\r\n" on Windows (see .gitattributes)
            path.write_text("---\n" + "\n".join(lines) + "\n---\n" + text[m.end():], encoding="utf-8", newline="\n")
    return changed


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    n = migrate(ap.parse_args().apply)
    print(f"\n{n} title(s) {'rewritten' if '--apply' in sys.argv else 'would change (dry run)'}")
