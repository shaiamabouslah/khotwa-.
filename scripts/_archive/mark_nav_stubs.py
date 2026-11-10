"""Mark confirmed pure-navigation stub documents as status=removed (not deleted -- same
convention the crawler itself uses for documents that disappear from a site).

Each entry below was read and judged by hand (see scripts/audit_stub_pages.py for how they
were found): every one is either a repeated link list with zero prose, or a "click here to
open a popup" page with no actual information. Short-but-real content (address cards on
annuaire_administration/equivalence_rnu, the khadamet life-event intro paragraph, the dated
best_rnu loan announcement) was deliberately NOT included -- word count alone over-flags those.

cnss/erasmus_tn's stub docs (CONFIRMED_EMPTY_AT_SOURCE below) are a different root cause --
see its own comment further down.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

PURE_NAV_STUBS = [
    "data/corpus/mes/page.php/q_code_menu=23.fr.md",
    "data/corpus/mes/page.php/q_code_menu=24_code_menu_parent=23.fr.md",
    "data/corpus/mes/page.php/q_code_menu=26_code_menu_parent=23.fr.md",
    "data/corpus/mes/page.php/q_code_menu=27_code_menu_parent=23.fr.md",
    "data/corpus/mes/page.php/q_code_menu=587_code_menu_parent=12.fr.md",
    "data/corpus/mes/page.php/q_code_menu=129.ar.md",
    "data/corpus/mes/page.php/q_code_menu=47.fr.md",
    "data/corpus/transtu/fr/0040-dossier-d-abonnement-scolaire-et-universitaire.html.fr.md",
    "data/corpus/aneti/fo/Fr/global.php/q_menu1=159.fr.md",
]

# Same disposal (status=removed), different root cause: these were caught by a content-selector
# bug (too-broad CSS selector also matching a sidebar/footer widget, so _pick_content() grabbed
# that instead of the real article -- see sources.yaml's cnss/erasmus_tn notes for the fix).
# Re-crawling with the fixed selector was tested directly against the raw HTML and confirmed
# the REAL article is itself empty or near-empty on the source for every one of these -- so the
# fix is correct, but the crawler's own min_words filter then silently drops the (now-empty)
# replacement doc instead of touching the stale wrong file, leaving the old sidebar/footer text
# sitting on disk indefinitely. Worth a real pipeline.py fix later (mark a URL's prior docs
# "missing" when a fresh fetch yields zero qualifying documents); done by hand here for now.
CONFIRMED_EMPTY_AT_SOURCE = [
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal-secteur_agr2_avantage3.fr.md",
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal_secteur_ag1_attestations.fr.md",
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal_secteur_ag1_cotisation.fr.md",
    "data/corpus/cnss/web/salarie/sal_asset_services/-/asset_publisher/0xT7/content/sal-secteur_-non_agr3_actions_licenciement2.fr.md",
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal-secteur_-non_agr3_avantage2.ar.md",
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal-secteur_-non_agr3_avantage3.ar.md",
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal_secteur_ag1_attestations.ar.md",
    "data/corpus/cnss/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal_secteur_ag1_cotisation.ar.md",
    "data/corpus/erasmus_tn/Index-Etudiants.php.en.md",
    "data/corpus/erasmus_tn/Le-programme-Erasmus+.php.en.md",
    "data/corpus/erasmus_tn/Mobilités-Internationales-Etudiants-EMJMD.php.en.md",
    "data/corpus/erasmus_tn/Mobilités-Internationales-Etudiants-ICM.php.en.md",
    "data/corpus/erasmus_tn/Trainings-Etudiants-Comment-préparer-votre-candidature.php.en.md",
]

PURE_NAV_STUBS = PURE_NAV_STUBS + CONFIRMED_EMPTY_AT_SOURCE

STATUS_RE = re.compile(r"^status: \w+$", re.M)


def main():
    changed = 0
    for rel in PURE_NAV_STUBS:
        path = ROOT / rel
        if not path.exists():
            print(f"SKIP (not found): {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if "status: removed" in text.split("---", 2)[1]:
            print(f"already removed: {rel}")
            continue
        new_text = STATUS_RE.sub("status: removed", text, count=1)
        path.write_text(new_text, encoding="utf-8")
        changed += 1
        print(f"marked removed: {rel}")
    print(f"\n{changed} document(s) marked status=removed")
    print("Run scripts/fix_doc_type_labels.py's catalog-rebuild pattern (or a crawl) to "
          "refresh catalog.jsonl/sync_state.json once satisfied with the list.")


if __name__ == "__main__":
    main()
