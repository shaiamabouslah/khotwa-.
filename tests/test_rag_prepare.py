"""Unit tests for the deterministic cleaning steps of scripts/rag_prepare.py (no model, no kb/).

Run: python -m unittest discover -s tests -t .
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rag_prepare import MIN_WORDS, gates, normalize  # noqa: E402

DEFINITION = ("L'équivalence est l'assimilation des diplômes et des titres délivrés par un système "
              "éducatif d'un pays étranger aux diplômes et titres délivrés par le système éducatif "
              "tunisien sur la base d'un ensemble de conditions telles que celles relatives à "
              "l'inscription, au nombre d'années d'études et au contenu des programmes.")


class TestNormalizeHeadings(unittest.TestCase):
    def test_short_empty_heading_is_removed(self):
        body, _, notes = normalize("## Procédure\n\nDéposer le dossier.\n\n### Ajouter un commentaire\n")
        self.assertNotIn("commentaire", body)
        self.assertIn("empty_heading_removed", notes)

    def test_long_empty_heading_is_kept_as_text(self):
        body, _, notes = normalize(f"## Définition\n\n### {DEFINITION}\n")
        self.assertIn("L'équivalence est l'assimilation", body)
        self.assertFalse(body.lstrip().startswith("#"))
        self.assertIn("long_heading_kept_as_text", notes)

    def test_page_made_of_one_long_heading_passes_the_length_gate(self):
        body, _, _ = normalize(f"## Définition\n\n### {DEFINITION}\n")
        self.assertGreaterEqual(len(body.split()), MIN_WORDS)
        self.assertNotIn("too_short", gates({"lang": "fr"}, body))

    def test_list_of_service_headings_is_kept(self):
        services = ["Remplissage de la fiche de choix lors de la session principale d'orientation",
                    "Consultation de la répartition globale et du classement selon le type de bac"]
        body, _, _ = normalize("## Services\n\n" + "".join(f"### {s}\n\n" for s in services))
        for s in services:
            self.assertIn(s, body)

    def test_heading_with_content_is_unchanged(self):
        body, _, _ = normalize("## Pièces à fournir\n\n- copie de la CIN\n- photo d'identité\n")
        self.assertTrue(body.startswith("## Pièces à fournir"))


if __name__ == "__main__":
    unittest.main()
