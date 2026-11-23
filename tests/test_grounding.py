"""Unit tests for scripts/grounding.py (pure text logic, no model).

Run: python -m unittest discover -s tests -t .
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from grounding import check, numbers  # noqa: E402

SOURCE = ("Bourse universitaire pour les nouveaux bacheliers\nhttps://www.ooun.rnu.tn/bourses\n"
          "Le revenu annuel du foyer ne doit pas dépasser 1 200 dinars par personne. Le dossier est "
          "déposé avant le 25/11/2025. Tél. : (216.71) 791 132")


class TestNumbers(unittest.TestCase):
    def test_thousands_separators_and_decimal_comma(self):
        self.assertEqual(numbers("1 200 dinars, 1.200 DT, 3,5 %"), {"1200", "3.5"})

    def test_list_markers_citations_and_single_digits_are_ignored(self):
        self.assertEqual(numbers("1. Déposer 2 copies [3]\n2. Attendre 15 jours"), {"15"})

    def test_arabic_indic_digits(self):
        self.assertEqual(numbers("قبل ٢٥ نوفمبر"), {"25"})


class TestCheck(unittest.TestCase):
    def test_supported_answer_is_grounded(self):
        r = check("Le revenu ne doit pas dépasser 1200 dinars [1], dossier avant le 25/11/2025 [1].", [SOURCE])
        self.assertTrue(r["grounded"], r)

    def test_invented_amount_is_flagged(self):
        r = check("Le revenu ne doit pas dépasser 1500 dinars [1].", [SOURCE])
        self.assertEqual(r["unsupported_numbers"], ["1500"])
        self.assertFalse(r["grounded"])

    def test_number_from_the_question_is_allowed(self):
        self.assertTrue(check("À 24 ans, vous pouvez déposer le dossier [1].", [SOURCE], "J'ai 24 ans")["grounded"])

    def test_phone_number_with_other_grouping_is_allowed(self):
        self.assertTrue(check("Appelez le 71 791 132 [1].", [SOURCE])["grounded"])

    def test_invented_link_is_flagged(self):
        r = check("Postulez sur https://bourses-tunisie.example.com [1].", [SOURCE])
        self.assertEqual(r["unsupported_urls"], ["https://bourses-tunisie.example.com"])

    def test_source_link_with_trailing_slash_is_allowed(self):
        self.assertTrue(check("Voir https://www.ooun.rnu.tn/bourses/ [1].", [SOURCE])["grounded"])

    def test_citation_to_a_missing_source_is_flagged(self):
        self.assertEqual(check("Voir la procédure [3].", [SOURCE, SOURCE])["bad_citations"], [3])

    def test_answer_without_facts_is_grounded(self):
        self.assertTrue(check("Je n'ai pas trouvé cette information dans les sources.", [SOURCE])["grounded"])


if __name__ == "__main__":
    unittest.main()
