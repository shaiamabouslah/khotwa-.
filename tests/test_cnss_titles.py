"""scripts/cnss_titles.py: CNSS CMS codes -> readable titles in the page's language."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cnss_titles import readable_title  # noqa: E402


class TestReadableTitle(unittest.TestCase):
    def test_french_page(self):
        self.assertEqual(readable_title("Emp Secteur pêche3 Immatriculation (Employeur)", "fr"),
                         "Immatriculation — employeur, secteur de la pêche")

    def test_arabic_page_gets_an_arabic_title(self):
        self.assertEqual(readable_title("Emp Secteur pêche3 Immatriculation (Employeur)", "ar"),
                         "التسجيل — المؤجر، قطاع الصيد البحري")

    def test_non_agricultural_is_not_read_as_agricultural(self):
        self.assertIn("secteur non agricole", readable_title("Non Sal Secteur non agr1 Cotisations (Non Salarié)", "fr"))

    def test_typos_and_separators_in_codes(self):
        self.assertEqual(readable_title("Emp Secteuré non agr1 Affiliation (Employeur)", "fr"),
                         "Affiliation — employeur, secteur non agricole")
        self.assertEqual(readable_title("Emp secteur-non agr2 affiliation regime complementaire (Employeur)", "fr"),
                         "Affiliation au régime complémentaire — employeur, secteur non agricole")

    def test_leading_heading_is_preferred_over_the_generic_procedure(self):
        body = "## Report de paiement des cotisations\n\nLes nouveaux promoteurs..."
        self.assertEqual(readable_title("Non Sal Secteur non agr2 avantage2 (Non Salarié)", "fr", body),
                         "Report de paiement des cotisations — non-salarié, secteur non agricole")

    def test_ordinary_titles_are_left_alone(self):
        self.assertIsNone(readable_title("Bourse universitaire pour les nouveaux bacheliers", "fr"))


if __name__ == "__main__":
    unittest.main()
