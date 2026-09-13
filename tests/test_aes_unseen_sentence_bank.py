import tempfile
import unittest
from pathlib import Path

from injection.evaluate_aes_unseen_sentence_bank import audit_banks, main
from paer.rhi_experiment_utils import bind_directory


class BankAuditTests(unittest.TestCase):
    def check_banks(self, old, new):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "old.txt", Path(folder) / "new.txt"
            a.write_text(old, encoding="utf-8")
            b.write_text(new, encoding="utf-8")
            return audit_banks(a, b)

    def test_disjoint(self):
        self.assertEqual(self.check_banks("Old one\nOld two", "New one\nNew two")["normalized_exact_overlap"], 0)

    def test_normalized_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.check_banks("ABC   sentence", "ＡＢＣ sentence")

    def test_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.check_banks("One\nTwo", "New\nNEW")

    def test_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "equal"):
            self.check_banks("One\nTwo", "Three")

    def test_empty(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.check_banks("", "")

    def test_foreign_output_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            original = p / "original.txt"
            original.write_text("keep")
            with self.assertRaises(FileExistsError):
                bind_directory(p, {"protocol": "new"})
            self.assertEqual(original.read_text(), "keep")

    def test_changed_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            bind_directory(p, {"bank": "old"})
            with self.assertRaises(ValueError):
                bind_directory(p, {"bank": "new"})


if __name__ == "__main__":
    unittest.main()
