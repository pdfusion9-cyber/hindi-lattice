import tempfile
import unittest
from pathlib import Path

from benchmark_neurocore_tokenizer import contains_pii_pattern, iter_safe_records


class PrivacyFilterTests(unittest.TestCase):
    def test_detects_email(self):
        self.assertTrue(contains_pii_pattern("संपर्क test@example.com"))

    def test_detects_indian_mobile(self):
        self.assertTrue(contains_pii_pattern("फोन 9876543210"))

    def test_detects_aadhaar_like_number(self):
        self.assertTrue(contains_pii_pattern("1234 5678 9012"))

    def test_keeps_ordinary_hindi_text(self):
        self.assertFalse(contains_pii_pattern("भारत की भाषाई विविधता"))

    def test_stream_skips_pii_without_persisting_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text(
                "भारत की भाषाई विविधता\nफोन 9876543210\nहिंदी भाषा\n",
                encoding="utf-8",
            )
            self.assertEqual(
                list(iter_safe_records(path, 10)),
                ["भारत की भाषाई विविधता", "हिंदी भाषा"],
            )


if __name__ == "__main__":
    unittest.main()
