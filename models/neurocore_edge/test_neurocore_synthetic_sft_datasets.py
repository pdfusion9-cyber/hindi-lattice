import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent / "synthetic_sft"
FILES = (
    "expert16_self_correction_train.jsonl",
    "expert16_self_correction_validation.jsonl",
    "expert17_planning_train.jsonl",
    "expert17_planning_validation.jsonl",
)
PII = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
    re.compile(r"(?i)\b[A-Z]{5}\d{4}[A-Z]\b"),
)
RUBRIC = ("लक्ष्य:", "अवधि:", "जिम्मेदार भूमिका:", "संसाधन/निर्भरता:", "सूचक:", "जोखिम:", "वैकल्पिक कदम:")

def load_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]

class SyntheticHindiSFTTests(unittest.TestCase):
    def test_schema_inventory_and_text_lengths(self):
        for name in FILES:
            for row in load_rows(name):
                self.assertEqual(set(row), {"prompt", "response"})
                self.assertGreater(len(row["prompt"]), 20)
                self.assertGreater(len(row["response"]), 60)
        self.assertEqual(len(load_rows(FILES[0])), 14)
        self.assertEqual(len(load_rows(FILES[1])), 4)
        self.assertEqual(len(load_rows(FILES[2])), 15)
        self.assertEqual(len(load_rows(FILES[3])), 4)

    def test_hindi_and_common_pii_patterns(self):
        for name in FILES:
            for row in load_rows(name):
                joined = row["prompt"] + " " + row["response"]
                self.assertRegex(joined, r"[\u0900-\u097F]")
                for pattern in PII:
                    self.assertIsNone(pattern.search(joined))

    def test_pii_detectors_match_representative_values(self):
        samples = ("contact@example.com", "9876543210", "+91 9876543210", "1234 5678 9012", "ABCDE1234F")
        for sample in samples:
            self.assertTrue(any(pattern.search(sample) for pattern in PII), sample)

    def test_train_and_validation_prompts_are_disjoint(self):
        for prefix in ("expert16_self_correction", "expert17_planning"):
            train = {row["prompt"] for row in load_rows(prefix + "_train.jsonl")}
            validation = {row["prompt"] for row in load_rows(prefix + "_validation.jsonl")}
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertFalse(train & validation)

    def test_expert16_every_training_example_is_explicitly_prior_answer_based(self):
        rows = load_rows("expert16_self_correction_train.jsonl")
        self.assertEqual(len([r for r in rows if "पिछले उत्तर" in r["prompt"]]), 14)
        self.assertTrue(all(any(term in r["response"] for term in ("मुझे", "पिछला", "मेरा")) for r in rows))

    def test_expert16_validation_tests_correction_calibration(self):
        rows = load_rows("expert16_self_correction_validation.jsonl")
        joined = "\n".join(r["prompt"] + " " + r["response"] for r in rows)
        self.assertIn("पिछला उत्तर सही है", joined)
        self.assertIn("जानकारी से वास्तविक दूरी तय नहीं की जा सकती", joined)
        self.assertIn("36", joined)
        self.assertIn("2500 मिलीलीटर", joined)

    def test_expert17_all_plans_cover_full_planning_rubric(self):
        for name in ("expert17_planning_train.jsonl", "expert17_planning_validation.jsonl"):
            for row in load_rows(name):
                for field in RUBRIC:
                    self.assertIn(field, row["response"], (name, row["prompt"], field))

    def test_expert17_planting_schedule_measurements_fit_90_days(self):
        rows = load_rows("expert17_planning_train.jsonl")
        planting = next(r for r in rows if "200 पौधे" in r["prompt"])
        self.assertIn("11–15", planting["response"])
        self.assertIn("रोपाई के 30 और 60 दिन बाद", planting["response"])
        self.assertNotIn("90 दिन बाद", planting["response"])

if __name__ == "__main__":
    unittest.main()
