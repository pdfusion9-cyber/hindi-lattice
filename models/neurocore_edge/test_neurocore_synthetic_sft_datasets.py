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
)

def load_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]

class SyntheticHindiSFTTests(unittest.TestCase):
    def test_schema_and_nonempty_splits(self):
        for name in FILES:
            for row in load_rows(name):
                self.assertEqual(set(row), {"prompt", "response"})
                self.assertGreater(len(row["prompt"]), 20)
                self.assertGreater(len(row["response"]), 60)
        self.assertGreaterEqual(len(load_rows(FILES[0])), 14)
        self.assertGreaterEqual(len(load_rows(FILES[1])), 4)
        self.assertGreaterEqual(len(load_rows(FILES[2])), 15)
        self.assertGreaterEqual(len(load_rows(FILES[3])), 4)

    def test_hindi_and_common_pii_patterns(self):
        for name in FILES:
            for row in load_rows(name):
                joined = row["prompt"] + " " + row["response"]
                self.assertRegex(joined, r"[\u0900-\u097F]")
                for pattern in PII:
                    self.assertIsNone(pattern.search(joined))

    def test_pii_detectors_match_representative_values(self):
        samples = ("contact@example.com", "9876543210", "+91 9876543210", "1234 5678 9012")
        for sample in samples:
            self.assertTrue(any(pattern.search(sample) for pattern in PII), sample)

    def test_train_and_validation_prompts_are_disjoint(self):
        for prefix in ("expert16_self_correction", "expert17_planning"):
            train = {row["prompt"] for row in load_rows(prefix + "_train.jsonl")}
            validation = {row["prompt"] for row in load_rows(prefix + "_validation.jsonl")}
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertFalse(train & validation)

    def test_expert16_contains_explicit_prior_answer_revision(self):
        rows = load_rows("expert16_self_correction_train.jsonl")
        prior_answer_rows = [r for r in rows if "पिछले उत्तर" in r["prompt"]]
        revised_rows = [r for r in prior_answer_rows if "पिछला उत्तर सुधारना चाहिए" in r["response"]]
        self.assertGreaterEqual(len(prior_answer_rows), 6)
        self.assertGreaterEqual(len(revised_rows), 6)

    def test_expert17_advanced_plans_cover_rubric_fields(self):
        rows = [r for r in load_rows("expert17_planning_train.jsonl") if "उन्नत योजना अभ्यास" in r["prompt"]]
        self.assertGreaterEqual(len(rows), 6)
        required = ("जिम्मेदार भूमिका:", "संसाधन/निर्भरता:", "सूचक:", "जोखिम:", "वैकल्पिक कदम:")
        for row in rows:
            for field in required:
                self.assertIn(field, row["response"])

if __name__ == "__main__":
    unittest.main()
