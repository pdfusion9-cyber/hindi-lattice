import unittest

from neurocore_intelligence_benchmark import heuristic_diagnostics


class BenchmarkDiagnosticTests(unittest.TestCase):
    def test_wrong_logic_conclusion_is_flagged(self):
        answer = "निष्कर्ष: नहीं, इन कथनों से यह निष्कर्ष नहीं निकलता कि रवि अशिक्षित नहीं है।"
        checks = heuristic_diagnostics("logic_hi_01", "", answer)
        self.assertEqual(checks[0]["status"], "fail")

    def test_correct_logic_conclusion_is_detected(self):
        answer = "निष्कर्ष: हाँ, रवि गणित पढ़ता है और अशिक्षित नहीं है।"
        checks = heuristic_diagnostics("logic_hi_01", "", answer)
        self.assertEqual(checks[0]["status"], "pass")

    def test_geocentric_error_in_corrected_section_is_flagged(self):
        answer = "सही वैज्ञानिक स्थिति: सूर्य पृथ्वी के चारों ओर एक वर्ष में घूमता है।"
        checks = heuristic_diagnostics("self_correction_hi_01", "", answer)
        self.assertEqual(checks[0]["status"], "fail")

    def test_correct_heliocentric_statement_is_detected(self):
        answer = "सही वैज्ञानिक स्थिति: पृथ्वी सूर्य की परिक्रमा लगभग 365.25 दिनों में करती है।"
        checks = heuristic_diagnostics("self_correction_hi_01", "", answer)
        self.assertEqual(checks[0]["status"], "pass")
        self.assertEqual(checks[1]["status"], "pass")

    def test_cutoff_does_not_count_missing_final_answer_as_definitive_failure(self):
        checks = heuristic_diagnostics(
            "math_hi_01", "", "गणना 120 + 270 = 390",
            generation_limit_reached=True,
        )
        self.assertEqual(checks[0]["status"], "incomplete_or_not_observed")


if __name__ == "__main__":
    unittest.main()
