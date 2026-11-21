import unittest

from tools.expert_router import DOMAINS, route_query

class ExpertRouterTests(unittest.TestCase):
    def test_all_ten_domains_are_registered(self):
        self.assertEqual(len(DOMAINS), 10)
        self.assertEqual(len(set(DOMAINS)), 10)

    def test_hindi_science(self):
        self.assertEqual(route_query("कार्बनिक रसायन में अभिक्रिया समझाइए").expert, "science")

    def test_english_math(self):
        self.assertEqual(route_query("Solve this algebra equation").expert, "math")

    def test_hindi_history(self):
        self.assertEqual(route_query("प्राचीन भारत का इतिहास बताइए").expert, "history")

    def test_english_geography(self):
        self.assertEqual(route_query("Explain river climate and geography").expert, "geography")

    def test_coding(self):
        self.assertEqual(route_query("Debug this Python function").expert, "coding")

    def test_news(self):
        self.assertEqual(route_query("आज के करंट अफेयर्स बताइए").expert, "news")

    def test_reasoning(self):
        self.assertEqual(route_query("इस तर्क का निष्कर्ष सिद्ध कीजिए").expert, "reasoning")

    def test_document(self):
        self.assertEqual(route_query("इस PDF का सारांश बनाइए").expert, "document")

    def test_qa_explicit_quiz(self):
        self.assertEqual(route_query("मुझे chemistry के MCQ दीजिए").expert, "qa")

    def test_ambiguous_or_unknown_falls_back(self):
        result = route_query("नमस्ते, कैसे हैं?")
        self.assertEqual(result.expert, "general")
        self.assertEqual(result.confidence, 0.0)

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            route_query("   ")

if __name__ == "__main__":
    unittest.main()
