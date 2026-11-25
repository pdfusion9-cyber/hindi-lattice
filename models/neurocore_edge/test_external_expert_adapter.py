import unittest

from external_expert_adapter import (
    ExpertRequest, ExternalExpertAdapter, FixedExpertRouter
)


class FakeBackend:
    def __init__(self, name):
        self.name = name

    def generate(self, prompt, **kwargs):
        return f"{self.name}:{prompt}"


class ExternalAdapterTests(unittest.TestCase):
    def test_fixed_expert_mapping_and_dispatch(self):
        adapters = {
            i: ExternalExpertAdapter(i, f"model-{i}", FakeBackend(f"m{i}"))
            for i in range(20)
        }
        router = FixedExpertRouter(adapters)
        result = router.dispatch(ExpertRequest(12, "translate", "नमस्ते"))
        self.assertEqual(result.expert_id, 12)
        self.assertEqual(result.model_key, "model-12")
        self.assertEqual(result.text, "m12:नमस्ते")
        self.assertTrue(result.weights_frozen)

    def test_cross_expert_dispatch_rejected(self):
        adapter = ExternalExpertAdapter(4, "tts", FakeBackend("tts"))
        with self.assertRaises(ValueError):
            adapter.run(ExpertRequest(5, "tts", "hello"))

    def test_router_requires_all_twenty_experts(self):
        with self.assertRaises(ValueError):
            FixedExpertRouter({0: ExternalExpertAdapter(0, "x", FakeBackend("x"))})


if __name__ == "__main__":
    unittest.main()
