import unittest

from micro_agent_runtime import (
    AgentResult, Difficulty, EvidenceVerifier, MicroAgentOrchestrator,
    ReasoningController, ReasoningMode, TaskStatus, make_task, synthesize_response,
)


class MicroAgentRuntimeTests(unittest.TestCase):
    def test_hard_routing_is_bounded_and_reflects(self):
        task = make_task("कानूनी शोध", difficulty=Difficulty.HARD,
                         required_capabilities=("legal", "research", "sources", "jurisdiction"))
        state = MicroAgentOrchestrator().plan(task)
        self.assertLessEqual(len(state.selected_agents), 3)
        self.assertEqual(len(state.reasoning_trace), 3)
        self.assertEqual(state.reasoning_trace[-1], ReasoningMode.REFLECT)

    def test_simple_uses_normal_mode(self):
        task = make_task("नमस्ते")
        state = MicroAgentOrchestrator().plan(task)
        self.assertEqual(state.reasoning_trace, [ReasoningMode.NORMAL])
        self.assertLessEqual(len(state.selected_agents), 1)

    def test_medium_uncertainty_adds_exploration(self):
        modes = ReasoningController().sequence(Difficulty.MEDIUM, uncertainty=True)
        self.assertEqual(modes, (ReasoningMode.DEEP, ReasoningMode.EXPLORE, ReasoningMode.REFLECT))

    def test_missing_handler_requires_review(self):
        task = make_task("हिंदी में उत्तर", required_capabilities=("language",))
        state = MicroAgentOrchestrator().execute(MicroAgentOrchestrator().plan(task))
        self.assertEqual(state.status, TaskStatus.NEEDS_REVIEW)
        self.assertTrue(state.findings[0].requires_review)

    def test_handler_identity_is_checked(self):
        task = make_task("hello", required_capabilities=("language",))
        state = MicroAgentOrchestrator().plan(task)
        agent_id = state.selected_agents[0]
        handler = lambda t: AgentResult("wrong", agent_id, TaskStatus.COMPLETED, "bad")
        with self.assertRaises(ValueError):
            MicroAgentOrchestrator(handlers={agent_id: handler}).execute(state)

    def test_verifier_checks_presence_not_truth(self):
        item = AgentResult("t", "research", TaskStatus.COMPLETED, "claim", ("src:1",))
        verifier = EvidenceVerifier()
        self.assertFalse(verifier.verify([item]).passed)
        report = verifier.verify([item], ["src:1"])
        self.assertTrue(report.passed)
        self.assertIn("truth was not assessed", report.notes[0])

    def test_unverified_response_is_blocked(self):
        item = AgentResult("t", "qa", TaskStatus.NEEDS_REVIEW, "uncertain", requires_review=True)
        report = EvidenceVerifier().verify([item])
        self.assertIn("समीक्षा आवश्यक", synthesize_response([item], report))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            make_task("  ")


if __name__ == "__main__":
    unittest.main()
