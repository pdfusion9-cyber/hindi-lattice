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
        handler = lambda t, context: AgentResult("wrong", agent_id, TaskStatus.COMPLETED, "bad")
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


    def test_parallel_handlers_run_and_results_keep_registry_order(self):
        task = make_task(
            "जवाब दीजिए", difficulty=Difficulty.MEDIUM,
            required_capabilities=("language", "research", "sources"),
        )
        planner = MicroAgentOrchestrator()
        state = planner.plan(task)
        handlers = {
            agent_id: (lambda t, ctx, aid=agent_id: AgentResult(
                t.task_id, aid, TaskStatus.COMPLETED, f"ok:{aid}"
            ))
            for agent_id in state.selected_agents
        }
        result = MicroAgentOrchestrator(handlers=handlers).execute_parallel(state, max_workers=2)
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual([x.agent_id for x in result.findings], state.selected_agents)
        self.assertEqual(result.resource_usage.completed_agents, len(state.selected_agents))
        self.assertEqual(result.resource_usage.agent_calls, len(state.selected_agents))

    def test_handler_exception_is_captured_as_failed_review(self):
        task = make_task("शोध करें", required_capabilities=("research",))
        state = MicroAgentOrchestrator().plan(task)
        agent_id = state.selected_agents[0]
        def broken(task, context):
            raise RuntimeError("private implementation detail")
        result = MicroAgentOrchestrator(handlers={agent_id: broken}).execute_parallel(state)
        self.assertEqual(result.status, TaskStatus.NEEDS_REVIEW)
        self.assertEqual(result.findings[0].status, TaskStatus.FAILED)
        self.assertNotIn("private implementation detail", result.findings[0].result)

    def test_output_character_budget_blocks_oversized_result(self):
        task = make_task("hi", required_capabilities=("language",))
        state = MicroAgentOrchestrator().plan(task)
        agent_id = state.selected_agents[0]
        def too_long(task, context):
            return AgentResult(task.task_id, agent_id, TaskStatus.COMPLETED, "x" * 20)
        result = MicroAgentOrchestrator(handlers={agent_id: too_long}).execute_parallel(
            state, max_output_chars_per_agent=10
        )
        self.assertEqual(result.status, TaskStatus.NEEDS_REVIEW)
        self.assertIn("exceeded", result.findings[0].result)

    def test_input_budget_rejects_before_handler_dispatch(self):
        task = make_task("this is longer than the allowed budget", required_capabilities=("language",))
        state = MicroAgentOrchestrator().plan(task)
        called = []
        agent_id = state.selected_agents[0]
        def handler(task, context):
            called.append(True)
            return AgentResult(task.task_id, agent_id, TaskStatus.COMPLETED, "no")
        result = MicroAgentOrchestrator(handlers={agent_id: handler}).execute_parallel(
            state, max_input_chars_total=2
        )
        self.assertEqual(result.status, TaskStatus.NEEDS_REVIEW)
        self.assertEqual(called, [])

    def test_timeout_signals_cooperative_cancellation(self):
        import time
        task = make_task("slow task", required_capabilities=("language",))
        state = MicroAgentOrchestrator().plan(task)
        agent_id = state.selected_agents[0]
        observed = []
        def cooperative(task, context):
            while not context.is_cancelled():
                time.sleep(0.002)
            observed.append(True)
            context.raise_if_cancelled()
        result = MicroAgentOrchestrator(handlers={agent_id: cooperative}).execute_parallel(
            state, timeout_seconds=0.03
        )
        self.assertEqual(result.status, TaskStatus.NEEDS_REVIEW)
        self.assertEqual(result.resource_usage.timed_out_agents, 1)
        time.sleep(0.03)
        self.assertEqual(observed, [True])

    def test_nonpositive_execution_budget_is_rejected(self):
        task = make_task("hi", required_capabilities=("language",))
        state = MicroAgentOrchestrator().plan(task)
        with self.assertRaises(ValueError):
            MicroAgentOrchestrator().execute_parallel(state, timeout_seconds=0)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            make_task("  ")


if __name__ == "__main__":
    unittest.main()
