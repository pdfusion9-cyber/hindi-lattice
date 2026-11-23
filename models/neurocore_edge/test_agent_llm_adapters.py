import json
import unittest

from agent_llm_adapters import (
    HindiAgentAdapter, PlanningAgentAdapter, ProviderReply, ResearchAgentAdapter,
    build_domain_handlers,
)
from micro_agent_runtime import (
    AgentExecutionContext, AgentTask, Difficulty, TaskStatus, make_task,
)
from threading import Event
import time


def context():
    return AgentExecutionContext(time.monotonic() + 2, Event())


class FakeProvider:
    def __init__(self, payload, model="test-model"):
        self.payload = payload
        self.requests = []
        self.model = model

    def __call__(self, request, execution_context):
        execution_context.raise_if_cancelled()
        self.requests.append(request)
        return ProviderReply(json.dumps(self.payload, ensure_ascii=False), self.model, 12, 20)


class AdapterTests(unittest.TestCase):
    def test_hindi_adapter_builds_structured_request_and_validates(self):
        provider = FakeProvider({
            "answer": "नमस्ते, मैं सहायता कर सकता हूँ।",
            "language": "hi",
            "notes": ["कृपया प्रश्न बताइए।"],
            "limitations": [],
            "confidence": 0.93,
            "needs_review": False,
            "evidence_refs": [],
        })
        task = make_task("Hello", required_capabilities=("language",))
        result = HindiAgentAdapter(provider)(task, context())
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.agent_id, "hindi")
        self.assertEqual(provider.requests[0].messages[0]["role"], "system")
        self.assertEqual(provider.requests[0].max_output_tokens, 256)

    def test_research_adapter_requires_supplied_source_ids(self):
        provider = FakeProvider({
            "findings": [{"claim": "दिल्ली भारत की राजधानी है।",
                          "evidence_refs": ["src:1"], "support_note": "दिए गए स्रोत में स्पष्ट है।"}],
            "gaps": [], "limitations": [], "confidence": 0.9, "needs_review": False,
        })
        task = make_task("भारत की राजधानी?", difficulty=Difficulty.MEDIUM,
                         required_capabilities=("research",),
                         context={"source_material": [{"ref": "src:1", "text": "भारत की राजधानी नई दिल्ली है।"}]})
        result = ResearchAgentAdapter(provider)(task, context())
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.evidence_refs, ("src:1",))

    def test_research_adapter_rejects_invented_source(self):
        provider = FakeProvider({
            "findings": [{"claim": "दावा", "evidence_refs": ["fake:99"], "support_note": "समर्थित"}],
            "gaps": [], "limitations": [], "confidence": 0.8, "needs_review": False,
        })
        task = make_task("शोध करें", required_capabilities=("research",),
                         context={"source_material": [{"ref": "src:1", "text": "source"}]})
        with self.assertRaisesRegex(ValueError, "invented evidence refs"):
            ResearchAgentAdapter(provider)(task, context())

    def test_research_without_sources_stays_needs_review(self):
        provider = FakeProvider({
            "findings": [], "gaps": ["कोई स्रोत उपलब्ध नहीं"], "limitations": [],
            "confidence": 0.2, "needs_review": False,
        })
        task = make_task("तथ्य जाँचें")
        result = ResearchAgentAdapter(provider)(task, context())
        self.assertEqual(result.status, TaskStatus.NEEDS_REVIEW)
        self.assertTrue(result.requires_review)

    def test_planning_adapter_validates_dependencies_and_is_plan_only(self):
        provider = FakeProvider({
            "goal": "परीक्षण पूरा करना",
            "steps": [
                {"step": "टेस्ट लिखें", "depends_on": []},
                {"step": "टेस्ट चलाएँ", "depends_on": [1]},
            ],
            "risks": ["पर्याप्त समय न मिलना"],
            "success_criteria": ["सभी परीक्षण पास हों"],
            "requires_human_approval": False,
            "limitations": [],
        })
        task = make_task("योजना बनाएँ", difficulty=Difficulty.HARD,
                         required_capabilities=("planning",))
        result = PlanningAgentAdapter(provider)(task, context())
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertIn("no external action was executed", result.limitations[0])

    def test_planning_rejects_forward_dependency(self):
        provider = FakeProvider({
            "goal": "x",
            "steps": [{"step": "पहला", "depends_on": [2]}],
            "risks": [], "success_criteria": ["पूरा"], "requires_human_approval": False,
            "limitations": [],
        })
        with self.assertRaisesRegex(ValueError, "earlier step"):
            PlanningAgentAdapter(provider)(make_task("योजना"), context())

    def test_adapter_rejects_malformed_json_and_invalid_confidence(self):
        class BadProvider:
            def __call__(self, request, execution_context):
                return ProviderReply("not-json", "bad")
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            HindiAgentAdapter(BadProvider())(make_task("हिंदी"), context())

        provider = FakeProvider({
            "answer": "उत्तर", "language": "hi", "notes": [], "limitations": [],
            "confidence": 7, "needs_review": False, "evidence_refs": [],
        })
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            HindiAgentAdapter(provider)(make_task("हिंदी"), context())

    def test_build_domain_handlers_registers_three_adapters(self):
        handlers = build_domain_handlers(FakeProvider({}))
        self.assertEqual(set(handlers), {"hindi", "research", "planning"})


if __name__ == "__main__":
    unittest.main()
