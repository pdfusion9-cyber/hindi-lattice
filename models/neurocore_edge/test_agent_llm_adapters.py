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




class HybridAgentEndToEndTests(unittest.TestCase):
    """Offline integration test for the complete Hindi -> research -> planning path."""

    def test_three_agents_complete_one_hindi_task_and_independent_verifier_accepts(self):
        from micro_agent_runtime import (
            EvidenceVerifier, MicroAgentOrchestrator, synthesize_response,
        )

        source_ref = "samagra-library"
        source_material = [{
            "ref": source_ref,
            "title": "Samagra Shiksha — Library grant for promoting reading habits",
            "url": "https://samagra.education.gov.in/library.html",
            "text": (
                "Schools must devote one period a week to library reading. "
                "If there is no separate library room, a reading corner may be created. "
                "Readers' Clubs and activities such as storytelling, story writing, "
                "poetry reading and sharing stories are recommended."
            ),
        }]
        task = make_task(
            "राजस्थान के एक छोटे सरकारी प्राथमिक विद्यालय में पुस्तकालय/पठन-कोना "
            "और पाठक क्लब शुरू करने की हिंदी कार्ययोजना बनाइए। केवल दिए गए सरकारी "
            "स्रोत से समर्थित तथ्य बताइए और वर्तमान स्थानीय नियमों की जानकारी न "
            "होने पर उसे स्पष्ट लिखिए।",
            language="hi",
            difficulty=Difficulty.HARD,
            required_capabilities=(
                "language", "research", "sources", "planning", "workflow",
            ),
            context={"source_material": source_material},
        )

        class OfflineWorkflowProvider:
            """Deterministic provider fixture; this is not a live LLM endpoint."""

            def __init__(self):
                self.seen_agents = []

            def __call__(self, request, execution_context):
                execution_context.raise_if_cancelled()
                system_prompt = request.messages[0]["content"]
                user_payload = json.loads(request.messages[1]["content"])
                supplied_sources = user_payload["context"]["source_material"]
                refs = [item["ref"] for item in supplied_sources]
                if source_ref not in refs:
                    raise AssertionError("the source allow-list was not passed to the provider")

                if "Hindi language specialist" in system_prompt:
                    self.seen_agents.append("hindi")
                    payload = {
                        "answer": (
                            "विद्यालय में पुस्तकालय को बच्चों के लिए नियमित और "
                            "आनंददायक पठन-केंद्र बनाया जा सकता है।"
                        ),
                        "language": "hi",
                        "notes": [
                            "सप्ताह में एक निर्धारित पुस्तकालय-अवधि रखें।",
                            "अलग कमरा न होने पर सुलभ पठन-कोना बनाएं।",
                            "कहानी, कविता और पुस्तक-साझाकरण जैसी गतिविधियाँ चलाएँ।",
                        ],
                        "limitations": ["उत्तर दिए गए Samagra Shiksha स्रोत तक सीमित है।"],
                        "confidence": 0.96,
                        "needs_review": False,
                        "evidence_refs": [source_ref],
                    }
                elif "research synthesis specialist" in system_prompt:
                    self.seen_agents.append("research")
                    payload = {
                        "findings": [
                            {
                                "claim": "विद्यालय में सप्ताह में एक पुस्तकालय-अवधि रखी जानी चाहिए।",
                                "evidence_refs": [source_ref],
                                "support_note": "दिए गए सरकारी पृष्ठ पर सप्ताह में एक अवधि का उल्लेख है।",
                            },
                            {
                                "claim": "अलग पुस्तकालय-कक्ष न होने पर पठन-कोना बनाया जा सकता है।",
                                "evidence_refs": [source_ref],
                                "support_note": "दिए गए सरकारी पृष्ठ पर reading corner का विकल्प दिया है।",
                            },
                            {
                                "claim": "पाठक क्लब, कहानी-कथन और कहानी-लेखन जैसी गतिविधियाँ उपयोगी हैं।",
                                "evidence_refs": [source_ref],
                                "support_note": "दिए गए सरकारी पृष्ठ पर Readers' Club और पठन गतिविधियाँ सूचीबद्ध हैं।",
                            },
                        ],
                        "gaps": [
                            "राजस्थान की वर्तमान स्वीकृत अनुदान राशि और स्थानीय खरीद-प्रक्रिया "
                            "की पुष्टि इस स्रोत से नहीं होती; संबंधित वर्तमान आदेश जाँचना होगा।"
                        ],
                        "limitations": ["शोध केवल उपलब्ध कराई गई स्रोत-सामग्री पर आधारित है।"],
                        "confidence": 0.93,
                        "needs_review": False,
                    }
                elif "planning specialist" in system_prompt:
                    self.seen_agents.append("planning")
                    payload = {
                        "goal": "विद्यालय में कम लागत वाला, नियमित और समावेशी पठन-कार्यक्रम शुरू करना",
                        "steps": [
                            {"step": "प्रधानाध्यापक और शिक्षक उपलब्ध स्थान तथा विद्यार्थियों की जरूरतें तय करें।", "depends_on": []},
                            {"step": "आयु-अनुकूल पुस्तकों और स्थानीय भाषा की सामग्री की सूची बनाएं।", "depends_on": [1]},
                            {"step": "अलग कक्ष न हो तो पठन-कोना तैयार करें और पुस्तकें सुलभ रखें।", "depends_on": [1, 2]},
                            {"step": "समय-सारणी में हर सप्ताह एक पुस्तकालय-अवधि निर्धारित करें।", "depends_on": [3]},
                            {"step": "पाठक क्लब बनाकर कहानी-कथन, कविता-पाठ और पुस्तक-साझाकरण करें।", "depends_on": [3]},
                            {"step": "पुस्तक निर्गम और गतिविधियों का सरल रजिस्टर रखकर मासिक समीक्षा करें।", "depends_on": [2, 4, 5]},
                        ],
                        "risks": [
                            "आयु-अनुकूल और स्थानीय भाषा की पुस्तकों की कमी।",
                            "नियमित पुस्तकालय-अवधि के लिए समय-सारणी में जगह न मिलना।",
                        ],
                        "success_criteria": [
                            "हर सप्ताह पुस्तकालय-अवधि आयोजित हो।",
                            "पठन-कोने/पुस्तकालय का उपयोग और पुस्तक निर्गम दर्ज हो।",
                            "स्थानीय अनुदान व खरीद नियम लागू करने से पहले वर्तमान आदेश से सत्यापित हों।",
                        ],
                        "requires_human_approval": False,
                        "limitations": ["यह योजना है; कोई खरीद या बाहरी कार्रवाई निष्पादित नहीं की गई।"],
                    }
                else:
                    raise AssertionError("unexpected specialist adapter was invoked")

                return ProviderReply(
                    json.dumps(payload, ensure_ascii=False),
                    "offline-e2e-fixture",
                    prompt_tokens=80,
                    completion_tokens=140,
                )

        provider = OfflineWorkflowProvider()
        orchestrator = MicroAgentOrchestrator(handlers=build_domain_handlers(provider))
        state = orchestrator.plan(task, uncertainty=True)
        result = orchestrator.execute_parallel(state)

        self.assertEqual(set(state.selected_agents), {"hindi", "research", "planning"})
        self.assertCountEqual(provider.seen_agents, ["hindi", "research", "planning"])
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual({item.agent_id for item in result.findings},
                         {"hindi", "research", "planning"})
        self.assertTrue(all(item.status == TaskStatus.COMPLETED and not item.requires_review
                            for item in result.findings))
        self.assertEqual(result.resource_usage.agent_calls, 3)
        self.assertEqual(result.resource_usage.completed_agents, 3)

        verification = EvidenceVerifier().verify(
            result.findings, available_evidence_refs=[source_ref],
        )
        self.assertTrue(verification.passed)
        self.assertEqual(verification.checked_results, 3)
        final_response = synthesize_response(
            result.findings, verification, language="hi",
        )
        self.assertIn("लक्ष्य:", final_response)
        self.assertIn("प्रमाण:", final_response)
        self.assertIn("पुस्तकालय", final_response)
        self.assertIn("Plan only; no external action was executed.",
                      next(item for item in result.findings
                           if item.agent_id == "planning").limitations)

    def test_independent_verifier_blocks_an_unknown_evidence_reference(self):
        from micro_agent_runtime import AgentResult, EvidenceVerifier, synthesize_response

        untrusted = AgentResult(
            "task_e2e", "research", TaskStatus.COMPLETED,
            "एक अपुष्ट दावा", ("invented:99",),
        )
        verification = EvidenceVerifier().verify(
            [untrusted], available_evidence_refs=["samagra-library"],
        )
        self.assertFalse(verification.passed)
        self.assertEqual(verification.missing_evidence_refs, ("invented:99",))
        self.assertIn("समीक्षा आवश्यक",
                      synthesize_response([untrusted], verification, language="hi"))


if __name__ == "__main__":
    unittest.main()
