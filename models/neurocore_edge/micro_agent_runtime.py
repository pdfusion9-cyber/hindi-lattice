"""Deterministic hybrid micro-agent control-plane reference; no model calls or training."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional, Sequence
import uuid


class ReasoningMode(str, Enum):
    NORMAL = "N"
    DEEP = "D"
    REFLECT = "R"
    EXPLORE = "E"


class Difficulty(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    user_input: str
    language: str = "hi"
    difficulty: Difficulty = Difficulty.SIMPLE
    required_capabilities: tuple[str, ...] = ()
    budget_profile: str = "balanced"
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    task_id: str
    agent_id: str
    status: TaskStatus
    result: str
    evidence_refs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    confidence: Optional[float] = None
    requires_review: bool = False


@dataclass
class TaskState:
    task: AgentTask
    status: TaskStatus = TaskStatus.PENDING
    selected_agents: list[str] = field(default_factory=list)
    reasoning_trace: list[ReasoningMode] = field(default_factory=list)
    findings: list[AgentResult] = field(default_factory=list)
    verification_status: str = "pending"
    final_response: Optional[str] = None


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    capabilities: frozenset[str]
    description: str


@dataclass(frozen=True)
class BudgetPolicy:
    max_domain_agents: int
    max_exploration_branches: int
    max_reasoning_rounds: int
    run_reflection: bool
    run_verification: bool


BALANCED_POLICIES: Mapping[Difficulty, BudgetPolicy] = {
    Difficulty.SIMPLE: BudgetPolicy(1, 0, 1, False, True),
    Difficulty.MEDIUM: BudgetPolicy(2, 1, 2, True, True),
    Difficulty.HARD: BudgetPolicy(3, 2, 3, True, True),
}


DEFAULT_AGENT_SPECS = (
    AgentSpec("hindi", frozenset({"language", "translation", "dialogue"}), "Hindi language and dialogue"),
    AgentSpec("research", frozenset({"research", "sources", "fact_finding"}), "Research and evidence collection"),
    AgentSpec("document", frozenset({"document", "ocr", "extraction"}), "Document parsing and extraction"),
    AgentSpec("legal", frozenset({"legal", "jurisdiction", "policy"}), "Legal-domain analysis; not legal advice"),
    AgentSpec("planning", frozenset({"planning", "decomposition", "workflow"}), "Planning and task decomposition"),
    AgentSpec("qa", frozenset({"question_answering", "reasoning", "explanation"}), "Question answering and explanation"),
)


def make_task(user_input: str, *, language: str = "hi",
              difficulty: Difficulty = Difficulty.SIMPLE,
              required_capabilities: Sequence[str] = (),
              context: Optional[Mapping[str, object]] = None) -> AgentTask:
    if not user_input.strip():
        raise ValueError("user_input must not be empty")
    return AgentTask(
        task_id=f"task_{uuid.uuid4().hex[:12]}",
        user_input=user_input,
        language=language,
        difficulty=difficulty,
        required_capabilities=tuple(required_capabilities),
        context=dict(context or {}),
    )


class AgentRegistry:
    def __init__(self, specs: Sequence[AgentSpec] = DEFAULT_AGENT_SPECS):
        self._specs = {spec.agent_id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("agent_id values must be unique")

    def select(self, capabilities: Sequence[str], limit: int) -> list[AgentSpec]:
        if limit < 1:
            return []
        wanted = set(capabilities)
        ranked = sorted(
            self._specs.values(),
            key=lambda spec: (-len(wanted.intersection(spec.capabilities)), spec.agent_id),
        )
        return [spec for spec in ranked if wanted.intersection(spec.capabilities)][:limit]

    def get(self, agent_id: str) -> AgentSpec:
        return self._specs[agent_id]


class ReasoningController:
    """Behavior sequencing inspired by, but not reproducing, Mole-Syn."""
    def sequence(self, difficulty: Difficulty, *, uncertainty: bool = False) -> tuple[ReasoningMode, ...]:
        if difficulty == Difficulty.SIMPLE:
            return (ReasoningMode.NORMAL,)
        if difficulty == Difficulty.MEDIUM:
            modes = [ReasoningMode.DEEP]
            if uncertainty:
                modes.append(ReasoningMode.EXPLORE)
            modes.append(ReasoningMode.REFLECT)
            return tuple(modes)
        modes = [ReasoningMode.DEEP, ReasoningMode.EXPLORE]
        if uncertainty:
            modes.append(ReasoningMode.EXPLORE)
        modes.append(ReasoningMode.REFLECT)
        return tuple(modes)


AgentCallable = Callable[[AgentTask], AgentResult]


class MicroAgentOrchestrator:
    """Routes tasks to explicitly registered host handlers."""
    def __init__(self, registry: Optional[AgentRegistry] = None,
                 handlers: Optional[Mapping[str, AgentCallable]] = None):
        self.registry = registry or AgentRegistry()
        self.handlers = dict(handlers or {})
        self.reasoning = ReasoningController()

    def plan(self, task: AgentTask, *, uncertainty: bool = False) -> TaskState:
        policy = BALANCED_POLICIES[task.difficulty]
        specs = self.registry.select(task.required_capabilities, policy.max_domain_agents)
        state = TaskState(task=task, selected_agents=[s.agent_id for s in specs])
        state.reasoning_trace.extend(self.reasoning.sequence(task.difficulty, uncertainty=uncertainty))
        return state

    def execute(self, state: TaskState) -> TaskState:
        state.status = TaskStatus.RUNNING
        for agent_id in state.selected_agents:
            handler = self.handlers.get(agent_id)
            if handler is None:
                state.findings.append(AgentResult(
                    task_id=state.task.task_id, agent_id=agent_id,
                    status=TaskStatus.NEEDS_REVIEW,
                    result="No runtime handler registered for this agent.",
                    limitations=("Role contract only; host handler is absent.",),
                    requires_review=True,
                ))
                continue
            result = handler(state.task)
            if result.task_id != state.task.task_id or result.agent_id != agent_id:
                raise ValueError("agent result identity does not match dispatched task")
            state.findings.append(result)
        state.status = (
            TaskStatus.NEEDS_REVIEW
            if any(item.requires_review or item.status != TaskStatus.COMPLETED for item in state.findings)
            else TaskStatus.COMPLETED
        )
        return state


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    checked_results: int
    missing_evidence_refs: tuple[str, ...]
    review_required_agents: tuple[str, ...]
    notes: tuple[str, ...]


class EvidenceVerifier:
    """Checks evidence-ID presence only, not source authenticity or claim truth."""
    def verify(self, results: Sequence[AgentResult],
               available_evidence_refs: Sequence[str] = ()) -> VerificationReport:
        available = set(available_evidence_refs)
        missing = sorted({ref for result in results for ref in result.evidence_refs if ref not in available})
        review = sorted({
            result.agent_id for result in results
            if result.requires_review or result.status != TaskStatus.COMPLETED
        })
        notes = ["Evidence IDs checked for presence only; source truth was not assessed."]
        if not results:
            notes.append("No agent results were supplied.")
        return VerificationReport(bool(results) and not missing and not review,
                                  len(results), tuple(missing), tuple(review), tuple(notes))


def synthesize_response(results: Sequence[AgentResult], verification: VerificationReport,
                        *, language: str = "hi") -> str:
    if not verification.passed:
        return ("समीक्षा आवश्यक है: एजेंट परिणामों या प्रमाण-संदर्भों की जाँच अधूरी है।"
                if language.startswith("hi") else
                "Review required: agent results or evidence references are incomplete.")
    content = [item.result.strip() for item in results if item.result.strip()]
    if not content:
        return "कोई सत्यापित परिणाम उपलब्ध नहीं है।" if language.startswith("hi") else "No verified result is available."
    return "\n".join(content)
