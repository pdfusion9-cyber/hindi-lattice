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
    resource_usage: Optional[ResourceUsage] = None


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


@dataclass(frozen=True)
class ExecutionBudget:
    """Character-count proxy until a real model tokenizer is connected."""
    max_agents: int
    max_input_chars_total: int
    max_output_chars_per_agent: int
    timeout_seconds: float
    max_workers: int


EXECUTION_BUDGETS: Mapping[Difficulty, ExecutionBudget] = {
    Difficulty.SIMPLE: ExecutionBudget(1, 4000, 4000, 3.0, 1),
    Difficulty.MEDIUM: ExecutionBudget(2, 8000, 6000, 6.0, 2),
    Difficulty.HARD: ExecutionBudget(3, 16000, 8000, 12.0, 3),
}


@dataclass
class AgentExecutionContext:
    """Cooperative cancellation/deadline signal for a host-provided handler."""
    deadline_monotonic: float
    cancellation_event: object

    def is_cancelled(self) -> bool:
        import time
        return self.cancellation_event.is_set() or time.monotonic() >= self.deadline_monotonic

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TimeoutError("agent execution deadline exceeded or cancelled")


@dataclass(frozen=True)
class ResourceUsage:
    agent_calls: int
    completed_agents: int
    failed_agents: int
    timed_out_agents: int
    input_chars_charged: int
    output_chars: int
    elapsed_seconds: float
    budget_profile: str


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


AgentCallable = Callable[[AgentTask, AgentExecutionContext], AgentResult]


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
        """Sequential compatibility path with the same two-argument handler contract."""
        return self.execute_parallel(state, max_workers=1)

    def execute_parallel(
        self,
        state: TaskState,
        *,
        timeout_seconds: Optional[float] = None,
        max_workers: Optional[int] = None,
        max_input_chars_total: Optional[int] = None,
        max_output_chars_per_agent: Optional[int] = None,
    ) -> TaskState:
        """Dispatch bounded independent handlers concurrently.

        Timeouts are cooperative: handlers must periodically call
        context.raise_if_cancelled() or inspect context.is_cancelled().
        Python cannot safely force-stop a running thread. The caller returns
        at the deadline, signals cancellation and marks those results NEEDS_REVIEW.
        """
        import time
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        from threading import Event

        policy = EXECUTION_BUDGETS[state.task.difficulty]
        timeout = policy.timeout_seconds if timeout_seconds is None else timeout_seconds
        worker_limit = policy.max_workers if max_workers is None else max_workers
        input_limit = policy.max_input_chars_total if max_input_chars_total is None else max_input_chars_total
        output_limit = policy.max_output_chars_per_agent if max_output_chars_per_agent is None else max_output_chars_per_agent
        if timeout <= 0 or worker_limit < 1 or input_limit < 1 or output_limit < 1:
            raise ValueError("execution budget values must be positive")

        selected = list(state.selected_agents)
        if len(selected) > policy.max_agents:
            state.status = TaskStatus.NEEDS_REVIEW
            state.findings = [AgentResult(
                task_id=state.task.task_id, agent_id="orchestrator",
                status=TaskStatus.NEEDS_REVIEW,
                result="Selected agent count exceeds the difficulty budget.",
                limitations=("No handlers were executed.",),
                requires_review=True,
            )]
            state.resource_usage = ResourceUsage(0, 0, 0, 0, 0, 0, 0.0, state.task.budget_profile)
            return state

        charged_input = len(state.task.user_input) * len(selected)
        if charged_input > input_limit:
            state.status = TaskStatus.NEEDS_REVIEW
            state.findings = [AgentResult(
                task_id=state.task.task_id, agent_id="orchestrator",
                status=TaskStatus.NEEDS_REVIEW,
                result="Input character budget exceeded before dispatch.",
                limitations=(f"charged={charged_input}; limit={input_limit}",),
                requires_review=True,
            )]
            state.resource_usage = ResourceUsage(0, 0, 0, 0, charged_input, 0, 0.0, state.task.budget_profile)
            return state

        started = time.monotonic()
        deadline = started + timeout
        contexts = {agent_id: AgentExecutionContext(deadline, Event()) for agent_id in selected}
        results: dict[str, AgentResult] = {}
        futures = {}
        submitted = []

        for agent_id in selected:
            handler = self.handlers.get(agent_id)
            if handler is None:
                results[agent_id] = AgentResult(
                    task_id=state.task.task_id, agent_id=agent_id,
                    status=TaskStatus.NEEDS_REVIEW,
                    result="No runtime handler registered for this agent.",
                    limitations=("Role contract only; host handler is absent.",),
                    requires_review=True,
                )
            else:
                submitted.append(agent_id)

        pool = ThreadPoolExecutor(max_workers=min(worker_limit, max(1, len(submitted))))
        try:
            for agent_id in submitted:
                futures[pool.submit(self.handlers[agent_id], state.task, contexts[agent_id])] = agent_id

            pending = set(futures)
            timed_out = 0
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    done = set()
                else:
                    done, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done:
                    for future in list(pending):
                        agent_id = futures[future]
                        contexts[agent_id].cancellation_event.set()
                        future.cancel()
                        results[agent_id] = AgentResult(
                            task_id=state.task.task_id, agent_id=agent_id,
                            status=TaskStatus.NEEDS_REVIEW,
                            result="Agent execution timed out; cooperative cancellation was signalled.",
                            limitations=("A running Python thread cannot be forcibly terminated.",),
                            requires_review=True,
                        )
                        timed_out += 1
                    pending.clear()
                    break

                for future in done:
                    pending.discard(future)
                    agent_id = futures[future]
                    try:
                        result = future.result()
                        if not isinstance(result, AgentResult):
                            raise TypeError("handler must return AgentResult")
                        if result.task_id != state.task.task_id or result.agent_id != agent_id:
                            raise ValueError("agent result identity does not match dispatched task")
                        if len(result.result) > output_limit:
                            result = AgentResult(
                                task_id=result.task_id, agent_id=result.agent_id,
                                status=TaskStatus.NEEDS_REVIEW,
                                result="Agent output exceeded the character budget.",
                                evidence_refs=result.evidence_refs,
                                limitations=result.limitations + (f"output_chars={len(result.result)}; limit={output_limit}",),
                                requires_review=True,
                            )
                        results[agent_id] = result
                    except Exception as exc:
                        results[agent_id] = AgentResult(
                            task_id=state.task.task_id, agent_id=agent_id,
                            status=TaskStatus.FAILED,
                            result=f"Agent handler failed: {type(exc).__name__}",
                            limitations=("Handler exception details are omitted from user-facing state.",),
                            requires_review=True,
                        )
        finally:
            # Do not block the caller after the global deadline. Cooperative
            # handlers should observe their cancellation event and exit promptly.
            pool.shutdown(wait=False, cancel_futures=True)

        state.findings = [results[agent_id] for agent_id in selected if agent_id in results]
        completed = sum(item.status == TaskStatus.COMPLETED and not item.requires_review for item in state.findings)
        failed = sum(item.status == TaskStatus.FAILED for item in state.findings)
        timed_out_count = sum(
            item.status == TaskStatus.NEEDS_REVIEW and "timed out" in item.result.lower()
            for item in state.findings
        )
        output_chars = sum(len(item.result) for item in state.findings)
        state.resource_usage = ResourceUsage(
            agent_calls=len(submitted),
            completed_agents=completed,
            failed_agents=failed,
            timed_out_agents=timed_out_count,
            input_chars_charged=charged_input,
            output_chars=output_chars,
            elapsed_seconds=round(time.monotonic() - started, 6),
            budget_profile=state.task.budget_profile,
        )
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
