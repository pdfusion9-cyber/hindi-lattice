# Anviksha AI / NeuroCore Hybrid Micro-Agent Architecture v0.1

**Status:** deterministic control-plane prototype. Not a trained multi-agent system,
production runtime, or demonstrated reasoning improvement.

## Decision

- Organization: **Hybrid** — specialist domain agents coordinated by one orchestrator.
- Reasoning budget: **Balanced** — bounded agent activation and reasoning modes.
- Domain specialization and reasoning behavior are separate concepts.
- Existing token-level NeuroCore MoE (20 experts, BG-Gate top-2) remains in the model
  plane; micro-agents live in the control plane.

## Layers

1. User task intake.
2. Orchestrator: capability routing, task identity and bounded dispatch.
3. Domain registry: Hindi, research, document/OCR, legal, planning and QA contracts.
4. Reasoning controller: Normal (N), Deep (D), Explore (E), Reflect (R).
5. Evidence verifier: structural reference checks and review flags.
6. Response synthesizer: blocks responses when structural checks fail.

The four modes are behavior policies, not four autonomous model instances.
The paper The Molecular Structure of Thought (arXiv:2601.06002) motivates these
behavior distinctions. This implementation does not reproduce Mole-Syn, learn a
transition matrix, or claim to reproduce the paper's experiments.

## Balanced budget defaults

| Difficulty | Max domain agents | Max explore branches | Max reasoning rounds | Reflection | Verification |
|---|---:|---:|---:|---|---|
| Simple | 1 | 0 | 1 | No | Yes |
| Medium | 2 | 1 | 2 | Yes | Yes |
| Hard | 3 | 2 | 3 | Yes | Yes |

These are initial engineering limits, not empirically optimized values. The reference
runtime now supports bounded parallel dispatch, a deadline, cooperative cancellation,
and per-agent output caps. Input/output character counts are only a proxy until a real
model tokenizer is connected. Overrides may tighten but cannot exceed the balanced
policy ceilings. A Python thread cannot be forcibly terminated; handlers must poll
the execution context and exit cooperatively. The prototype does not yet implement
model-token accounting, retries, external tool execution or persistent memory.

| Difficulty | Max total input chars (fan-out charged) | Max output chars / agent | Deadline | Max workers |
|---|---:|---:|---:|---:|
| Simple | 4,000 | 4,000 | 3 s | 1 |
| Medium | 8,000 | 6,000 | 6 s | 2 |
| Hard | 16,000 | 8,000 | 12 s | 3 |

## Shared contracts

AgentTask carries task ID, input, language, difficulty, requested capabilities,
budget profile and caller context. AgentResult carries task ID, agent ID, status,
result, evidence references, limitations, optional confidence and review requirement.
Mismatched task or agent identities are rejected.

The host must provide explicit handler callables. A role without a registered handler
is reported as needs_review; no answer is fabricated.

## Provider-neutral domain adapters

`agent_llm_adapters.py` defines typed request/reply contracts and adapters for:

- **Hindi:** natural-language response with language, notes, confidence and review metadata.
- **Research:** synthesis strictly against caller-supplied `context.source_material`;
  every finding must cite an ID in that supplied allow-list. Missing sources force
  review; invented IDs are rejected.
- **Planning:** ordered steps, dependencies, risks and success criteria. It creates
  plans only and does not execute tools or external actions.

The host injects a provider callable through `build_domain_handlers(provider)`.
No API key, provider SDK, network client or model endpoint is stored in this repository.
The provider must honor the cooperative execution context. These adapters validate
the returned JSON schema; they do not independently establish factual truth or source
entailment. Fake-provider unit tests run offline in CI; no paid API call is made.

## Verification boundary

EvidenceVerifier checks only that evidence IDs appear in a supplied allow-list and
flags incomplete/review-required results. It does not verify source authenticity,
currentness, legal reusability, or whether a source supports a claim. A production
verifier needs provenance, claim-to-source entailment, privacy controls and
domain-specific validation.

## Memory and privacy

Version 0.1 uses in-memory task state only. There is no persistent or cross-user
memory and no external connector call. A production host must isolate user/task
state, apply retention controls, redact sensitive data and treat retrieved content
as untrusted data rather than executable instructions.

## Tests

Run from this directory:

    python -m unittest -v test_micro_agent_runtime.py
    python -m unittest -v test_agent_llm_adapters.py

Tests cover bounded routing, behavior sequencing, parallel dispatch, cooperative
timeout signalling, input/output budgets, handler failures, identity validation,
evidence-reference presence, structured adapter schemas, supplied-source citation
allow-lists, planning dependencies and conservative synthesis.

## Explicit non-goals

- No production LLM inference or agent model is included.
- No external datasets are read and no training is started.
- No changes to config.json, dataset_approval.json or the 20-expert manifest.
- No mobile deployment or performance claim is made.
- No private chain-of-thought is exposed; interfaces carry task results and
  verification metadata only.

## Next milestones

1. Connect a host-owned provider client and secrets manager; add provider integration tests.
2. Add claim-level evidence verification and adversarial prompt-injection tests.
3. Evaluate on held-out synthetic Hindi tasks against a single-agent baseline.
4. Add measured token/latency accounting using the connected provider's tokenizer and usage metadata.
5. Only after measured benefit, consider learning a transition policy from reviewed
   trajectories. Keep dataset rights approvals separate from synthetic tests.
