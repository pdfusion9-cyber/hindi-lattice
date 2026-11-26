# Preliminary quality audit — synthetic Hindi SFT pilot

**Status: REVIEW NOTES ONLY — NOT A HUMAN APPROVAL AND NOT TRAINING AUTHORIZATION.**

Audit basis: the four synthetic JSONL files currently in this branch, inspected row-by-row. This is a model-assisted preliminary content review; a designated human reviewer must independently verify every item before approval.

## Inventory
- Expert 16 / self-correction: 8 train, 2 held-out validation.
- Expert 17 / planning: 9 train, 2 held-out validation.
- Train and validation prompt sets are disjoint under exact-string comparison.
- Rows use the two-field `prompt` / `response` schema.

## Preliminary findings

### Expert 16 — self-correction
The visible arithmetic examples recompute correctly: 120 + 18 × 15 − 75 = 315; rectangle area 12 × 5 = 60 m²; 20% of ₹800 = ₹160 and the resulting price is ₹640. The Earth rotation/revolution, seasons, photosynthesis oxygen source, and basic magnetism explanations are broadly consistent with standard science.

Before approval:
- Have a subject-matter reviewer verify scientific wording and ensure explanations are not overgeneralized.
- Add harder multi-step cases where the model must identify its own earlier error, not only correct an error embedded in the prompt.
- Expand validation beyond two examples and include adversarial near-miss cases.

### Expert 17 — planning
The examples generally provide ordered, time-bounded steps and some measurable indicators. Safety-aware details appear in the rainwater, digital-literacy, solar-light, and pond-cleaning examples.

Gaps against the stated planning rubric:
- Responsibility and resource ownership are not explicit in every response.
- Budget, procurement dependencies, permissions, and local feasibility are unevenly covered.
- Risks and fallback actions appear in some examples but are missing or implicit in others.
- Several indicators measure activity (for example, items installed) rather than durable outcomes (for example, functionality or survival after a defined period).

Before approval:
- Revise each planning response to include an accountable role, necessary resources/dependencies, a measurable outcome, at least one material risk, and a realistic fallback.
- Preserve safety constraints and avoid inventing local costs, permits, or technical specifications.
- Add validation scenarios with conflicting constraints, delayed resources, and failed milestones.

## Automated checks and their limits
The CI checks schema, minimum text lengths, Devanagari presence, basic email/phone/Aadhaar-like patterns, representative PII detector behavior, and exact prompt separation between train and validation. These checks do **not** establish factual correctness, comprehensive PII absence, source rights, consent, bias, or production readiness.

## Approval boundary
- No approval fields were changed.
- No corpus or tokenizer SHA-256 has been approved.
- The dataset approval policy remains default-deny.
- No model training or optimizer step was performed.
- Keep validation rows out of training.
