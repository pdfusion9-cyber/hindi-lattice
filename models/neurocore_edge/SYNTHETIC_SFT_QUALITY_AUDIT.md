# Preliminary quality audit — synthetic Hindi SFT pilot

**Status: REVIEW NOTES ONLY — NOT A HUMAN APPROVAL AND NOT TRAINING AUTHORIZATION.**

Audit basis: all four synthetic JSONL files in this branch inspected row-by-row. This is a model-assisted preliminary content review; a designated human reviewer must independently verify every item before approval.

## Inventory
- Expert 16 / self-correction: 14 train, 4 held-out validation.
- Expert 17 / planning: 15 train, 4 held-out validation.
- Total: 29 training and 8 held-out validation examples.
- Train and validation prompt sets are disjoint under exact-string comparison.
- Rows use the two-field `prompt` / `response` schema.

## Preliminary findings

### Expert 16 — self-correction
The initial examples cover arithmetic, geometry, unit concepts, and basic science. The added training examples now explicitly present a previous assistant answer containing an error, then ask for a re-check and correction. They include common denominator addition, speed/time, mass conversion, successive percentage change, mean versus median, and triangle area. The added held-out examples use different percentage and volume-conversion prompts.

These are stronger self-revision demonstrations than simple factual correction, but they remain short synthetic examples. A subject-matter reviewer should check every calculation and ensure the model does not learn to apologize mechanically when its previous answer was already correct. Expand validation with multi-turn cases, ambiguous inputs, and cases where the earlier answer should be retained.

### Expert 17 — planning
The added advanced training rows use explicit fields for accountable role, resources/dependencies, indicators, risks, and fallback actions. Scenarios cover flood preparedness, solar irrigation feasibility, waste separation, school-zone road safety, administrative coordination for a health camp, and authorized water-quality sampling. Held-out scenarios cover a market evacuation drill and bus-stop accessibility.

These examples intentionally avoid inventing costs, permissions, technical specifications, or test results. A human reviewer should verify local feasibility, safe role boundaries, relevant authority approvals, and whether the indicators measure durable outcomes rather than activity alone.

## Automated checks and their limits
The CI checks schema, minimum text lengths, inventory minima, Devanagari presence, basic email/phone/Aadhaar-like patterns, representative PII detector behavior, exact prompt separation between train and validation, explicit prior-answer revision examples for Expert 16, and rubric fields in advanced Expert 17 rows. These checks do **not** establish factual correctness, comprehensive PII absence, source rights, consent, bias, or production readiness.

## Approval boundary
- No approval fields were changed.
- No corpus or tokenizer SHA-256 has been approved.
- The dataset approval policy remains default-deny.
- No model training or optimizer step was performed.
- Keep validation rows out of training.
