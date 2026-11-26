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
## Row-level review findings — 2026-11-27

This section records a further model-assisted review of the four checked-in JSONL files. It is a review aid, not a human sign-off.

### Expert 16 — task identity and validation coverage
- Of 14 training prompts, 6 explicitly describe a prior assistant answer and ask for its correction. The other 8 are useful error-spotting or factual-correction tasks, but do not clearly establish that the model itself previously produced the answer.
- Of 4 validation prompts, 2 explicitly ask the assistant to correct its own prior answer; 2 are general correction/checking questions.
- Recommendation: retain the distinction in reporting. Do not claim all 18 examples are direct self-revision demonstrations. Expand the held-out set with multi-turn examples, including cases where the prior answer is already correct and should be retained, cases with insufficient information, and corrections across varied task families.
- Concept families overlap despite exact prompt disjointness: percentages occur in both train and validation; mass conversion in training and volume conversion in validation share a general unit-conversion skill. The present validation split checks changed prompts and values, not broad unseen-skill generalization.

### Expert 17 — planning rubric and timeline
- Training rows 10–15 are the advanced structured examples and include accountable roles, resources/dependencies, indicators, risks, and fallback actions. Rows 1–9 are shorter foundational plans and do not consistently contain all those fields. The two advanced validation rows cover the full rubric; the two shorter validation rows do not.
- Training row 4 (local-species planting) is internally inconsistent: planting is scheduled for days 31–45, while survival is to be measured at 30/60/90 days. A 90-day post-planting measurement cannot fit within the stated 90-day plan. Revise the schedule or define the measurement dates relative to the project start before this row is used for training.
- Training row 5 (school-roof rainwater collection) calls for a flow test but does not define an explicit pass/fail or measurable outcome indicator. Add a qualified technical acceptance measure only after the relevant site/design requirements are known.
- Training row 8 (pond cleaning) includes sensible expert/permit/safety boundaries but no explicit outcome indicator. Add a measurable, environmentally appropriate completion measure.
- The basic planning rows are uneven in ownership, dependencies, risks and fallback detail. Decide whether these are deliberately foundational tasks or whether every example should meet the complete planning rubric; apply the same decision to validation.
- Validation contains only four examples. The reading-festival and public-tank-maintenance scenarios are relatively brief and do not probe the full structured rubric. They are suitable only for an initial smoke check, not a reliable estimate of planning quality.

### Cross-dataset limitations
- Exact-string train/validation prompt separation passed, but this does not prove semantic independence. Planning validation overlaps some broad themes with training (reading, water services, public safety/accessibility).
- The PII checks are pattern-based filters for common phone/email/identity-number shapes. They are not comprehensive privacy review or proof that a synthetic prompt contains no personal or sensitive information.
- The advanced public-safety, health-camp, emergency-planning and water-quality examples appropriately defer clinical, emergency, engineering, sampling and approval decisions to qualified or authorized personnel. A human reviewer should still verify local role boundaries and feasibility.
- The repository root has no LICENSE, LICENSE.md or COPYING file on this branch, and the two synthetic registry rows have no license ID or license terms URL. Their authored synthetic origin is recorded, but it does not itself establish commercial-training or public-redistribution permission. Keep both permissions false until the rights holder makes and records an explicit decision.

### Training-readiness decision
- The tokenizer dry-run passed against the pinned Dhīr-2B tokenizer and the production trainer's parser. It verified native chat-template prefixes, assistant-completion-only labels, row retention, and SHA-256 values. It did not load model weights, run a forward pass, assess model loss/quality, or train.
- The 29 training examples are a small pilot seed. Repeated optimization on this quantity is likely to overfit; do not treat the dry-run as evidence that a useful expert can be trained or that validation is statistically meaningful.
- No dataset, commercial-use, privacy, consent, corpus-hash or tokenizer-hash approval was granted by this audit. The registry remains default-deny and training remains unauthorized.
