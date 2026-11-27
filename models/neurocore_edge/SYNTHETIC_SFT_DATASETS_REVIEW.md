# Synthetic Hindi SFT pilot datasets — pending independent review

## Inventory
- Expert 16 (self-correction): 20 training and 4 held-out validation examples.
- Expert 17 (planning): 15 training and 4 held-out validation examples.
- Total: 35 training and 8 held-out validation examples.

## Origin and format
These are newly authored synthetic Hindi prompt/response pairs for the NeuroCore Dhīr-2B adapter project. They are not copied from IndicCorpV2, Sangraha, or another third-party corpus. Each JSONL row has exactly two fields: prompt and response. Training uses the pinned Dhīr native chat template and assistant-completion-only supervision.

This is a small development seed, not a representative or production-ready corpus. Validation rows must remain outside training.

## Changes from the previous draft
- Expert 16 has 20 training prompts: the original 14 correction cases plus six calibration examples covering correct-answer retention, insufficient information, and partial correction. The four held-out prompts remain unchanged and held out: they test retaining a correct prior answer, admitting an unsupported claim without the required map scale, and two arithmetic/unit corrections.
- Expert 17's nine foundational training plans and two foundational validation plans now use a common structure: goal, duration, accountable role, resources/dependencies, steps, indicator, risk and fallback.
- The planting plan schedules planting on days 11–15 and measures survival 30 and 60 days after planting, within the 90-day project window.
- No dataset approval, license, privacy, consent, corpus-hash or tokenizer-hash fields were changed.


## Expert 16 calibration expansion — 2026-11-27
- Added two examples where the prior answer is correct and should be retained.
- Added two examples where the prior claim is unsupported and the model must request missing evidence rather than invent a number.
- Added two examples where one part of the prior answer is correct and should be preserved while only the faulty part is corrected.
- Expert 16 is now 20 train / 4 validation; Expert 17 remains 15 train / 4 validation.
- The six new rows are repository-authored synthetic examples. They have not received independent factual, legal, privacy, consent or rights review.
- Both registry entries are pinned to source commit `e1b7824cccdc32004f4cfcd5acf91a2c76acfec9` for provenance only. Approval fields remain false and no tokenizer or prepared-corpus artifact is approved.
- The previous Dhīr tokenizer dry-run report covered the 29-row training corpus before this expansion. Repeat the dry-run and review its new exact hashes before any training decision.

## Review requirements
**PENDING INDEPENDENT HUMAN REVIEW — NO TRAINING AUTHORIZED.** A reviewer must inspect every row for factual correctness, Hindi quality, task fit, duplication, safety, bias, completeness and planning assumptions. Exact source-file SHA-256, prepared corpus SHA-256 and saved tokenizer SHA-256 need separate review and approval. Reviewer identity/date and rights-holder licensing decisions must be recorded. Synthetic authorship alone does not establish commercial-training or public-redistribution permission.

## Rubric
- Expert 16: checks what was previously said, re-evaluates it, corrects a real error with steps, retains a correct answer when warranted, and states when available information is insufficient.
- Expert 17: states a goal and time bounds, names accountable roles and dependencies, measures outcomes, and includes realistic risks and fallback actions.

## Known limits
- The validation set is only four examples per expert and cannot provide a statistically reliable quality estimate.
- The examples are authored seed material and can overfit rapidly if repeatedly optimized.
- Pattern-based PII scans do not constitute comprehensive privacy review.
- The repository has no root LICENSE file; the synthetic registry entries still have no license ID or license terms URL. Commercial-training and public-redistribution permissions must remain false until the rights holder makes and records an explicit decision.
- A model-assisted review is not the independent human approval required by the training gate.

## Approval registry linkage
The two synthetic datasets remain registered against their source-data commit for provenance only. Their approval/review flags remain false, and no approved training artifact or tokenizer hash has been added. The trainer must continue to reject both entries until all required human and exact-hash approvals are separately recorded.
