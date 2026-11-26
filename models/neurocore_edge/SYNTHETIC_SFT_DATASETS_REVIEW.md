# Synthetic Hindi SFT pilot datasets — pending review

## Inventory
- Expert 16 (self-correction): 14 training and 4 held-out validation examples.
- Expert 17 (planning): 15 training and 4 held-out validation examples.
- Total: 29 training and 8 held-out validation examples.

## Origin and format
These examples are newly authored synthetic Hindi prompt/response pairs for the NeuroCore Dhīr-2B adapter project. They are not copied from IndicCorpV2, Sangraha, or another third-party corpus. Each JSONL row has exactly two fields: prompt and response. Use the pinned Dhīr native chat template and supervise assistant-completion tokens only.

This remains a small development seed, not a representative or production-ready corpus. Keep validation rows out of training.

## Mandatory review
**PENDING HUMAN REVIEW — NO TRAINING AUTHORIZED.** Review factual accuracy, Hindi quality, task fit, duplication, safety, bias, completeness, and planning assumptions. Approve exact source-file SHA-256, prepared corpus SHA-256, and saved tokenizer SHA-256 separately. Human reviewer identity, review date, and rights/licensing decisions are required. Synthetic origin does not automatically approve commercial training or public redistribution.

## Rubric
- Expert 16: identifies its own prior error, shows the corrected reasoning, gives an accurate result, and avoids introducing a new misconception.
- Expert 17: states a goal and time bounds, names accountable roles and dependencies, measures outcomes, and includes realistic risks and fallback actions.

## Expansion notes
The new Expert 16 training rows explicitly include a prior assistant answer and a correction, while its new validation rows use unseen arithmetic/unit-conversion prompts. The new Expert 17 rows use consistent rubric fields for roles, resources/dependencies, indicators, risks, and fallback actions. The expanded examples are still synthetic drafts and require independent human review.
