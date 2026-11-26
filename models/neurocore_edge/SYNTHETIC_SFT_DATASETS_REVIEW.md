# Synthetic Hindi SFT pilot datasets — pending review

## Inventory
- Expert 16 (self-correction): 8 training and 2 held-out validation examples.
- Expert 17 (planning): 9 training and 2 held-out validation examples.

## Origin and format
These examples are newly authored synthetic Hindi prompt/response pairs for the NeuroCore Dhīr-2B adapter project. They are not copied from IndicCorpV2, Sangraha, or another third-party corpus. Each JSONL row has exactly two fields: prompt and response. Use the pinned Dhīr native chat template and supervise assistant-completion tokens only.

This is a small development seed, not a representative or production-ready corpus. Keep validation rows out of training.

## Mandatory review
**PENDING HUMAN REVIEW — NO TRAINING AUTHORIZED.** Review factual accuracy, Hindi quality, task fit, duplication, safety, bias, completeness, and planning assumptions. Approve exact source-file SHA-256, prepared corpus SHA-256, and saved tokenizer SHA-256 separately. Human reviewer identity, review date, and rights/licensing decisions are required. Synthetic origin does not automatically approve commercial training or public redistribution.

## Rubric
- Expert 16: identifies the incorrect claim, gives an accurate correction, reasons consistently, and avoids introducing a new misconception.
- Expert 17: states a goal, orders and time-bounds steps, identifies responsibilities/resources, measures outcomes, and includes realistic risks and fallback actions.
