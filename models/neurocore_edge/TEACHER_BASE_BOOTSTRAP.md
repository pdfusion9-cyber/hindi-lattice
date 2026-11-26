# Full NeuroCore base bootstrap from frozen open-weight teachers

## Two-stage weight plan

1. **Base bootstrap:** load a pinned external teacher in its own architecture and tokenizer; freeze it; train the complete native NeuroCore student (shared backbone, router and sparse experts) on teacher-generated targets using the NeuroCore tokenizer. This gives the student an initial trained base rather than trying to specialize experts against a random backbone.
2. **Twenty fixed expert specialists:** after the base is independently validated and certified, runner/expert IDs 0–19 remain permanently mapped to their existing experts. Each specialist trains only its assigned expert, starting from the certified common base.

The base bootstrap does not change the 20-runner mapping. It is a separate prerequisite stage.

## Safety against premature expert training

The bootstrap checkpoint is saved with status `BASE_DISTILLATION_IN_PROGRESS` and provenance explicitly says `expert_workers_may_consume=false`. It is intentionally incompatible with the expert launcher, which requires a separately certified `TRAINED_BASE_CHECKPOINT_AVAILABLE` checkpoint. No automatic promotion occurs merely because training steps completed.

## Current implementation

- `teacher_base_distillation.py`: frozen teacher, full-student causal sequence distillation, router auxiliary loss, gradient checks, resumable-ready training state and provisional checkpoint.
- `test_teacher_base_distillation.py`: tests full-student updates and teacher freezing on a tiny instance of the canonical model.
- Existing `teacher_weight_assimilation.py` and `train_teacher_assimilated_expert.py` remain the second-stage specialist pipeline.

## Operational constraints

The full canonical model is about 3.075B parameters. BF16 raw weights alone are about 6.15 GB decimal; optimizer states, gradients, activations, temporary attention tensors and teacher weights require substantially more memory. The current bootstrap trainer is a correctness/reference implementation, not yet a distributed/FSDP production launcher. Do not run the full-size student on an undersized host. A real run additionally needs a pinned teacher revision, approved prompt data, and the approved NeuroCore tokenizer matching the model vocabulary.

No base checkpoint is produced by CI; CI tests only a tiny configuration. No real model weights are downloaded or trained by these tests.
