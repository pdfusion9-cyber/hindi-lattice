# NeuroCore cross-model weight assimilation

## Correct target

The foreign model is **not** a runtime replacement for NeuroCore. Its pretrained checkpoint is read through its own original architecture and tokenizer as a frozen teacher. The knowledge encoded in those weights is transferred into NeuroCore by training NeuroCore's own expert parameters. The end product is a native NeuroCore checkpoint; teacher files are not required for ordinary NeuroCore inference after distillation.

## Assimilation stages

1. **Acquire and verify teacher checkpoint** — pin immutable Hugging Face commit SHA; verify the exact model-card license and use terms; record repository, revision, weight file hashes, dtype, architecture and tokenizer hashes. Keep checkpoints in private artifact storage, never in this public Git repo.
2. **Load the teacher's true architecture** — instantiate its native Transformers/model implementation and tokenizer. Freeze all teacher parameters and run inference only.
3. **Choose expert assignment** — runner ID equals expert ID 0–19, fixed for the complete training campaign. Each expert receives only its assigned teacher/task mix and its own permitted prompt/data mixture.
4. **Generate teacher supervision** — each teacher reads the input examples with its own tokenizer and produces targets/soft outputs. For different tokenizers, use sequence/text distillation: tokenize teacher answers with the NeuroCore tokenizer and train the selected NeuroCore expert. When tokenizer and vocab fingerprints are identical, optional logits-level KL distillation is allowed. Do not compare logits from unrelated vocabularies.
5. **Train NeuroCore's own weights** — teacher remains frozen; only selected expert parameters are trainable. The shared backbone, router, other 19 experts and any unassigned modules stay frozen in the independent expert stage. The student forward path must force routing through the worker's expert ID.
6. **Optional exact tensor warm-start** — copy source tensors only when an explicit key mapping, exact tensor shape, compatible semantics and dtype are validated. No flattening, arbitrary reshape, truncation or averaging of unlike model layers.
7. **Checkpoint/resume and merge** — persist student expert weights, optimizer, scheduler, RNG, tokenizer/data/teacher/base fingerprints and data cursor every 250 steps and at the session boundary. Strict merge requires exactly one compatible shard for each expert 0–19 from the same base/config/tokenizer contract.
8. **Evaluate and consolidate** — held-out task success, Hindi/English metrics, factuality, citation precision, contamination checks, latency/memory and safety. Only after a measured pass may teacher supervision be distilled further and teacher endpoints removed from the serving graph.

## Model candidates

The registry lists Indian teachers (Sarvam-30B/105B, Dhīr-5B/2B, IndicTrans2 and Indic Parler-TTS) and international teachers (Qwen3.5-35B-A3B, GPT-OSS-20B/120B and Mistral Small 4) with task-specific fixed assignments, upstream license, access and hardware notes. Candidate models are not automatically approved by being called open weight; each source revision and intended use still needs review.

## Weight compatibility reality

Raw weights are not a text corpus. A GPT-OSS expert matrix is meaningful only inside GPT-OSS's own layer layout; a Qwen tensor cannot be blindly loaded into a Dhīr or NeuroCore parameter. For unrelated architectures, the correct assimilation path is frozen teacher forward passes plus student losses (sequence distillation, optionally feature/logit distillation where dimensions and tokenizers permit). This actually trains NeuroCore's own parameters rather than just routing user requests to an outside model.

## Modalities

Text-only NeuroCore experts can distil text capabilities from vision/audio teachers only after a suitable teacher task produces text supervision. For pixel-level OCR or image understanding, the student needs a compatible vision encoder/projector; for speech synthesis it needs an audio decoder/codec/vocoder student. A text-only MLP expert cannot acquire image/audio input/output capability by copying text-model tensors.

## Current implementation status

- Registry, fixed one-runner-to-one-expert assignments and source/license/hardware records are present.
- `teacher_weight_assimilation.py` implements pinned teacher loading, freezing, teacher-generated sequence targets, selected-expert-only student loss and exact-shape-only tensor copy.
- The trainer deliberately requires a canonical NeuroCore student forward callback that forces the assigned expert ID; it refuses arbitrary model implementations.
- The included tests use a tiny fake teacher/student to verify the assimilation mechanism without downloading large checkpoints.
- This does **not** mean model weights have already been downloaded or the 3.075B campaign has started. Full run needs canonical student architecture/runner wiring, license-cleared datasets, private checkpoint storage and qualified hardware.
