# Integrated Native Word + Grouped-Answer Training

## Scope

This pipeline trains the repository's native NeuroCore reference model on the
Gyaan subject-relation-object corpus. One training example is one unique
(subject, relation) pair. Its target is the complete, normalized answer list:
whole-word tokens form each answer, `<ALT>` separates answers, and `<EOS>`
ends the list.

This means the same causal next-token objective learns both:
- next whole-word prediction inside answer text; and
- continuation across multiple answers and termination of the complete answer set.

It is not a separate pretrained language-model stage and does not import any
external model weights or tokenizer.

## Training control

Before the optimizer is created, the trainer calls
`MicroAgentTrainingController`. The three existing micro-agents:
- check corpus provenance and scope;
- allocate a bounded step/CPU/memory budget; and
- validate native-model checkpoint provenance.

The Gyaan corpus audit currently says source ownership/license are unverified,
privacy is not formally cleared, and production approval is false. The new
internal-only gate permits only a bounded experiment with no public artifacts.
It does **not** approve production training or change
`models/neurocore_edge/config.json: training.enabled=false`.

The current controller has a hard cap of 200 optimizer steps. This first run is
a pipeline smoke experiment, not convergence training or a full-model run.
The trainer writes an atomic, resumable checkpoint every 50 steps by default.
Resume validates model config, corpus Git blob and SHA-256, tokenizer hash,
training settings, and internal-only provenance before restoring model,
optimizer, and random-number-generator states. `--steps` remains the total
target step count, not the number of additional steps.

## Run locally only

Use a private, access-controlled machine with PyTorch installed. Do not run this
unreviewed corpus experiment on a public runner, publish its checkpoint, upload
the report as a public artifact, or distribute derived model weights.

From the repository root:

```bash
python -m py_compile \
  tools/neurocore_native_grouped_answer_word_prediction.py \
  models/neurocore_edge/training_micro_agents.py

python -m pytest -q \
  models/neurocore_edge/test_training_micro_agents.py \
  models/neurocore_edge/test_grouped_answer_word_prediction.py

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python tools/neurocore_native_grouped_answer_word_prediction.py \
  --corpus data/gyaan_base.txt \
  --steps 200 \
  --batch-size 8 \
  --cpu-threads 2 \
  --learning-rate 0.00015 \
  --checkpoint-every 50 \
  --generation-limit 96 \
  --report reports/neurocore-integrated-word-grouped-200.json \
  --checkpoint artifacts/neurocore-integrated-word-grouped-200.pt \
  --tokenizer artifacts/neurocore-integrated-word-tokenizer.json
```

The script checks the exact corpus Git blob SHA before training. It performs a
deterministic subject-relation group split, so no answer from a pair can leak
between train and validation.

To continue a run from its last local checkpoint, retain the same corpus,
training arguments, tokenizer, and model settings, increase the total target
steps (never above the controller's 200-step cap), and run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python tools/neurocore_native_grouped_answer_word_prediction.py \
  --corpus data/gyaan_base.txt \
  --steps 200 \
  --batch-size 8 \
  --cpu-threads 2 \
  --learning-rate 0.00015 \
  --resume-checkpoint artifacts/neurocore-integrated-word-grouped-200.pt \
  --checkpoint artifacts/neurocore-integrated-word-grouped-200.pt \
  --report reports/neurocore-integrated-word-grouped-200-resumed.json
```

The current 200-step hard cap means resuming a completed 200-step run is
intentionally rejected; a later approved change to the controller budget is
needed before extending beyond this smoke experiment.

## Required evaluation

Review both objectives, not loss alone:
- `next_word_top1_accuracy` and `next_word_top5_accuracy`;
- `answer_set_exact_match`;
- `answer_set_precision`, `answer_set_recall`, and `answer_set_f1`;
- initial-versus-final validation loss and the validation curve;
- sample generated answer lists and the corpus audit.

A successful process exit only means the bounded experiment completed and wrote
its report/checkpoint. It does not mean factual correctness, production
readiness, corpus-rights clearance, or successful training of the 2.87B model.

## Explicit non-goals

- No external/open-source model weights.
- No change to production training gates.
- No public GitHub Actions training run or artifact upload for this corpus.
- No claim that the current 64-hidden, 3-layer experiment is the full 2.87B
  architecture.
