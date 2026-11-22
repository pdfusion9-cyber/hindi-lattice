# Public runner and training status

- Repository visibility: public.
- Runner target: GitHub-hosted `ubuntu-24.04` standard runners only; no private/self-hosted runner.
- Preflight: 20-job matrix, CPU-only; measures CPU/RAM/disk and performs a tiny PyTorch backward/optimizer step.
- Requested session target for future training: 280 minutes; checkpoint cadence target: every 250 optimizer steps plus the session-boundary checkpoint.
- Training: **disabled**. This branch's current model scaffold does not yet have the full trainer/strict merger required for a safe 20-expert continuous cycle.
- Dataset approval: `models/neurocore_edge/dataset_approval.json` is default-deny; every HF source needs immutable revision/config/splits and license, commercial training, redistribution, PII, consent and schema review.
- Architecture mismatch: current main config reports 2,869,448,704 parameters; the previously discussed v0.3.2 local bundle reported 3,074,885,656. Do not mix their checkpoints. Reconcile and implement one canonical model before training.
- Important: a passing preflight proves runner availability and basic CPU PyTorch operation only. It does not prove model convergence, professional accuracy or mobile readiness.
