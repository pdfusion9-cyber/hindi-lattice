# Public runner and training status

- Repository visibility: public.
- Runner target: GitHub-hosted `ubuntu-24.04` standard runners only; no private/self-hosted runner.
- Public runner preflight: **20/20 matrix jobs passed** in GitHub Actions run `45025693341` on 2026-11-22. This verified standard CPU runner capacity and a tiny CPU BF16 backward/Adafactor step on each job.
- Canonical reference tests: passed in run `45031135549` on 2026-11-22. This validates the current reference scaffold only.
- Requested session target for future training: 280 minutes; checkpoint cadence target: every 250 optimizer steps plus the session-boundary checkpoint.
- Training: **disabled**. The current `main` branch does not yet contain the full staged trainer, release-backed resumable expert shards, or strict merger needed for the intended continuous 20-expert cycle.
- Dataset approval: `models/neurocore_edge/dataset_approval.json` is default-deny; every HF source needs immutable revision/config/splits and license, commercial training, redistribution, PII, consent and schema review. No dataset is approved for commercial training yet.
- Architecture mismatch: current main config reports 2,869,448,704 parameters; the local v0.3.2 bundle reports 3,074,885,656. **Do not mix their checkpoints.** A canonical config/model decision and parity tests are required before training.
- The local v0.3.2 bundle independently passed 25 Python tests and a meta-device count audit, but it is not yet published as the canonical `main` model. Its meta-device count is not a trained checkpoint.
- Important: passing preflight proves runner availability and basic CPU PyTorch operation only. It does not prove full-model throughput, convergence, professional accuracy, mobile readiness, or 580 MB packaged size.
- Next gate: reconcile the two architecture versions, port/validate one canonical trainer and strict merge path on `main`, complete dataset approvals, then dispatch a small bounded dry run before enabling any continuous training loop.
