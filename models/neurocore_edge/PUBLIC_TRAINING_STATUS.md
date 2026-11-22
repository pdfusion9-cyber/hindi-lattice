# Public runner and training status

- Repository visibility: public.
- Runner target: GitHub-hosted `ubuntu-24.04` standard runners only; no private/self-hosted runner.
- Public runner preflight: **20/20 matrix jobs passed** in GitHub Actions run `45034271836` on 2026-11-22. This verified standard CPU runner capacity and a tiny CPU BF16 backward/Adafactor step on each job; it is not a full-model training benchmark.
- The prior `main` reference-model test run `45034271886` failed. Do not treat the earlier successful run `45031135549` as validation of the later source revision.
- Local v0.3.3 package: ZIP SHA256 `92309fbc02b7d35118a171b65caad3d813684de8b47787dbbf91136ce9077bb9`; archive size 93,356 bytes. Extracted package test command `python -m pytest -q test_*.py` passed **27/27** in the local validation environment; Python compileall passed.
- Important deployment status: v0.3.3 has **not** yet been committed to `main` and its packaged workflows have not run on GitHub Actions. Do not describe it as deployed or GitHub-verified.
- Canonical architecture candidate: v0.3.3 package selects the PyTorch v0.3.2 architecture family with 3,074,885,656 parameters, 872,876,056 shared parameters, and 2,202,009,600 expert parameters. It supersedes the incompatible `main` scaffold count of 2,869,448,704; this is a source/config replacement, not a checkpoint conversion. No trained checkpoint is included.
- The local tiny end-to-end synthetic test covers a shared base, 20 expert shards, strict merge and router calibration. It does not establish full-size training convergence, CPU throughput or production accuracy.
- Dataset approval: `models/neurocore_edge/dataset_approval.json` is default-deny with status `ALL_DATASETS_PENDING_REVIEW`. Every dataset needs immutable revision/config/splits and documented license, commercial-training, redistribution, PII, consent and schema review. No dataset is approved for commercial training yet.
- Training remains **disabled** until v0.3.3 is committed and GitHub-tested, dataset approvals are complete, and a bounded public CPU dry run passes.
- Target after all gates: standard public `ubuntu-24.04` CPU runners; one shared-base stage followed by 20 disjoint expert shards; 280-minute bounded sessions; checkpoint every 250 optimizer steps and at session boundary; strict merge requires exact tensor-key/shape/parameter-count preservation. Public release assets are publicly downloadable.
- Mobile-size reality: 1.58-bit raw arithmetic estimate is 607.29 MB before scales, metadata, tokenizer and runtime. 580 MB is a stretch target, not an achieved artifact. BF16 model storage is about 6.15 GB.
- Known incomplete production components are listed in `BLUEPRINT_TRACEABILITY.md`; the package is an untrained reference implementation, not a production document agent or mobile-ready model.
