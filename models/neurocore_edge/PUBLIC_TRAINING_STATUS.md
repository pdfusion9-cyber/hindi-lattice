# Public runner and training status

- Repository visibility: public.
- Runner target: GitHub-hosted `ubuntu-24.04` standard runners only; no private/self-hosted runner.
- Public runner preflight: **20/20 matrix jobs passed** in GitHub Actions run `45034271836` on 2026-11-22. This verified standard CPU runner capacity and a tiny CPU BF16 backward/Adafactor step on each job; it is not a full-model training benchmark.
- The prior `main` reference-model test run `45034271886` failed. Do not treat the earlier successful run `45031135549` as validation of the later source revision.
- Local v0.3.3 package: ZIP SHA256 `92309fbc02b7d35118a171b65caad3d813684de8b47787dbbf91136ce9077bb9`; archive size 93,356 bytes. Extracted package test command `python -m pytest -q test_*.py` passed **27/27** in the local validation environment; Python compileall passed.
- Important deployment status: v0.3.3 has **not** yet been committed to `main` and its packaged workflows have not run on GitHub Actions. Do not describe it as deployed or GitHub-verified.
- Canonical architecture candidate: v0.3.3 package selects the PyTorch v0.3.2 architecture family with 3,074,885,656 parameters, 872,876,056 shared parameters, and 2,202,009,600 expert parameters. It supersedes the incompatible `main` scaffold count of 2,869,448,704; this is a source/config replacement, not a checkpoint conversion. No trained checkpoint is included.
- The local tiny end-to-end synthetic test covers a shared base, 20 expert shards, strict merge and router calibration. It does not establish full-size training convergence, CPU throughput or production accuracy.
- **20-independent-runner orchestration verified on GitHub Actions run #34** (2026-11-24, commit `d3798f15a686`): the workflow created one native synthetic shared-base artifact, dispatched matrix jobs for expert IDs 0..19, and all 20 expert jobs completed successfully. The strict merge job verified exact expert coverage, config/base fingerprints and unchanged parameter count (134,224 before and after). The merged model artifact was uploaded as `neurocore-native-20-runner-merged-synthetic` (artifact ID `14899576096`). Run URL: https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45348209039.
- This run is a **tiny synthetic orchestration proof**, not production learning: the tested model used vocab 256, hidden size 32, two layers and 20 experts; one shared-base step and one step per expert. It used no external data or model weights. No 2.87B checkpoint was produced.
- The same run's production readiness gate remained **CLOSED**: `training.enabled=false`, all 27 dataset assignment rows pending review, the full model remains `UNTRAINED_INITIALIZATION_ONLY`, and the standard runner reported 4 CPUs, no CUDA GPU, and about 13.39 GB available host RAM. Full-scale training is not dispatched to these CPU-only standard runners.
- GitHub's standard public Ubuntu runner specification is 4 vCPU / 16 GB RAM; GitHub's larger-runner documentation lists a GPU option with a Tesla T4 and 16 GB VRAM. Larger runners require an organization on GitHub Team or Enterprise Cloud; usage is billed. See https://docs.github.com/en/actions/reference/runners/github-hosted-runners and https://docs.github.com/en/actions/concepts/runners/larger-runners.
- Dataset approval: `models/neurocore_edge/dataset_approval.json` is default-deny with status `ALL_DATASETS_PENDING_REVIEW`. Every dataset needs immutable revision/config/splits and documented license, commercial-training, redistribution, PII, consent and schema review. No dataset is approved for commercial training yet.
- Production training remains **disabled** until one canonical full-model architecture is committed and GitHub-tested, all training dataset assignments are approved, and a production-capable runner profile is provisioned and verified. The tiny 20-runner synthetic integration is already GitHub-verified but does not satisfy these production gates.
- Target after all gates: standard public `ubuntu-24.04` CPU runners; one shared-base stage followed by 20 disjoint expert shards; 280-minute bounded sessions; checkpoint every 250 optimizer steps and at session boundary; strict merge requires exact tensor-key/shape/parameter-count preservation. Public release assets are publicly downloadable.
- Mobile-size reality: 1.58-bit raw arithmetic estimate is 607.29 MB before scales, metadata, tokenizer and runtime. 580 MB is a stretch target, not an achieved artifact. BF16 model storage is about 6.15 GB.
- Known incomplete production components are listed in `BLUEPRINT_TRACEABILITY.md`; the package is an untrained reference implementation, not a production document agent or mobile-ready model.


## CPU-first micro-agent work in PR #9 (2026-11-24)

- Added cpu_training_agents.py: cgroup-aware CPU/RAM/disk profiling, CPU thread policy, conservative FP32 AdamW memory estimates, micro-batch/gradient-accumulation planning, checkpoint interval planning, and 20-expert runner-slot assignment.
- Added compute_efficiency_agents.py: bounded 20-domain scheduling, layer-aware activation-memory proxy for adaptive micro-batches, and a bounded checkpoint planner. This remains a planning layer; it does not yet dispatch production expert training.
- Connected the micro-agent training controller to actual runner CPU capacity. A bounded 1/2/4-thread CPU matmul proxy calibration selects a thread count, and the tiny synthetic training loop now uses that selection and reports elapsed seconds and tokens/second.
- Added a 20-runner CPU inventory workflow plus a verifier requiring 20 distinct hostnames and complete runner IDs 0..19.
- The planner now blocks overall CPU production readiness when the shared-backbone AdamW estimate does not fit one runner's safe RAM budget, even if selected-expert shards appear to fit.
- NOT YET CI-VERIFIED: latest CPU-first branch commit 536e2e60fd89d3018f091c49f2fb981354b9f15 has no workflow run/status returned by the available GitHub run/status lookup at the time of this update. Do not call the tests passed or claim a CPU speedup.
- Important limitation: the active branch config is 0.2.0-pytorch-reference with 2,869,448,704 parameters and does not declare the shared/expert parameter split cited by an earlier v0.3.3 package candidate. Reconcile one canonical architecture before using parameter-count inputs for production planning.
- Training remains synthetic-only in the pilot; production training.enabled remains false, all dataset approvals remain pending, and no production checkpoint or production training run has been created.


## CPU inventory preflight diagnosis (2026-11-24)

- GitHub Actions run #45352812975 reached all 20 CPU inventory jobs. Every observed runner successfully executed CPU profiling; the profile example reported 4 effective CPUs, 14.646 GiB available memory, and 85.147 GiB free disk.
- The inventory job uploads failed because literal backslashes had been committed before GitHub Actions matrix expressions, producing invalid artifact names such as neurocore-cpu-runner-\\11. The profile step itself succeeded; the failure occurred at upload-artifact.
- Fixed the workflow by removing escaped expression markers and using a single-line profile command with a RUNNER_ID environment variable. The verifier still requires exactly IDs 0..19 and 20 distinct hostnames.
- CPU planner tests passed in run #45352812975; the independent public CI run #45352812840 passed its then-current CPU planner and reference suites. The compute-efficiency tests were added after that run, so they are not yet certified by that result.
- Latest workflow/test corrections are committed on the PR branch, but a post-fix Actions run has not yet appeared. Inventory completion and distinct-host verification therefore remain pending.
- The expert-domain scheduler is explicitly an equal-visit scheduler ordered by observed routing counts; it is not yet a measured CPU-throughput optimizer. Production training remains disabled.


## Runner identity verification correction (2026-11-24)

- The previous 20-job inventory did execute all 20 profiling steps, but log inspection found repeated hostnames across matrix jobs. The old hostname-uniqueness verifier was therefore too strict for this hosted environment and could not establish distinct VM identity from hostname alone.
- CPUProfile now records GitHub runner metadata (`RUNNER_NAME`, `RUNNER_OS`, `RUNNER_ARCH`, `GITHUB_JOB`, `GITHUB_RUN_ID`) and an optional SHA-256 machine fingerprint derived from DMI product UUID or `/etc/machine-id`; raw machine identifiers are not stored.
- Replaced hostname-based acceptance with unique GitHub `RUNNER_NAME` checks, exact runner IDs 0..19, complete CPU/RAM/disk measurements, and one common workflow run ID. Hostname is diagnostic only; machine fingerprints are reported when available.
- Added verifier tests for repeated hostnames, duplicate runner names, and mixed workflow run IDs. Added those tests to the CPU preflight and public CI workflows.
- The first Actions run after adding the verifier failed before jobs were created because a path-filter line contained a literal escaped newline. Fixed the YAML path list in the next commit and re-read the complete workflow file. No post-fix Actions result is available yet; do not claim the corrected inventory passes.
- Current production gate remains closed. No production training was dispatched.
