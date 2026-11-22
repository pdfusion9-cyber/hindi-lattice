# NeuroCore-Edge blueprint traceability audit

Audit basis: the six-page uploaded *NeuroCore-Edge — मस्तिष्क-प्रेरित ऑफ़लाइन स्मार्टफोन एआई आर्किटेक्चर रिपोर्ट.pdf* and the code currently present on `main` at commit `9eb3c918406c6899a68e6624867c9f083ead1e2c`.

## Critical source-of-truth finding

The current `main` branch is still at architecture version `0.2.0-pytorch-reference`. Its config records **2,869,448,704** parameters and 566.72 MB raw 1.58-bit arithmetic. The previously discussed v0.3.2 local package reported **3,074,885,656** parameters, but that package's complete source was not present in the current Git tree at audit time. These are different architecture snapshots and must not be silently merged or treated as the same checkpoint.

The current model file exposes 24 separately instantiated layers. It does not implement the later claimed 2-prelude + 4 shared core blocks repeated five times + 2-coda tied sandwich structure (8 unique blocks / 24 effective operations). There is also no complete trainer, strict expert merge program, tokenizer pipeline or dataset gate in the current `main` snapshot. Training is therefore kept disabled.

## Page-by-page coverage

| PDF page / topic | Current main-branch status | Missing work / acceptance evidence |
|---|---|---|
| 1 — target 3B sparse document MoE, 20 experts, BG-Gate Top-2, mobile target | Partial reference | Current config has 20 experts and top-2, but its parameter count differs from the later 3.075B design. Reconcile one canonical config and meta-device count before training. |
| 2 — sandwich topology, shared blocks, GQA/KV heads, RoPE, sliding/global attention | Partial / not implemented | Current source has 24 separately instantiated layers and no attention/RoPE implementation. Implement tied sandwich execution and GQA/RoPE attention, then test cache shapes and context masks. |
| 2 — ReLU² experts and recurrent/working memory | Reference implementation | ReLU²-style expert MLP and a bounded recurrent state exist. The current memory cell is not the full Gated-DeltaNet/Mamba production kernel. |
| 3 — Titans fast memory, surprise update, replay/sleep consolidation, DoRA/LoRA | Not implemented / reference only | Implement bounded fast weights, replay dataset, adapter merge and validation gates; demonstrate state mutation and rollback with tests. |
| 3 — SHY / forgetting, privacy, tenant read-write-execute-legal isolation | Not implemented | Build policy-enforced memory deletion and tenant isolation; test that forbidden records cannot be retrieved after deletion. |
| 3 — GIF / sigma-delta / spiking and Hessian outlier handling | Not implemented | Add quantization-aware residual path, outlier calibration and reference-vs-packed parity tests. |
| 4 — SnapKV/KIVI cache selection and 2-bit key/value quantization | Not integrated | Add cache-aware attention implementation; measure perplexity and memory/latency against uncompressed cache. |
| 4 — OCR/PDF layout, tables, visual document understanding | Separate sidecar required | Text-only MoE cannot learn image pixels/layout without a vision encoder, layout targets and OCR evaluation dataset. |
| 4 — multilingual TTS / read-aloud | Separate audio sidecar required | SSML text does not synthesize audio. Add speech encoder/decoder or codec/vocoder, alignment data and voice-consent review. |
| 5 — Coconut latent planning, Huginn loop depth, wavefront/CDB, DoLa, MNL, SCoRe | Not implemented or reference-only | Implement each algorithm in the actual inference/training path and create separate unit and held-out evaluation gates. |
| 5 — legal contracts, compliance, citations, document agents | Task scaffolding only | Add jurisdiction/version provenance, source citations, contract benchmark and legal red-team/abstention evaluation. |
| 6 — W1.58A8 ternary packing, mobile runtimes (Android/iOS/CPU NPU), offline profile | Not implemented | Build native packing kernels, tokenizer/runtime packaging, Android NNAPI/vendor-NPU and iOS Core ML exports; publish measured file size, RAM, latency, energy and thermal tests. |

## Training readiness gates

1. One canonical architecture and config must match exact tensor shapes and the meta-device parameter budget.
2. Implement tokenizer training and deterministic encode/decode round-trip.
3. Implement a shared-base trainer and a worker that trains exactly one expert ID without updating other experts or router weights.
4. Save model weights **and** optimizer state every 250 optimizer steps and at the 280-minute session boundary.
5. Strict merge must reject missing/duplicate expert IDs, base/config/tokenizer hash mismatch, key/shape mismatch and non-finite weights. Parameter-count delta must be exactly zero.
6. Keep a fixed held-out validation split across resumptions; only explicitly approved training snapshots may rotate.
7. Run a tiny synthetic end-to-end base → 20 expert shards → merge → router-calibration test before any 3B training.
8. Only after real public-runner preflight, dataset rights review and held-out quality gates may continuous training be enabled.

**Status:** audit committed; this is not a trained model and this commit does not start training.
