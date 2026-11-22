# NeuroCore-Edge (scaffold only)

Status: **UNTRAINED / NO CHECKPOINT / TRAINING DISABLED**.

This directory defines a mobile-first sparse MoE reference architecture for the
Fabric Swarm document-processing application. It is not yet a usable assistant.

## Architecture target

- 20 domain experts, BG-Gate top-2 routing per token.
- Shared hidden size 2048, 24 layers, 16 query heads / 4 KV heads.
- Expert gated MLPs with ReLU^2; target ternary 1.58-bit weights and INT8 activations.
- Bounded recurrent working-memory reference block; production Gated-DeltaNet/Mamba,
  sliding-window attention, SnapKV, and device-specific NPU kernels remain TODO.
- Mobile runtime targets: Android NNAPI/vendor NPU, iOS Core ML, and CPU fallback.
- Dataset-to-expert mapping: `expert_manifest.json`.

## Important size reality check

The 580 MB target is not guaranteed. At 2.85B parameters, raw 1.58-bit packing
alone is about 562.9 MB (decimal), before scale metadata, embeddings, tokenizer,
routing tables, alignment, and runtime. The uploaded NeuroCore-Edge report itself
estimates a 0.68-0.75 GB weight footprint for its 3B MoE reference model (page 6).
The config records 580 MB as an aspiration, not a verified deliverable.

The reference model code uses ordinary PyTorch tensors. It does not implement
native ternary packing, quantization kernels, production attention, or mobile
export yet. A small CPU smoke test is provided; do not instantiate the 3B
configuration on a phone until the packed runtime exists.

## 20-runner training plan (not enabled)

1. Each runner streams only its assigned expert-domain data and writes a shard
   with a common tokenizer/config fingerprint.
2. Shared-backbone updates must be coordinated; independent expert runs alone
   do not train a coherent 3B MoE.
3. Merge only compatible shards after schema, tokenizer, tensor-shape, license,
   and validation checks.
4. Run staged distillation, calibration, INT8 activation calibration, ternary
   weight conversion, then Android/iOS export.
5. Keep optimizer state and raw training data off the mobile artifact.

GitHub-hosted runners are not guaranteed to provide GPUs or enough RAM for this
model. A runner matrix is orchestration, not a promise of 20 concurrent GPU
machines. Confirm actual runner hardware/quota before any training is enabled.

## Safety and data governance

Dataset names are proposed sources from the project brief, not a verified
license/compliance clearance. Verify each dataset's license, terms, provenance,
PII policy, and jurisdictional suitability before streaming. Legal outputs must
be labeled as drafting/information, not legal advice; jurisdiction and source
citations must be explicit. TTS requires separate speech/audio model assets.

## Smoke test

```bash
python -m pip install torch
python models/neurocore_edge/modeling_neurocore_edge.py
```
