# External open-weight adoption for NeuroCore / Fabric Swarm

## Decision

Adopt strong Indian and international pretrained models as **frozen external specialist backends** first. Do not directly transplant or average their tensors into NeuroCore: architectures, hidden dimensions, vocabularies, tokenizers, positional schemes, MoE layouts and training objectives differ. The supported bridge is:

1. preserve each upstream model and tokenizer in its native runtime;
2. give each NeuroCore expert a fixed worker ID and task-specific adapter;
3. route task requests to the assigned frozen specialist;
4. evaluate specialists independently against the same held-out suite;
5. optionally distil approved teacher outputs into native NeuroCore expert shards later.

This is an adoption/inference integration, not a claim that the external weights have already been downloaded or trained into NeuroCore.

## Model registry

See `external_weight_registry.json` for 20 fixed expert assignments, candidate model repositories, license notes, modality, runtime, and access status. Model weights are not committed to Git; fetch only immutable upstream commit revisions into a private model cache.

Primary Indian candidates:
- Sarvam-30B (Apache-2.0): multilingual Indian-language generalist and MoE.
- Sarvam-105B-1711 (Apache-2.0): high-capability agent/teacher candidate; its published NVFP4 build is hardware-specific and not appropriate for ordinary CPU runners.
- Dhīr-5B-A1.7B and Dhīr-2B (Apache-2.0): Indic language and smaller edge/gating specialists.
- AI4Bharat IndicTrans2 (MIT): translation specialist; Hugging Face access terms must be accepted.
- AI4Bharat Indic Parler-TTS (Apache-2.0): speech-generation sidecar; separate audio runtime.

International candidates:
- Qwen3.5-35B-A3B (Apache-2.0): vision/document QA, tables and multimodal tasks.
- GPT-OSS-20B and GPT-OSS-120B (Apache-2.0): reasoning, critique, tool-use and technical tasks; 120B needs substantial accelerator memory.
- Mistral Small 4 (Apache-2.0): optional high-memory multimodal/agent alternative.

Conditional candidates:
- Bodhan Indic-Translate and Indic-Speak use the Indic Open Model License and access gates. Do not enable until the project accepts the exact terms, access requirements and derivative/hosting conditions.

## 20 fixed workers

Expert IDs 0..19 are mapped in the registry. A worker must never rotate to another expert. External backend parameters remain frozen. Model/tokenizer revisions must be immutable commit hashes, and all weights stay outside this public Git repository.

## Runtime and hardware

A 15-24 GiB GPU cannot serve every model listed here. The registry intentionally separates small, medium and large models. Use the small Indic specialists where the worker hardware allows; use larger models only on suitable GPU hosts or as separately provisioned teacher endpoints. Do not claim 20 simultaneous model instances are loaded until a hardware inventory and VRAM admission test proves it.

## Distillation path

After inference parity and task evaluation:
- capture permitted teacher demonstrations with source and license provenance;
- remove PII and enforce consent/redistribution rules;
- keep fixed train/validation/test splits;
- distil one expert at a time on its permanently assigned runner;
- save base/tokenizer/model/optimizer/RNG/data-cursor fingerprints;
- merge only 20 compatible native NeuroCore expert shards from the same base fingerprint;
- compare native model before/after with held-out metrics.

Do not distil teacher outputs if their source terms prohibit the intended use or if evaluation contamination cannot be ruled out.

## Verification

Run:
`python -m unittest models/neurocore_edge/test_external_expert_adapter.py`

The test uses a fake backend and verifies fixed expert routing only. It does not download models or claim model quality.
