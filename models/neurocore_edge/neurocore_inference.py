"""One-call end-to-end inference runner for the native Dhīr-NeuroCore model.

This composes the existing NeuroCore tokenizer metadata, task-to-expert mapping,
fixed-expert adapters, native Top-2 path and autoregressive generation.
It performs inference only; it does not train or calibrate experts/router.
"""
from __future__ import annotations

import time
from typing import Any


VALID_MODES = {"task_expert", "top2", "base_dhir"}


def run_neurocore_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    task: str,
    language_tag: str,
    mode: str = "task_expert",
    max_new_tokens: int = 32,
    do_sample: bool = False,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
    if not 1 <= int(max_new_tokens) <= 128:
        raise ValueError("max_new_tokens must be between 1 and 128")

    encoded = tokenizer.encode_with_metadata(
        prompt,
        role="user",
        task=task,
        language_tag=language_tag,
        return_tensors="pt",
    )
    inputs = encoded["encoding"]
    control = encoded["control_metadata"]
    text_meta = encoded["text_metadata"]
    suggested_expert = control.get("suggested_expert_id")

    if mode == "task_expert":
        if suggested_expert is None:
            raise ValueError("task has no registered fixed-expert mapping")
        generate_fn = lambda: model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            expert_id=int(suggested_expert),
        )
        selected_expert = int(suggested_expert)
    elif mode == "top2":
        generate_fn = lambda: model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        selected_expert = None
    else:
        generate_base = getattr(model, "generate_base", None)
        if not callable(generate_base):
            raise ValueError("base_dhir mode requires model.generate_base to bypass adapter hooks")
        generate_fn = lambda: generate_base(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        selected_expert = None

    start = time.perf_counter()
    output = generate_fn()
    elapsed = max(time.perf_counter() - start, 1e-9)
    sequences = getattr(output, "sequences", output)
    if isinstance(sequences, dict):
        sequences = sequences["sequences"]
    if not hasattr(sequences, "shape") or len(sequences.shape) != 2:
        raise TypeError("model.generate must return a rank-2 token tensor or generation output")
    prompt_length = int(inputs["input_ids"].shape[-1])
    generated_ids = sequences[0, prompt_length:]
    generated_count = int(generated_ids.numel())
    if generated_count < 1:
        raise RuntimeError("model generated no new tokens")

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "status": "INFERENCE_COMPLETED",
        "mode": mode,
        "task": task,
        "language_tag": language_tag,
        "detected_script": text_meta.get("dominant_script"),
        "tokenizer": encoded.get("tokenizer_name"),
        "tokenizer_vocab_sha256": encoded.get("tokenizer_vocab_sha256"),
        "token_ids_unchanged": bool(encoded.get("token_ids_unchanged")),
        "suggested_expert_id": suggested_expert,
        "suggested_expert_name": control.get("suggested_expert_name"),
        "selected_expert_id": selected_expert,
        "router_status": (
            "UNTRAINED_NOT_QUALITY_VALIDATED" if mode == "top2" else None
        ),
        "prompt_tokens": prompt_length,
        "generated_tokens": generated_count,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(generated_count / elapsed, 3),
        "answer": answer,
        "training_performed": False,
    }
