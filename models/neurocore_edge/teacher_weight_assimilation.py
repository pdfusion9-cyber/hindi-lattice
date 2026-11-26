"""Frozen open-weight teacher -> native NeuroCore expert distillation.

Teachers are always loaded with their own architecture and tokenizer. Their
weights are frozen. Only the assigned NeuroCore expert tensors are trainable.
Direct tensor transfer is a separate, opt-in path requiring an explicit key
map, exact shape equality, and a semantic-compatibility attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import os
import re

import torch
from torch import nn
import torch.nn.functional as F


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _validate_expert_id(expert_id: int) -> None:
    if not isinstance(expert_id, int) or isinstance(expert_id, bool):
        raise TypeError("expert_id must be an integer")
    if expert_id not in range(20):
        raise ValueError("expert_id must be in [0, 19]")


@dataclass(frozen=True)
class TeacherSpec:
    repo_id: str
    revision: str
    expert_id: int
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError("repo_id must be an owner/model identifier")
        if not _COMMIT_RE.fullmatch(self.revision):
            raise ValueError("teacher revision must be a pinned 40-character commit SHA")
        _validate_expert_id(self.expert_id)


class FrozenTeacher:
    """Loads real upstream checkpoint weights in native form and freezes them."""

    def __init__(self, spec: TeacherSpec, model: Any, tokenizer: Any) -> None:
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        spec: TeacherSpec,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> "FrozenTeacher":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=spec.trust_remote_code,
        )
        return cls(spec, model, tokenizer)

    def _input_device(self) -> torch.device:
        # With HF device_map="auto", the first parameter is normally placed on
        # the input device. Prefer the model's input embeddings when available.
        try:
            embedding = self.model.get_input_embeddings()
            if embedding is not None:
                device = embedding.weight.device
                if device.type != "meta":
                    return device
        except (AttributeError, RuntimeError):
            pass
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        raise RuntimeError("could not determine the frozen teacher input device")

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (TypeError, ValueError):
                rendered = prompt
        else:
            rendered = prompt

        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = self._input_device()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        answer_ids = output[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def copy_exact_compatible_tensors(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    key_map: Mapping[str, str],
    *,
    semantic_compatibility: Mapping[str, bool] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy tensors only after mapping, shape, and semantic checks all pass.

    semantic_compatibility is keyed by source tensor name. A missing/false
    attestation fails closed, because equal dimensions do not prove that two
    models' tensors have the same mathematical meaning.
    """
    updated = {key: value.detach().clone() for key, value in target_state.items()}
    semantic_compatibility = semantic_compatibility or {}
    copied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for source_key, target_key in key_map.items():
        if source_key not in source_state or target_key not in target_state:
            rejected.append({
                "source": source_key, "target": target_key, "reason": "missing_key"
            })
            continue
        if semantic_compatibility.get(source_key) is not True:
            rejected.append({
                "source": source_key, "target": target_key,
                "reason": "semantic_compatibility_not_attested",
            })
            continue

        src, dst = source_state[source_key], target_state[target_key]
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            rejected.append({
                "source": source_key, "target": target_key, "reason": "not_tensor"
            })
            continue
        if tuple(src.shape) != tuple(dst.shape):
            rejected.append({
                "source": source_key, "target": target_key,
                "reason": "shape_mismatch",
                "source_shape": list(src.shape),
                "target_shape": list(dst.shape),
            })
            continue
        if not bool(torch.isfinite(src).all()):
            rejected.append({
                "source": source_key, "target": target_key,
                "reason": "non_finite_source",
            })
            continue
        updated[target_key] = src.detach().to(
            device=dst.device, dtype=dst.dtype
        ).clone()
        copied.append({
            "source": source_key, "target": target_key,
            "shape": list(src.shape),
        })

    return updated, {
        "copied_count": len(copied),
        "rejected_count": len(rejected),
        "copied": copied,
        "rejected": rejected,
        "mode": "explicit_mapping_exact_shape_and_semantic_attestation",
    }


class TeacherToStudentExpertDistiller:
    """Distil teacher-generated sequences into exactly one native expert."""

    def __init__(
        self,
        *,
        teacher: FrozenTeacher,
        student_model: Any,
        student_tokenizer: Any,
        expert_id: int,
        trainable_named_parameters: Sequence[tuple[str, Any]],
        student_forward: Callable[[Any, Any, int], Any],
        learning_rate: float = 1e-4,
    ) -> None:
        _validate_expert_id(expert_id)
        if teacher.spec.expert_id != expert_id:
            raise ValueError("teacher and student expert_id must match the fixed assignment")
        if not trainable_named_parameters:
            raise ValueError("no trainable student-expert parameters supplied")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        marker = f".experts.{expert_id}."
        bad_names = [
            name for name, _ in trainable_named_parameters
            if marker not in f".{name}."
        ]
        if bad_names:
            raise ValueError(
                "only parameters belonging to the assigned expert may be trainable: "
                + ", ".join(bad_names[:5])
            )

        selected_ids = {id(parameter) for _, parameter in trainable_named_parameters}
        if len(selected_ids) != len(trainable_named_parameters):
            raise ValueError("trainable expert parameter list contains duplicates")

        for parameter in student_model.parameters():
            parameter.requires_grad_(False)
        parameters = []
        for _, parameter in trainable_named_parameters:
            parameter.requires_grad_(True)
            parameters.append(parameter)

        self.teacher = teacher
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.expert_id = expert_id
        self.trainable_named_parameters = tuple(trainable_named_parameters)
        self.student_forward = student_forward
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        self.steps_completed = 0
        self.data_cursor = 0

    def _student_batch(self, prompt: str, target: str):
        if hasattr(self.student_tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]
            try:
                all_text = self.student_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_text = self.student_tokenizer.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )
            except (TypeError, ValueError):
                all_text, prompt_text = prompt + "\n" + target, prompt + "\n"
        else:
            all_text, prompt_text = prompt + "\n" + target, prompt + "\n"

        all_ids = self.student_tokenizer(
            all_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        prompt_ids = self.student_tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        if all_ids.ndim != 2 or prompt_ids.ndim != 2:
            raise ValueError("student tokenizer must return rank-2 input_ids")
        if all_ids.shape[0] != 1 or prompt_ids.shape[0] != 1:
            raise ValueError("student tokenizer must return a single-example batch")
        if all_ids.shape[1] <= prompt_ids.shape[1]:
            raise ValueError("student tokenizer produced no target tokens")

        labels = all_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        if not bool((labels[:, 1:] != -100).any()):
            raise ValueError("student target has no next-token labels after causal shift")
        return all_ids, labels

    def _selected_parameter_map(self) -> dict[str, Any]:
        return {name: parameter for name, parameter in self.trainable_named_parameters}

    def save_checkpoint(
        self,
        path: str,
        *,
        base_fingerprint: str,
        tokenizer_fingerprint: str,
        dataset_fingerprint: str,
    ) -> dict[str, Any]:
        """Atomically save only the assigned expert plus resumable optimizer state."""
        fingerprints = {
            "base_fingerprint": base_fingerprint,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
        }
        if any(not isinstance(value, str) or not value for value in fingerprints.values()):
            raise ValueError("base, tokenizer and dataset fingerprints are required")

        selected = self._selected_parameter_map()
        payload = {
            "schema_version": 1,
            "expert_id": self.expert_id,
            "teacher_repo_id": self.teacher.spec.repo_id,
            "teacher_revision": self.teacher.spec.revision,
            "steps_completed": self.steps_completed,
            "data_cursor": self.data_cursor,
            "fingerprints": fingerprints,
            "student_expert_state": {
                name: parameter.detach().cpu().clone()
                for name, parameter in selected.items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "provenance": {
                "teacher_frozen": True,
                "student_only_assigned_expert_trainable": True,
                "external_teacher_weights_used_for_supervision": True,
                "student_weights_are_native_neurocore": True,
                "transfer_mode": "teacher_generated_sequence_distillation",
            },
        }

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        return {
            "status": "TEACHER_DISTILLATION_CHECKPOINT_SAVED",
            "path": str(destination),
            "expert_id": self.expert_id,
            "steps_completed": self.steps_completed,
            "fingerprints": fingerprints,
        }

    def load_checkpoint(
        self,
        path: str,
        *,
        base_fingerprint: str,
        tokenizer_fingerprint: str,
        dataset_fingerprint: str,
    ) -> dict[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported teacher-distillation checkpoint")

        expected = {
            "base_fingerprint": base_fingerprint,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
        }
        if payload.get("expert_id") != self.expert_id:
            raise ValueError("checkpoint belongs to a different expert")
        if payload.get("teacher_repo_id") != self.teacher.spec.repo_id:
            raise ValueError("teacher repository mismatch")
        if payload.get("teacher_revision") != self.teacher.spec.revision:
            raise ValueError("teacher revision mismatch")
        if payload.get("fingerprints") != expected:
            raise ValueError("base/tokenizer/dataset fingerprint mismatch")

        selected = self._selected_parameter_map()
        incoming = payload.get("student_expert_state")
        if not isinstance(incoming, dict) or set(incoming) != set(selected):
            raise ValueError("checkpoint expert tensor-key set mismatch")
        with torch.no_grad():
            for name, parameter in selected.items():
                tensor = incoming[name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"checkpoint tensor is invalid: {name}")
                if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError(f"checkpoint tensor shape/dtype mismatch: {name}")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"checkpoint tensor is non-finite: {name}")
                parameter.copy_(tensor.to(device=parameter.device))

        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.steps_completed = int(payload["steps_completed"])
        self.data_cursor = int(payload.get("data_cursor", self.steps_completed))
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_states = payload.get("cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("CUDA RNG device-count mismatch on resume")
            torch.cuda.set_rng_state_all(cuda_states)

        return {
            "status": "TEACHER_DISTILLATION_CHECKPOINT_RESUMED",
            "expert_id": self.expert_id,
            "steps_completed": self.steps_completed,
            "fingerprints": expected,
        }

    def train_step(self, prompt: str, *, max_new_tokens: int = 256) -> float:
        target = self.teacher.generate(prompt, max_new_tokens=max_new_tokens)
        if not target:
            raise RuntimeError("teacher returned an empty target")
        input_ids, labels = self._student_batch(prompt, target)
        device = next(self.student_model.parameters()).device
        input_ids, labels = input_ids.to(device), labels.to(device)

        self.student_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.student_forward(self.student_model, input_ids, self.expert_id)
        logits = output["logits"] if isinstance(output, dict) else output
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError("student_forward must return [batch, sequence, vocab] logits")

        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite student distillation loss")
        loss.backward()

        selected_ids = {id(parameter) for _, parameter in self.trainable_named_parameters}
        unexpected_gradients = [
            name for name, parameter in self.student_model.named_parameters()
            if id(parameter) not in selected_ids and parameter.grad is not None
        ]
        if unexpected_gradients:
            raise RuntimeError(
                "non-assigned student parameters received gradients: "
                + ", ".join(unexpected_gradients[:5])
            )
        gradients = [parameter.grad for _, parameter in self.trainable_named_parameters]
        if not any(gradient is not None for gradient in gradients):
            raise RuntimeError("assigned expert received no gradients")
        if not all(
            bool(torch.isfinite(gradient).all())
            for gradient in gradients if gradient is not None
        ):
            raise RuntimeError("non-finite gradients during expert distillation")

        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in self.trainable_named_parameters], 1.0
        )
        self.optimizer.step()
        self.steps_completed += 1
        self.data_cursor += 1

        if any(parameter.requires_grad for parameter in self.teacher.model.parameters()):
            raise RuntimeError("teacher weights unexpectedly became trainable")
        return float(loss.detach().cpu())

    def train(
        self,
        prompts: Iterable[str],
        *,
        steps: int,
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        prompt_list = list(prompts)
        if not prompt_list:
            raise ValueError("at least one prompt is required")
        if steps < 1:
            raise ValueError("steps must be positive")
        losses = [
            self.train_step(
                prompt_list[self.data_cursor % len(prompt_list)],
                max_new_tokens=max_new_tokens,
            )
            for _ in range(steps)
        ]
        return {
            "status": "TEACHER_TO_STUDENT_EXPERT_DISTILLATION_PASS",
            "expert_id": self.expert_id,
            "teacher_repo_id": self.teacher.spec.repo_id,
            "teacher_revision": self.teacher.spec.revision,
            "steps": self.steps_completed,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "losses": losses,
            "teacher_frozen": True,
            "student_only_assigned_expert_trainable": True,
            "student_tokenizer_used_for_loss": True,
            "raw_cross_model_tensor_reshape_used": False,
        }


def build_native_neurocore_expert_distiller(
    *,
    teacher: FrozenTeacher,
    student_model: Any,
    student_tokenizer: Any,
    expert_id: int,
    learning_rate: float = 1e-4,
) -> TeacherToStudentExpertDistiller:
    """Bind the distiller to canonical NeuroCore's force_expert_id API.

    Canonical parameter paths are, for example:
      prelude_layers.0.experts.11.gate.weight
      core_layers.2.experts.11.up.weight
      coda_layers.1.experts.11.down.weight

    The same fixed expert ID is forced through every unique sandwich block.
    """
    _validate_expert_id(expert_id)
    model_config = getattr(student_model, "config", None)
    if getattr(model_config, "num_experts", None) != 20:
        raise ValueError("student must be the native NeuroCore model with exactly 20 experts")
    if not callable(getattr(student_model, "forward", None)):
        raise TypeError("student model must expose a forward method")

    marker = f".experts.{expert_id}."
    trainable = [
        (name, parameter)
        for name, parameter in student_model.named_parameters()
        if marker in f".{name}."
    ]
    if not trainable:
        raise ValueError(f"native NeuroCore expert {expert_id} has no trainable tensors")

    def native_forward(model: Any, input_ids: Any, assigned_expert_id: int) -> Any:
        return model(input_ids, force_expert_id=assigned_expert_id)

    return TeacherToStudentExpertDistiller(
        teacher=teacher,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        expert_id=expert_id,
        trainable_named_parameters=trainable,
        student_forward=native_forward,
        learning_rate=learning_rate,
    )
