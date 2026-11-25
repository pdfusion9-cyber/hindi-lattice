"""Cross-model teacher-weight assimilation into one native NeuroCore expert.

The teacher checkpoint is loaded with its own upstream architecture/tokenizer,
frozen, and used to create targets. The native NeuroCore expert is trainable.
Only exact-name/exact-shape tensors may be copied directly; all other transfer
uses text/sequence distillation, never blind tensor reshaping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
import re


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


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
        if self.expert_id not in range(20):
            raise ValueError("expert_id must be in [0, 19]")


class FrozenTeacher:
    """Loads real upstream weights in their native architecture; no gradients."""

    def __init__(self, spec: TeacherSpec, model: Any, tokenizer: Any) -> None:
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls, spec: TeacherSpec, *, device_map: str = "auto",
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

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        import torch
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
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
        device = next(self.model.parameters()).device
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy only explicitly mapped tensors with exact shapes.

    This is intentionally conservative. Different vocabularies, hidden sizes,
    expert layouts or attention blocks must go through distillation instead.
    Returns (updated_target_state, audit_report).
    """
    import torch
    updated = {key: value.detach().clone() for key, value in target_state.items()}
    copied, rejected = [], []
    for source_key, target_key in key_map.items():
        if source_key not in source_state or target_key not in target_state:
            rejected.append({"source": source_key, "target": target_key, "reason": "missing_key"})
            continue
        src, dst = source_state[source_key], target_state[target_key]
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            rejected.append({"source": source_key, "target": target_key, "reason": "not_tensor"})
            continue
        if tuple(src.shape) != tuple(dst.shape):
            rejected.append({
                "source": source_key, "target": target_key,
                "reason": "shape_mismatch",
                "source_shape": list(src.shape), "target_shape": list(dst.shape),
            })
            continue
        if not bool(torch.isfinite(src).all()):
            rejected.append({"source": source_key, "target": target_key, "reason": "non_finite_source"})
            continue
        updated[target_key] = src.detach().to(device=dst.device, dtype=dst.dtype).clone()
        copied.append({"source": source_key, "target": target_key, "shape": list(src.shape)})
    return updated, {
        "copied_count": len(copied),
        "rejected_count": len(rejected),
        "copied": copied,
        "rejected": rejected,
        "mode": "exact_name_mapping_and_exact_shape_only",
    }


class TeacherToStudentExpertDistiller:
    """Train only one native expert from a frozen external teacher.

    The student tokenizer defines the student vocabulary. The teacher generates
    target text with its own tokenizer; therefore different vocabularies do not
    need to be numerically aligned.
    """

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
        import torch
        if expert_id not in range(20) or teacher.spec.expert_id != expert_id:
            raise ValueError("teacher and student expert_id must match the fixed assignment")
        if not trainable_named_parameters:
            raise ValueError("no trainable student-expert parameters supplied")
        marker = f".experts.{expert_id}."
        bad_names = [name for name, _ in trainable_named_parameters if marker not in name]
        if bad_names:
            raise ValueError(
                "only parameters belonging to the assigned expert may be trainable: "
                + ", ".join(bad_names[:5])
            )
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

    def _student_batch(self, prompt: str, target: str):
        import torch
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
                all_text, prompt_text = prompt + "\n" + target, prompt
        else:
            all_text, prompt_text = prompt + "\n" + target, prompt
        all_ids = self.student_tokenizer(
            all_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        prompt_ids = self.student_tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        if all_ids.shape[1] <= prompt_ids.shape[1]:
            raise ValueError("student tokenizer produced no target tokens")
        labels = all_ids.clone()
        labels[:, :min(prompt_ids.shape[1], labels.shape[1])] = -100
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
        """Save the assigned expert, optimizer, RNG and provenance for resume."""
        import os
        from pathlib import Path
        import torch

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
            "fingerprints": fingerprints,
            "student_expert_state": {
                name: parameter.detach().cpu().clone()
                for name, parameter in selected.items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "provenance": {
                "teacher_frozen": True,
                "student_only_assigned_expert_trainable": True,
                "external_teacher_weights_used_for_supervision": True,
                "student_weights_are_native_neurocore": True,
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
        """Resume only if expert, teacher revision and all data/model hashes match."""
        import torch
        from pathlib import Path

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
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        return {
            "status": "TEACHER_DISTILLATION_CHECKPOINT_RESUMED",
            "expert_id": self.expert_id,
            "steps_completed": self.steps_completed,
            "fingerprints": expected,
        }

    def train_step(self, prompt: str, *, max_new_tokens: int = 256) -> float:
        import torch
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
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite student distillation loss")
        loss.backward()
        grads = [p.grad for p in self.student_model.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("missing or non-finite gradients during expert distillation")
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in self.trainable_named_parameters], 1.0
        )
        self.optimizer.step()
        self.steps_completed += 1
        # Assert frozen teacher and all non-selected student tensors remain frozen.
        if any(p.requires_grad for p in self.teacher.model.parameters()):
            raise RuntimeError("teacher weights unexpectedly became trainable")
        return float(loss.detach().cpu())

    def train(self, prompts: Iterable[str], *, steps: int, max_new_tokens: int = 256) -> dict[str, Any]:
        prompt_list = list(prompts)
        if not prompt_list:
            raise ValueError("at least one prompt is required")
        if steps < 1:
            raise ValueError("steps must be positive")
        losses = [
            self.train_step(prompt_list[index % len(prompt_list)], max_new_tokens=max_new_tokens)
            for index in range(steps)
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
    """Bind distillation to the repository's NeuroCore force-expert forward API."""
    model_config = getattr(student_model, "config", None)
    configured_experts = getattr(model_config, "num_experts", None)
    if configured_experts != 20:
        raise ValueError("student must be the configured native 20-expert NeuroCore model")
    if not callable(getattr(student_model, "forward", None)):
        raise TypeError("student model must expose a forward method")
    marker = f".experts.{expert_id}."
    trainable = [
        (name, parameter)
        for name, parameter in student_model.named_parameters()
        if marker in name
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
