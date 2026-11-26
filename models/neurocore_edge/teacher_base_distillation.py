"""Full canonical NeuroCore base distillation from a frozen open-weight teacher.

This bootstrap stage trains the entire native student (shared layers, routers,
and sparse experts) from teacher-generated sequences. It is separate from the
later fixed-worker expert specialization stage. Checkpoints remain provisional
until independent validation certifies them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import os

import torch
from torch import nn
import torch.nn.functional as F

from teacher_weight_assimilation import FrozenTeacher


class TeacherToStudentBaseDistiller:
    def __init__(
        self,
        *,
        teacher: FrozenTeacher,
        student_model: nn.Module,
        student_tokenizer: Any,
        learning_rate: float = 1e-4,
        router_aux_weight: float = 0.01,
        weight_decay: float = 0.01,
        optimizer_name: str = "adafactor",
    ) -> None:
        if learning_rate <= 0 or router_aux_weight < 0 or weight_decay < 0:
            raise ValueError("invalid base-distillation optimizer settings")
        if optimizer_name not in {"adafactor", "adamw"}:
            raise ValueError("optimizer_name must be adafactor or adamw")
        self.teacher = teacher
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        for parameter in teacher.model.parameters():
            parameter.requires_grad_(False)
        teacher.model.eval()
        for parameter in student_model.parameters():
            parameter.requires_grad_(True)
        trainable = [p for p in student_model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("student has no trainable parameters")
        if optimizer_name == "adafactor":
            try:
                from transformers.optimization import Adafactor
            except ImportError as exc:
                raise RuntimeError(
                    "Adafactor requires transformers; install the training requirements"
                ) from exc
            self.optimizer = Adafactor(
                trainable,
                lr=learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable, lr=learning_rate, weight_decay=weight_decay
            )
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.router_aux_weight = router_aux_weight
        self.weight_decay = weight_decay
        self.steps_completed = 0
        self.data_cursor = 0

    def _encode(self, prompt: str, target: str):
        full = prompt + "\n" + target
        prefix = prompt + "\n"
        full_ids = self.student_tokenizer(
            full, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        prefix_ids = self.student_tokenizer(
            prefix, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        if full_ids.ndim != 2 or prefix_ids.ndim != 2 or full_ids.shape[0] != 1:
            raise ValueError("student tokenizer must return one rank-2 sequence")
        if full_ids.shape[1] <= prefix_ids.shape[1]:
            raise ValueError("teacher response produced no student target tokens")
        labels = full_ids.clone()
        labels[:, :prefix_ids.shape[1]] = -100
        if not bool((labels[:, 1:] != -100).any()):
            raise ValueError("no next-token target remains after causal shift")
        return full_ids, labels

    def train_step(self, prompt: str, *, max_new_tokens: int = 128) -> dict[str, Any]:
        target = self.teacher.generate(prompt, max_new_tokens=max_new_tokens)
        if not target:
            raise RuntimeError("frozen teacher returned an empty response")
        input_ids, labels = self._encode(prompt, target)
        device = next(self.student_model.parameters()).device
        input_ids, labels = input_ids.to(device), labels.to(device)

        self.student_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.student_model(input_ids)
        logits = output["logits"]
        token_loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        router_loss = output.get("router_aux_loss", token_loss.new_zeros(()))
        loss = token_loss + self.router_aux_weight * router_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite base-distillation loss")
        loss.backward()

        parameters = [p for p in self.student_model.parameters() if p.requires_grad]
        if not any(p.grad is not None for p in parameters):
            raise RuntimeError("student received no gradients")
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in parameters
        ):
            raise RuntimeError("non-finite student gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        self.steps_completed += 1
        self.data_cursor += 1
        if any(p.requires_grad for p in self.teacher.model.parameters()):
            raise RuntimeError("teacher unexpectedly became trainable")
        return {
            "total_loss": float(loss.detach().cpu()),
            "token_loss": float(token_loss.detach().cpu()),
            "router_aux_loss": float(router_loss.detach().cpu()),
        }

    def train(
        self, prompts: Iterable[str], *, steps: int, max_new_tokens: int = 128
    ) -> dict[str, Any]:
        prompt_list = list(prompts)
        if not prompt_list:
            raise ValueError("at least one authorized prompt is required")
        if steps < 1:
            raise ValueError("steps must be positive")
        reports = [
            self.train_step(
                prompt_list[self.data_cursor % len(prompt_list)],
                max_new_tokens=max_new_tokens,
            )
            for _ in range(steps)
        ]
        return {
            "status": "BASE_DISTILLATION_STEP_COMPLETED",
            "steps_completed": self.steps_completed,
            "initial_loss": reports[0]["total_loss"],
            "final_loss": reports[-1]["total_loss"],
            "teacher_frozen": True,
            "all_student_parameters_trainable": True,
            "reports": reports,
        }

    def _student_config_dict(self) -> dict[str, Any]:
        cfg = self.student_model.config
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(cfg).items()
            if not key.startswith("_")
        }

    def save_checkpoint(
        self,
        path: str,
        *,
        tokenizer_fingerprint: str,
        prompt_fingerprint: str,
        teacher_revision: str,
    ) -> dict[str, Any]:
        if not all((tokenizer_fingerprint, prompt_fingerprint, teacher_revision)):
            raise ValueError("tokenizer, prompt and teacher fingerprints are required")
        payload = {
            "schema_version": 1,
            "status": "BASE_DISTILLATION_IN_PROGRESS",
            "architecture_version": "0.3.2-canonical",
            "config": self._student_config_dict(),
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.student_model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "router_aux_weight": self.router_aux_weight,
            "steps_completed": self.steps_completed,
            "data_cursor": self.data_cursor,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "provenance": {
                "teacher_repo_id": self.teacher.spec.repo_id,
                "teacher_revision": teacher_revision,
                "teacher_frozen": True,
                "student_all_weights_trainable": True,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "prompt_fingerprint": prompt_fingerprint,
                "independent_validation_required": True,
                "expert_workers_may_consume": False,
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
            "status": payload["status"],
            "path": str(destination),
            "steps_completed": self.steps_completed,
            "expert_workers_may_consume": False,
        }

    def load_checkpoint(
        self,
        path: str,
        *,
        tokenizer_fingerprint: str,
        prompt_fingerprint: str,
        teacher_revision: str,
    ) -> dict[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported base-distillation checkpoint")
        if payload.get("status") != "BASE_DISTILLATION_IN_PROGRESS":
            raise ValueError("checkpoint is not a provisional base-distillation checkpoint")
        if payload.get("architecture_version") != "0.3.2-canonical":
            raise ValueError("checkpoint architecture version mismatch")
        if payload.get("config") != self._student_config_dict():
            raise ValueError("checkpoint student config mismatch")
        provenance = payload.get("provenance", {})
        expected = {
            "teacher_repo_id": self.teacher.spec.repo_id,
            "teacher_revision": teacher_revision,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f"base checkpoint provenance mismatch: {key}")
        if provenance.get("expert_workers_may_consume") is not False:
            raise ValueError("provisional checkpoint cannot be used by expert workers")
        if payload.get("optimizer_name") != self.optimizer_name:
            raise ValueError("checkpoint optimizer mismatch")

        incoming = payload.get("model_state_dict")
        current = self.student_model.state_dict()
        if not isinstance(incoming, dict) or set(incoming) != set(current):
            raise ValueError("base checkpoint tensor key set mismatch")
        with torch.no_grad():
            for key, target in current.items():
                tensor = incoming[key]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"invalid checkpoint tensor: {key}")
                if tensor.shape != target.shape or tensor.dtype != target.dtype:
                    raise ValueError(f"checkpoint tensor shape/dtype mismatch: {key}")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"checkpoint tensor is non-finite: {key}")
                target.copy_(tensor.to(device=target.device))

        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.steps_completed = int(payload["steps_completed"])
        self.data_cursor = int(payload["data_cursor"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_states = payload.get("cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("CUDA RNG device-count mismatch on resume")
            torch.cuda.set_rng_state_all(cuda_states)
        return {
            "status": "BASE_DISTILLATION_RESUMED",
            "steps_completed": self.steps_completed,
            "data_cursor": self.data_cursor,
            "expert_workers_may_consume": False,
        }
