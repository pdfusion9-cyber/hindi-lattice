"""Frozen external model adapters for NeuroCore/Fabric Swarm.

External model weights are never tensor-copied into NeuroCore. Each backend
keeps its native model class, tokenizer, chat template and serving runtime.
Only the task contract crosses the adapter boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


class Backend(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class ExpertRequest:
    expert_id: int
    task: str
    prompt: str
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExpertResponse:
    expert_id: int
    model_key: str
    text: str
    weights_frozen: bool = True


class ExternalExpertAdapter:
    """One fixed expert -> one frozen backend adapter."""

    def __init__(
        self,
        expert_id: int,
        model_key: str,
        backend: Backend,
        prompt_builder: Callable[[ExpertRequest], str] | None = None,
    ) -> None:
        if not 0 <= expert_id < 20:
            raise ValueError("expert_id must be in [0, 19]")
        self.expert_id = expert_id
        self.model_key = model_key
        self.backend = backend
        self.prompt_builder = prompt_builder or (lambda req: req.prompt)

    def run(self, request: ExpertRequest, **generation_kwargs: Any) -> ExpertResponse:
        if request.expert_id != self.expert_id:
            raise ValueError(
                f"fixed assignment violation: adapter={self.expert_id}, "
                f"request={request.expert_id}"
            )
        prompt = self.prompt_builder(request)
        result = self.backend.generate(prompt, **generation_kwargs)
        if not isinstance(result, str):
            raise TypeError("external backend must return text")
        return ExpertResponse(self.expert_id, self.model_key, result, True)


class TransformersBackend:
    """Optional lazy Transformers backend; imports heavy dependencies on use."""

    def __init__(self, repo_id: str, revision: str, *, device_map: str = "auto",
                 torch_dtype: str = "auto", trust_remote_code: bool = False) -> None:
        if not revision or revision in {"main", "master", "latest"}:
            raise ValueError("pin an immutable Hugging Face commit revision")
        self.repo_id = repo_id
        self.revision = revision
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.repo_id, revision=self.revision, device_map=self.device_map,
            torch_dtype=self.torch_dtype if self.torch_dtype != "auto" else "auto",
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        allowed = {"max_new_tokens", "temperature", "top_p", "do_sample"}
        generation = {k: v for k, v in kwargs.items() if k in allowed}
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


class FixedExpertRouter:
    """Validates 1:1 worker/expert assignment and dispatches by explicit ID."""

    def __init__(self, adapters: Mapping[int, ExternalExpertAdapter]) -> None:
        if set(adapters) != set(range(20)):
            raise ValueError("exactly 20 adapters (expert IDs 0..19) are required")
        for expert_id, adapter in adapters.items():
            if expert_id != adapter.expert_id:
                raise ValueError("adapter map key does not match fixed expert ID")
        self.adapters = dict(adapters)

    def dispatch(self, request: ExpertRequest, **kwargs: Any) -> ExpertResponse:
        return self.adapters[request.expert_id].run(request, **kwargs)
