import json
import tempfile
import unittest
from pathlib import Path

from preflight_dhir2b_expert import verify_pending_registry, verify_fixed_expert_trainability


class TinyExpert:
    def __init__(self):
        import torch
        self.parameters_list = [torch.nn.Parameter(torch.ones(2))]
    def parameters(self):
        return iter(self.parameters_list)


class TinyLayer:
    def __init__(self):
        import torch
        self.router = torch.nn.Linear(2, 2, bias=False)
        self.experts = [TinyExpert(), TinyExpert()]


class TinyModel:
    def __init__(self):
        import torch
        self.backbone = torch.nn.Linear(2, 2)
        self.adapter_layers = [TinyLayer(), TinyLayer()]
        self.num_experts = 2


class PreflightGateTests(unittest.TestCase):
    def make_registry(self, **overrides):
        row = {
            "expert_id": 16,
            "dataset_id": "local/neurocore_synthetic_sft/expert16_self_correction",
            "revision": "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9",
            "approved": False,
            "commercial_training_allowed": False,
            "public_redistribution_allowed": False,
            "pii_reviewed": False,
            "consent_reviewed": False,
            "schema_reviewed": False,
            "split_reviewed": False,
        }
        row.update(overrides)
        return {
            "status": "ALL_DATASETS_PENDING_REVIEW",
            "datasets": [row],
            "approved_training_artifacts": [],
            "approved_tokenizers": [],
        }

    def test_pending_registry_allows_only_non_training_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.json"
            path.write_text(json.dumps(self.make_registry()))
            result = verify_pending_registry(
                path, 16,
                "local/neurocore_synthetic_sft/expert16_self_correction",
                "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9",
            )
        self.assertFalse(result["dataset_approved"])

    def test_preflight_rejects_any_approval_state_change(self):
        for field, value in (
            ("approved", True),
            ("commercial_training_allowed", True),
            ("pii_reviewed", True),
            ("consent_reviewed", True),
            ("schema_reviewed", True),
            ("split_reviewed", True),
            ("public_redistribution_allowed", True),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "approval.json"
                path.write_text(json.dumps(self.make_registry(**{field: value})))
                with self.assertRaises(PermissionError):
                    verify_pending_registry(
                        path, 16,
                        "local/neurocore_synthetic_sft/expert16_self_correction",
                        "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9",
                    )

    def test_preflight_rejects_artifact_or_tokenizer_approval(self):
        for key in ("approved_training_artifacts", "approved_tokenizers"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                document = self.make_registry()
                document[key] = [{"approval_id": "unexpected"}]
                path = Path(directory) / "approval.json"
                path.write_text(json.dumps(document))
                with self.assertRaises(PermissionError):
                    verify_pending_registry(
                        path, 16,
                        "local/neurocore_synthetic_sft/expert16_self_correction",
                        "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9",
                    )

    def test_only_selected_expert_may_be_trainable(self):
        import torch
        model = TinyModel()
        for layer in model.adapter_layers:
            layer.router.requires_grad_(False)
        selected = []
        for layer in model.adapter_layers:
            for index, expert in enumerate(layer.experts):
                for parameter in expert.parameters():
                    parameter.requires_grad_(index == 1)
                    if index == 1:
                        selected.append(parameter)
        with self.assertRaises(RuntimeError):
            verify_fixed_expert_trainability(model, 1, selected)
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        result = verify_fixed_expert_trainability(model, 1, selected)
        self.assertTrue(result["base_backbone_frozen"])
        self.assertTrue(result["all_routers_frozen"])


if __name__ == "__main__":
    unittest.main()
