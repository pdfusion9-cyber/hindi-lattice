import tempfile
import unittest
from pathlib import Path

import torch

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM
from teacher_weight_assimilation import (
    FrozenTeacher,
    TeacherSpec,
    TeacherToStudentExpertDistiller,
    build_native_neurocore_expert_distiller,
    copy_exact_compatible_tensors,
)


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(character) % 61 for character in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "उत्तर"


class FakeTeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class FakeTeacher(FrozenTeacher):
    def __init__(self, expert_id):
        spec = TeacherSpec(
            repo_id="org/frozen-teacher",
            revision="0123456789abcdef0123456789abcdef01234567",
            expert_id=expert_id,
        )
        super().__init__(spec, FakeTeacherModel(), FakeTokenizer())

    def generate(self, prompt, *, max_new_tokens=256):
        return "उत्तर"


def tiny_canonical_config():
    return NeuroCoreEdgeConfig(
        vocab_size=61,
        hidden_size=16,
        num_hidden_layers=3,
        num_experts=20,
        experts_per_token=2,
        expert_intermediate_size=8,
        attention_heads=4,
        kv_heads=2,
        sliding_window=8,
        recurrent_state_size=4,
        prelude_layers=1,
        core_unique_layers=1,
        core_repeats=1,
        coda_layers=1,
        meta_token_count=2,
        global_attention_layers=(1,),
    )


class TeacherAssimilationTests(unittest.TestCase):
    def test_exact_copy_needs_semantic_attestation_and_matching_shape(self):
        source = {"w": torch.ones(2, 3), "bad": torch.ones(4)}
        target = {"target_w": torch.zeros(2, 3), "target_bad": torch.zeros(5)}

        copied, audit = copy_exact_compatible_tensors(
            source, target, {"w": "target_w", "bad": "target_bad"}
        )
        self.assertEqual(audit["copied_count"], 0)
        self.assertEqual(
            audit["rejected"][0]["reason"],
            "semantic_compatibility_not_attested",
        )

        copied, audit = copy_exact_compatible_tensors(
            source,
            target,
            {"w": "target_w", "bad": "target_bad"},
            semantic_compatibility={"w": True, "bad": True},
        )
        self.assertTrue(torch.equal(copied["target_w"], torch.ones(2, 3)))
        self.assertEqual(audit["copied_count"], 1)
        self.assertEqual(audit["rejected"][0]["reason"], "shape_mismatch")

    def test_fixed_expert_11_resolves_all_canonical_sandwich_blocks(self):
        model = NeuroCoreEdgeForCausalLM(tiny_canonical_config())
        distiller = build_native_neurocore_expert_distiller(
            teacher=FakeTeacher(11),
            student_model=model,
            student_tokenizer=FakeTokenizer(),
            expert_id=11,
        )
        names = [name for name, _ in distiller.trainable_named_parameters]
        self.assertEqual(len(names), 9)  # 3 matrices x 3 unique tiny blocks
        self.assertTrue(any(name.startswith("prelude_layers.0.experts.11.") for name in names))
        self.assertTrue(any(name.startswith("core_layers.0.experts.11.") for name in names))
        self.assertTrue(any(name.startswith("coda_layers.0.experts.11.") for name in names))
        self.assertTrue(all(".experts.11." in "." + name + "." for name in names))
        self.assertTrue(all(parameter.requires_grad for _, parameter in distiller.trainable_named_parameters))
        selected = {id(parameter) for _, parameter in distiller.trainable_named_parameters}
        self.assertTrue(all(
            parameter.requires_grad == (id(parameter) in selected)
            for parameter in model.parameters()
        ))

    def test_teacher_distillation_updates_only_assigned_native_expert(self):
        torch.manual_seed(19)
        model = NeuroCoreEdgeForCausalLM(tiny_canonical_config())
        teacher = FakeTeacher(11)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        distiller = build_native_neurocore_expert_distiller(
            teacher=teacher,
            student_model=model,
            student_tokenizer=FakeTokenizer(),
            expert_id=11,
        )
        report = distiller.train(["हिंदी दस्तावेज़ का सारांश बनाओ"], steps=2)
        self.assertEqual(report["status"], "TEACHER_TO_STUDENT_EXPERT_DISTILLATION_PASS")
        self.assertTrue(report["teacher_frozen"])
        self.assertTrue(report["student_only_assigned_expert_trainable"])

        changed = []
        selected_marker = ".experts.11."
        for name, parameter in model.named_parameters():
            if not torch.equal(before[name], parameter.detach()):
                changed.append(name)
                self.assertIn(selected_marker, "." + name + ".")
        self.assertTrue(changed)
        self.assertTrue(all(
            not parameter.requires_grad or selected_marker in "." + name + "."
            for name, parameter in model.named_parameters()
        ))
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.model.parameters()))

    def test_checkpoint_resume_rejects_wrong_fingerprints(self):
        model = NeuroCoreEdgeForCausalLM(tiny_canonical_config())
        teacher = FakeTeacher(11)
        distiller = build_native_neurocore_expert_distiller(
            teacher=teacher,
            student_model=model,
            student_tokenizer=FakeTokenizer(),
            expert_id=11,
        )
        distiller.train(["प्रशिक्षण"], steps=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "expert-11.pt")
            distiller.save_checkpoint(
                checkpoint,
                base_fingerprint="base-sha",
                tokenizer_fingerprint="tokenizer-sha",
                dataset_fingerprint="prompts-sha",
            )
            result = distiller.load_checkpoint(
                checkpoint,
                base_fingerprint="base-sha",
                tokenizer_fingerprint="tokenizer-sha",
                dataset_fingerprint="prompts-sha",
            )
            self.assertEqual(result["steps_completed"], 1)
            with self.assertRaises(ValueError):
                distiller.load_checkpoint(
                    checkpoint,
                    base_fingerprint="wrong-base",
                    tokenizer_fingerprint="tokenizer-sha",
                    dataset_fingerprint="prompts-sha",
                )

    def test_revision_and_worker_assignment_are_pinned(self):
        with self.assertRaises(ValueError):
            TeacherSpec("org/model", "main", 11)
        with self.assertRaises(ValueError):
            TeacherSpec(
                "org/model",
                "0123456789abcdef0123456789abcdef01234567",
                20,
            )
        with self.assertRaises(ValueError):
            TeacherToStudentExpertDistiller(
                teacher=FakeTeacher(10),
                student_model=NeuroCoreEdgeForCausalLM(tiny_canonical_config()),
                student_tokenizer=FakeTokenizer(),
                expert_id=11,
                trainable_named_parameters=[],
                student_forward=lambda model, ids, expert_id: model(
                    ids, force_expert_id=expert_id
                ),
            )


if __name__ == "__main__":
    unittest.main()
