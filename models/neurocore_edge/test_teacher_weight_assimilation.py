import unittest
from types import SimpleNamespace

import torch

from teacher_weight_assimilation import (
    FrozenTeacher, TeacherSpec, TeacherToStudentExpertDistiller,
    copy_exact_compatible_tensors,
)


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(ch) % 31 for ch in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class FakeTeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class FakeTeacher(FrozenTeacher):
    def __init__(self, spec):
        super().__init__(spec, FakeTeacherModel(), FakeTokenizer())

    def generate(self, prompt, *, max_new_tokens=256):
        return "उत्तर"


class FakeStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([
            torch.nn.ModuleDict({"head": torch.nn.Linear(1, 31)})
            for _ in range(20)
        ])

    def forward(self, input_ids, expert_id):
        x = torch.nn.functional.one_hot(input_ids % 31, num_classes=1).float()
        return {"logits": self.experts[expert_id]["head"](x)}


class AssimilationTests(unittest.TestCase):
    def test_teacher_student_only_selected_expert_updates(self):
        spec = TeacherSpec(
            repo_id="org/teacher",
            revision="0123456789abcdef0123456789abcdef01234567",
            expert_id=3,
        )
        teacher = FakeTeacher(spec)
        student = FakeStudent()
        before = [p.detach().clone() for p in student.parameters()]
        named = [
            (name, p) for name, p in student.named_parameters()
            if ".experts.3." in name
        ]
        distiller = TeacherToStudentExpertDistiller(
            teacher=teacher,
            student_model=student,
            student_tokenizer=FakeTokenizer(),
            expert_id=3,
            trainable_named_parameters=named,
            student_forward=lambda model, ids, expert_id: model(ids, expert_id),
        )
        report = distiller.train(["नमस्ते"], steps=2)
        self.assertEqual(report["status"], "TEACHER_TO_STUDENT_EXPERT_DISTILLATION_PASS")
        self.assertTrue(report["teacher_frozen"])
        self.assertTrue(report["student_only_assigned_expert_trainable"])
        params_after = list(student.parameters())
        changed = [i for i, (a, b) in enumerate(zip(before, params_after))
                   if not torch.equal(a, b.detach())]
        allowed = {i for i, name in enumerate(
            n for n, _ in student.named_parameters()
        ) if ".experts.3." in name}
        self.assertTrue(changed)
        self.assertTrue(set(changed).issubset(allowed))

    def test_direct_tensor_copy_rejects_shape_mismatch(self):
        source = {"a": torch.ones(3), "b": torch.ones(2, 2)}
        target = {"x": torch.zeros(3), "y": torch.zeros(4)}
        updated, audit = copy_exact_compatible_tensors(
            source, target, {"a": "x", "b": "y"}
        )
        self.assertTrue(torch.equal(updated["x"], torch.ones(3)))
        self.assertEqual(audit["copied_count"], 1)
        self.assertEqual(audit["rejected"][0]["reason"], "shape_mismatch")

    def test_unpinned_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            TeacherSpec("org/model", "main", 2)

    def test_cross_expert_teacher_assignment_is_rejected(self):
        student = FakeStudent()
        teacher = FakeTeacher(TeacherSpec(
            "org/model", "0123456789abcdef0123456789abcdef01234567", 2
        ))
        named = [(name, p) for name, p in student.named_parameters()
                 if ".experts.3." in name]
        with self.assertRaises(ValueError):
            TeacherToStudentExpertDistiller(
                teacher=teacher, student_model=student,
                student_tokenizer=FakeTokenizer(), expert_id=3,
                trainable_named_parameters=named,
                student_forward=lambda model, ids, expert_id: model(ids, expert_id),
            )


if __name__ == "__main__":
    unittest.main()
