import tempfile
import unittest
from pathlib import Path

import torch

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM
from teacher_weight_assimilation import FrozenTeacher, TeacherSpec
from teacher_base_distillation import TeacherToStudentBaseDistiller


class TinyTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids=[ord(c)%61 for c in text]
        return {"input_ids":torch.tensor([ids],dtype=torch.long)}


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(1))


class Teacher(FrozenTeacher):
    def __init__(self):
        super().__init__(
            TeacherSpec("org/base-teacher","0123456789abcdef0123456789abcdef01234567",0),
            TinyTeacher(), TinyTokenizer()
        )
    def generate(self,prompt,*,max_new_tokens=128):
        return "उत्तर"


def tiny_config():
    return NeuroCoreEdgeConfig(
        vocab_size=61,hidden_size=16,num_hidden_layers=3,num_experts=20,
        experts_per_token=2,expert_intermediate_size=8,attention_heads=4,
        kv_heads=2,sliding_window=8,recurrent_state_size=4,
        prelude_layers=1,core_unique_layers=1,core_repeats=1,coda_layers=1,
        meta_token_count=2,global_attention_layers=(1,)
    )


def make_trainer():
    model=NeuroCoreEdgeForCausalLM(tiny_config())
    trainer=TeacherToStudentBaseDistiller(
        teacher=Teacher(),student_model=model,student_tokenizer=TinyTokenizer(),
        optimizer_name="adamw"
    )
    return model,trainer


class BaseDistillationTests(unittest.TestCase):
    def test_base_stage_updates_student_and_freezes_teacher(self):
        torch.manual_seed(5)
        model,trainer=make_trainer()
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        report=trainer.train(["हिंदी दस्तावेज़"],steps=2)
        self.assertEqual(report["status"],"BASE_DISTILLATION_STEP_COMPLETED")
        self.assertTrue(report["teacher_frozen"])
        self.assertTrue(report["all_student_parameters_trainable"])
        self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters()))
        self.assertTrue(all(not p.requires_grad for p in trainer.teacher.model.parameters()))

    def test_empty_prompt_set_rejected(self):
        _,trainer=make_trainer()
        with self.assertRaises(ValueError):
            trainer.train([],steps=1)

    def test_checkpoint_roundtrip_and_provenance_guard(self):
        torch.manual_seed(7)
        model,trainer=make_trainer()
        trainer.train(["पहला"],steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/"base.pt")
            kwargs=dict(tokenizer_fingerprint="tok-sha",prompt_fingerprint="prompt-sha",
                        teacher_revision=trainer.teacher.spec.revision)
            trainer.save_checkpoint(path,**kwargs)
            restored_model,restored=make_trainer()
            result=restored.load_checkpoint(path,**kwargs)
            self.assertEqual(result["status"],"BASE_DISTILLATION_RESUMED")
            self.assertEqual(restored.steps_completed,1)
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,restored_model.state_dict()[key]))
            with self.assertRaisesRegex(ValueError,"prompt_fingerprint"):
                make_trainer()[1].load_checkpoint(
                    path,tokenizer_fingerprint="tok-sha",prompt_fingerprint="wrong",
                    teacher_revision=trainer.teacher.spec.revision
                )


if __name__=="__main__":
    unittest.main()
