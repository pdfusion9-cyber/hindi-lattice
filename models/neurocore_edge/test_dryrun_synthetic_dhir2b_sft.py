import unittest
from dryrun_synthetic_dhir2b_sft import verify_mask, verify_registry_for_review_dry_run

class DryRunLossMaskTests(unittest.TestCase):
    def _scoped_registry(self):
        return {
            "status": "PARTIAL_APPROVAL_EXPERT16_INTERNAL_RESEARCH_ONLY",
            "datasets": [
                {"expert_id": 16, "dataset_id": "local/neurocore_synthetic_sft/expert16_self_correction",
                 "revision": "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9", "approved": True,
                 "internal_research_allowed": True, "approval_scope": "internal_research_only",
                 "pii_reviewed": True, "consent_reviewed": True, "schema_reviewed": True,
                 "split_reviewed": True, "commercial_training_allowed": False,
                 "public_redistribution_allowed": False, "approval_id": "owner-attested-expert16-internal-research-2026-11-27",
                 "reviewer": "owner", "reviewed_at": "2026-11-27"},
                {"expert_id": 17, "dataset_id": "local/neurocore_synthetic_sft/expert17_planning",
                 "revision": "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9", "approved": False,
                 "commercial_training_allowed": False, "public_redistribution_allowed": False,
                 "pii_reviewed": False, "consent_reviewed": False, "schema_reviewed": False,
                 "split_reviewed": False}
            ],
            "approved_training_artifacts": [{
                "expert_id": 16, "dataset_id": "local/neurocore_synthetic_sft/expert16_self_correction",
                "dataset_revision": "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9",
                "sha256": "ce148fd31b2850b18f0b398dd55cda12f30a33844b315c98dc8a16847ed23e66",
                "approved": True, "approval_scope": "internal_research_only",
                "commercial_training_allowed": False, "public_redistribution_allowed": False
            }],
            "approved_tokenizers": [{
                "sha256": "b" * 64, "approved": True, "approval_scope": "internal_research_only",
                "commercial_training_allowed": False, "public_redistribution_allowed": False,
                "approval_id": "owner-attested-expert16-internal-research-2026-11-27",
                "reviewer": "owner", "reviewed_at": "2026-11-27"
            }]
        }

    def test_scoped_expert16_internal_approval_passes(self):
        result = verify_registry_for_review_dry_run(self._scoped_registry(), "b" * 64)
        self.assertTrue(result["expert16_approved_internal_only"])
        self.assertFalse(result["expert17_approved"])

    def test_scoped_approval_rejects_commercial_and_expert17(self):
        registry = self._scoped_registry()
        registry["datasets"][0]["commercial_training_allowed"] = True
        with self.assertRaises(PermissionError):
            verify_registry_for_review_dry_run(registry, "b" * 64)
        registry = self._scoped_registry()
        registry["datasets"][1]["approved"] = True
        with self.assertRaises(PermissionError):
            verify_registry_for_review_dry_run(registry, "b" * 64)

    def test_prompt_masked_assistant_supervised(self):
        self.assertEqual(verify_mask([1,4,5],[1,4,5,8,9,2],
          {"input_ids":[1,4,5,8,9,2],"labels":[-100,-100,-100,8,9,2]}),(3,3,3,6))
    def test_prompt_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError,"loss mask"):
            verify_mask([1,4],[1,4,8,2],{"input_ids":[1,4,8,2],"labels":[1,-100,8,2]})
    def test_empty_completion_rejected(self):
        with self.assertRaisesRegex(ValueError,"completion is empty"):
            verify_mask([1,4],[1,4],{"input_ids":[1,4],"labels":[-100,-100]})
    def test_truncation_rejected(self):
        with self.assertRaisesRegex(ValueError,"max_length"):
            verify_mask([1,4],[1]*257,{"input_ids":[1]*256,"labels":[-100]*256})
    def test_bad_native_prefix_rejected(self):
        with self.assertRaisesRegex(ValueError,"prefix mismatch"):
            verify_mask([1,4],[1,5,8,2],{"input_ids":[1,5,8,2],"labels":[-100,-100,8,2]})

if __name__=="__main__": unittest.main()
