import unittest
from dryrun_synthetic_dhir2b_sft import verify_mask

class DryRunLossMaskTests(unittest.TestCase):
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
