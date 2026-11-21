import unittest
from pathlib import Path
from tools.expert_router import DOMAINS

class CompatibilityAuditSourceTests(unittest.TestCase):
    def test_ten_domains_are_unique(self):
        self.assertEqual(len(DOMAINS),10)
        self.assertEqual(len(set(DOMAINS)),10)
    def test_audit_script_exists(self):
        self.assertTrue(Path("tools/audit_expert_compat.py").is_file())
    def test_audit_is_read_only_by_design(self):
        source=Path("tools/audit_expert_compat.py").read_text(encoding="utf-8")
        self.assertIn("weights_only=True",source)
        self.assertIn("strict=True",source)
        self.assertIn("real_graph_forward",source)
        self.assertNotIn("torch.save(",source)
        self.assertNotIn("optimizer.step(",source)

if __name__=="__main__": unittest.main()
