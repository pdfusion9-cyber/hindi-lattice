import tempfile
import unittest
from pathlib import Path
from micro_agent_runtime import NeuroCoreMicroAgent, safe_calculate

class MicroAgentTests(unittest.TestCase):
    def test_calculator_basic_arithmetic(self):
        self.assertEqual(safe_calculate("(12 + 8) * 3"), 60)
        self.assertEqual(safe_calculate("2 ** 8"), 256)
    def test_calculator_rejects_code_and_names(self):
        for value in ("__import__('os')", "open('/etc/passwd')", "x + 1", "[1,2]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_calculate(value)
    def test_calculator_bounds(self):
        for value in ("2 ** 9", "10 ** 20", "1 / 0", "1" * 300):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_calculate(value)
    def test_allowlisted_tool_executes(self):
        agent=NeuroCoreMicroAgent()
        result=agent.handle({"tool":"calculator","arguments":{"expression":"144 / 12"}})
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["result"],12)
    def test_unknown_tool_rejected_without_execution(self):
        agent=NeuroCoreMicroAgent()
        self.assertEqual(agent.handle({"tool":"shell","arguments":{"command":"id"}})["status"],"rejected")
    def test_bad_arguments_rejected(self):
        agent=NeuroCoreMicroAgent()
        self.assertEqual(agent.handle({"tool":"calculator","arguments":{"expression":"1+1","extra":1}})["status"],"rejected")
    def test_security_audit_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"audit.jsonl"
            agent=NeuroCoreMicroAgent(str(path))
            agent.handle({"tool":"calculator","arguments":{"expression":"1+1"}})
            self.assertTrue(path.exists())
            self.assertIn('"action": "allow"',path.read_text())
if __name__=="__main__": unittest.main()
