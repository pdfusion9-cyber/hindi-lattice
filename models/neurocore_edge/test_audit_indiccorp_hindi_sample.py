import unittest
from audit_indiccorp_hindi_sample import audit_rows

class IndicCorpHindiAuditTests(unittest.TestCase):
    def test_bounded_aggregate_audit_counts_nonempty_records(self):
        rows=[
            {"text":""},
            {"text":"यह सामान्य हिंदी वाक्य है।"},
            {"text":"   "},
            {"text":"यह सामान्य हिंदी वाक्य है।"},
            {"text":"संपर्क 9876543210"},
            {"text":"ईमेल demo@example.org"},
            {"text":"आधार 1234 5678 9012"},
            {"text":"PAN ABCDE1234F"},
            {"text":"https://example.org"},
        ]
        report=audit_rows(rows,8,20)
        self.assertEqual(report["rows_seen"],9)
        self.assertEqual(report["nonempty_text"],8)
        self.assertEqual(report["empty_text"],1)
        self.assertEqual(report["duplicate_text_rows"],1)
        self.assertEqual(report["phone_like_rows"],1)
        self.assertEqual(report["email_like_rows"],1)
        self.assertEqual(report["aadhaar_like_rows"],1)
        self.assertEqual(report["pan_like_rows"],1)
        self.assertEqual(report["url_like_rows"],1)
        self.assertTrue(report["nonempty_target_reached"])
        self.assertFalse(report["scan_budget_exhausted"])
        self.assertFalse(report["sample_text_persisted"])
        self.assertEqual(report["approval_status"],"PENDING_REVIEW")
        self.assertFalse(report.get("training_performed",False))

    def test_empty_separators_do_not_consume_nonempty_quota(self):
        rows=[{"text":""},{"text":"एक हिंदी रिकॉर्ड"},{"text":""},{"text":"दूसरा हिंदी रिकॉर्ड"}]
        report=audit_rows(rows,2,10)
        self.assertEqual(report["nonempty_text"],2)
        self.assertEqual(report["empty_text"],2)
        self.assertEqual(report["rows_seen"],4)
        self.assertTrue(report["nonempty_target_reached"])

    def test_scan_limit_is_a_hard_bound(self):
        rows=[{"text":""} for _ in range(7)]+[{"text":"यह रिकॉर्ड बाद में है"}]
        report=audit_rows(rows,1,5)
        self.assertEqual(report["rows_seen"],5)
        self.assertEqual(report["nonempty_text"],0)
        self.assertTrue(report["scan_budget_exhausted"])
        self.assertFalse(report["nonempty_target_reached"])

    def test_no_sample_text_leaks(self):
        unique="THIS_MUST_NOT_APPEAR_IN_A_REPORT_482910"
        report=audit_rows([{"text":unique}],1,10)
        self.assertNotIn(unique,str(report))

    def test_nonempty_limit_is_respected(self):
        report=audit_rows([{"text":"एक"},{"text":"दो"},{"text":"तीन"}],2,10)
        self.assertEqual(report["rows_seen"],2)
        self.assertEqual(report["nonempty_text"],2)
        self.assertTrue(report["nonempty_target_reached"])

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            audit_rows([],501,1000)
        with self.assertRaises(ValueError):
            audit_rows([],5,4)

if __name__=="__main__":
    unittest.main()
