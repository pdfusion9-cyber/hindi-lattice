import unittest
from audit_indiccorp_hindi_sample import audit_rows

class IndicCorpHindiAuditTests(unittest.TestCase):
    def test_bounded_aggregate_audit(self):
        rows=[
            {"text":"यह सामान्य हिंदी वाक्य है।"},
            {"text":"यह सामान्य हिंदी वाक्य है।"},
            {"text":"संपर्क 9876543210"},
            {"text":"ईमेल demo@example.org"},
            {"text":"आधार 1234 5678 9012"},
            {"text":"PAN ABCDE1234F"},
            {"text":"https://example.org"},
            {"text":""},
            {"text":None},
        ]
        report=audit_rows(rows,7)
        self.assertEqual(report["rows_seen"],7)
        self.assertEqual(report["nonempty_text"],7)
        self.assertEqual(report["duplicate_text_rows"],1)
        self.assertEqual(report["phone_like_rows"],1)
        self.assertEqual(report["email_like_rows"],1)
        self.assertEqual(report["aadhaar_like_rows"],1)
        self.assertEqual(report["pan_like_rows"],1)
        self.assertEqual(report["url_like_rows"],1)
        self.assertFalse(report["sample_text_persisted"])
        self.assertEqual(report["approval_status"],"PENDING_REVIEW")
        self.assertFalse(report.get("training_performed",False))

    def test_no_sample_text_leaks(self):
        unique="THIS_MUST_NOT_APPEAR_IN_A_REPORT_482910"
        report=audit_rows([{"text":unique}],1)
        self.assertNotIn(unique,str(report))

    def test_limit_is_respected(self):
        report=audit_rows([{"text":"एक"},{"text":"दो"},{"text":"तीन"}],2)
        self.assertEqual(report["rows_seen"],2)

if __name__=="__main__":
    unittest.main()
