import unittest
from audit_sangraha_hindi_sample import audit_rows

class SangrahaHindiAuditTests(unittest.TestCase):
    def test_aggregate_only_and_bounded_sample(self):
        rows=[
          {"doc_id":"1","text":"यह हिंदी नमूना है और इसमें कोई निजी जानकारी नहीं।"},
          {"doc_id":"2","text":"English text for comparison."},
          {"doc_id":"2","text":"संपर्क test@example.com"},
          {"doc_id":"4","text":""},
          {"doc_id":"5","text":None},
        ]
        report=audit_rows(rows,3)
        self.assertEqual(report["rows_seen"],3)
        self.assertEqual(report["nonempty_text"],3)
        self.assertEqual(report["duplicate_doc_id_rows"],1)
        self.assertEqual(report["email_like_rows"],1)
        self.assertFalse(report["sample_text_persisted"])
        self.assertEqual(report["approval_status"],"PENDING_REVIEW")

    def test_no_text_leaks_in_report(self):
        secret="UNIQUE_PRIVATE_SAMPLE_TEXT"
        report=audit_rows([{"doc_id":"1","text":secret}],1)
        self.assertNotIn(secret,str(report))

if __name__=="__main__": unittest.main()
