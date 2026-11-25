import json, tempfile, unittest
from pathlib import Path
from train_neurocore_tokenizer import load_approved_sources, SPECIAL_TOKENS

class TokenizerApprovalTests(unittest.TestCase):
    def test_default_deny_manifest_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            p.write_text(json.dumps({"status":"ALL_DATASETS_PENDING_REVIEW","approved_tokenizer_sources":[]}))
            with self.assertRaisesRegex(RuntimeError,"no explicitly approved"):
                load_approved_sources(p)
    def test_unapproved_source_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            source={"dataset_id":"example/data","revision":"a"*40,"config_name":"default","split":"train",
              "license_id":"MIT","license_terms_url":"https://example.org/license","approval_id":"APP-1",
              "approved":False,"commercial_training_allowed":True,"pii_reviewed":True,
              "consent_reviewed":True,"schema_reviewed":True,"split_reviewed":True,"text_columns":["text"]}
            p.write_text(json.dumps({"status":"TOKENIZER_SOURCES_APPROVED","approved_tokenizer_sources":[source]}))
            with self.assertRaisesRegex(RuntimeError,"not approved"):
                load_approved_sources(p)
    def test_missing_provenance_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            p.write_text(json.dumps({"status":"TOKENIZER_SOURCES_APPROVED","approved_tokenizer_sources":[{"approved":True}]}))
            with self.assertRaisesRegex(RuntimeError,"incomplete"):
                load_approved_sources(p)
    def test_127_reserved_control_tokens(self):
        controls=[x for x in SPECIAL_TOKENS if x.startswith("<|control_")]
        self.assertEqual(len(controls),127)
        self.assertEqual(len(SPECIAL_TOKENS),131)

if __name__=="__main__": unittest.main()
