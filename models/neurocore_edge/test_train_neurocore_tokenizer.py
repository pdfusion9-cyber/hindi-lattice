import json, tempfile, unittest
from pathlib import Path
from train_neurocore_tokenizer import load_approved_sources, SPECIAL_TOKENS

def valid_source(**overrides):
    row={"dataset_id":"example/data","revision":"a"*40,"config_name":"default","split":"train",
      "license_id":"MIT","license_terms_url":"https://example.org/license","approval_id":"APP-1",
      "reviewer":"dataset-reviewer","reviewed_at":"2026-11-25T10:00:00Z",
      "approved":True,"commercial_training_allowed":True,"pii_reviewed":True,
      "consent_reviewed":True,"schema_reviewed":True,"split_reviewed":True,"text_columns":["text"]}
    row.update(overrides)
    return row

def write_manifest(path, approval):
    path.write_text(json.dumps({"tokenizer_source_approval":approval}))

class TokenizerApprovalTests(unittest.TestCase):
    def test_default_deny_manifest_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            write_manifest(p,{"status":"PENDING","approved_sources":[]})
            with self.assertRaisesRegex(RuntimeError,"no explicitly approved"):
                load_approved_sources(p)

    def test_unapproved_source_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            write_manifest(p,{"status":"APPROVED","approved_sources":[valid_source(approved=False)]})
            with self.assertRaisesRegex(RuntimeError,"not approved"):
                load_approved_sources(p)

    def test_missing_provenance_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            write_manifest(p,{"status":"APPROVED","approved_sources":[{"approved":True}]})
            with self.assertRaisesRegex(RuntimeError,"incomplete"):
                load_approved_sources(p)

    def test_mutable_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"approval.json"
            write_manifest(p,{"status":"APPROVED","approved_sources":[valid_source(revision="main")]})
            with self.assertRaisesRegex(RuntimeError,"immutable 40-char"):
                load_approved_sources(p)

    def test_127_reserved_control_tokens(self):
        controls=[x for x in SPECIAL_TOKENS if x.startswith("<|control_")]
        self.assertEqual(len(controls),127)
        self.assertEqual(len(SPECIAL_TOKENS),131)
        self.assertEqual(len(set(SPECIAL_TOKENS)),131)

if __name__=="__main__": unittest.main()
