import json, tempfile, unittest
from pathlib import Path
from train_neurocore_tokenizer import load_approved_sources, SPECIAL_TOKENS, validate_tokenizer_object, ROUND_TRIP_SAMPLES, EXPECTED_VOCAB_SIZE

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


class FakeEncoding:
    def __init__(self, ids):
        self.ids=ids

class FakeTokenizer:
    def __init__(self):
        self.ids={token:i for i,token in enumerate(SPECIAL_TOKENS)}
        self.vocab_size=EXPECTED_VOCAB_SIZE
    def get_vocab_size(self,with_added_tokens=True):
        return self.vocab_size
    def token_to_id(self,token):
        return self.ids.get(token)
    def encode(self,text,add_special_tokens=True):
        if text in self.ids:
            return FakeEncoding([self.ids[text]])
        return FakeEncoding([1000+ROUND_TRIP_SAMPLES.index(text)])
    def decode(self,ids,skip_special_tokens=True):
        return ROUND_TRIP_SAMPLES[ids[0]-1000]

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

    def test_tokenizer_validation_checks_vocab_and_unicode(self):
        report=validate_tokenizer_object(FakeTokenizer())
        self.assertEqual(report["vocab_size"],65536)
        self.assertEqual(report["control_token_count"],127)
        self.assertEqual(report["unicode_round_trip"],"PASS")
        self.assertEqual(report["round_trip_samples"],len(ROUND_TRIP_SAMPLES))

    def test_wrong_vocab_is_rejected(self):
        tokenizer=FakeTokenizer()
        tokenizer.vocab_size=65535
        with self.assertRaisesRegex(RuntimeError,"expected 65536"):
            validate_tokenizer_object(tokenizer)

    def test_127_reserved_control_tokens(self):
        controls=[x for x in SPECIAL_TOKENS if x.startswith("<|control_")]
        self.assertEqual(len(controls),127)
        self.assertEqual(len(SPECIAL_TOKENS),131)
        self.assertEqual(len(set(SPECIAL_TOKENS)),131)

if __name__=="__main__": unittest.main()
