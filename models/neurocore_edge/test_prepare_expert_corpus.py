import json
from pathlib import Path

import pytest

from prepare_expert_corpus import load_experts, load_approved_rows, write_split

def test_manifest_has_exact_twenty_experts():
    path = Path(__file__).with_name("expert_manifest.json")
    rows = load_experts(path)
    assert len(rows) == 20
    assert {row["expert_id"] for row in rows} == set(range(20))

def test_default_dataset_approval_is_fail_closed():
    path = Path(__file__).with_name("dataset_approval.json")
    with pytest.raises(SystemExit, match="APPROVED_FOR_COMMERCIAL_TRAINING"):
        load_approved_rows(path)

def test_pinned_validation_rows_go_to_validation_not_train():
    from io import StringIO
    train, valid = StringIO(), StringIO()
    stream = [
        {"src": "I has a pen", "tgt": "I have a pen"},
        {"src": "She go home", "tgt": "She goes home"},
    ]
    rows, train_count, valid_count = write_split(
        "grammarly/coedit", 0, stream, train, valid,
        route_hash_to_validation=False, force_validation=True,
        max_examples=10,
    )
    assert rows == 2
    assert train_count == 0 and valid_count == 2
    assert train.getvalue() == ""
    assert "I have a pen" in valid.getvalue()
