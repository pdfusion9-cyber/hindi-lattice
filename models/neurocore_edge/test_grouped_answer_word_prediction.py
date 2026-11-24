from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "models" / "neurocore_edge"))

from neurocore_native_grouped_answer_word_prediction import (  # noqa: E402
    ALT,
    EOS,
    SPECIAL,
    audit_groups,
    collate,
    encode_group,
    load_groups,
    make_vocab,
    split_by_group,
)


def test_loader_groups_all_answers_and_deduplicates_normalized_objects(tmp_path):
    corpus = tmp_path / "facts.txt"
    corpus.write_text(
        "दक्षिण अफ्रीका|राजधानी|प्रिटोरिया\n"
        "दक्षिण अफ्रीका|राजधानी|केप टाउन\n"
        "दक्षिण अफ्रीका|राजधानी|प्रिटोरिया\n"
        "भारत|राजधानी|नई दिल्ली\n",
        encoding="utf-8",
    )
    _, rows, groups, counts, markup = load_groups(corpus)
    assert len(rows) == 4
    assert len(groups) == 2
    multi = next(group for group in groups if group["subject"] == "दक्षिण अफ्रीका")
    assert multi["objects"] == ["केप टाउन", "प्रिटोरिया"]
    assert counts["राजधानी"] == 4
    assert audit_groups(rows, groups, markup)["multi_answer_groups"] == 1


def test_group_split_never_leaks_subject_relation_pairs():
    groups = [
        {"subject": f"विषय {i}", "relation": "संबंध", "objects": [f"उत्तर {i}"]}
        for i in range(200)
    ]
    train, valid = split_by_group(groups, validation_percent=20)
    train_keys = {(g["subject"], g["relation"]) for g in train}
    valid_keys = {(g["subject"], g["relation"]) for g in valid}
    assert train and valid
    assert train_keys.isdisjoint(valid_keys)


def test_target_encodes_all_answers_with_alt_and_eos():
    group = {
        "subject": "दक्षिण अफ्रीका",
        "relation": "राजधानी",
        "objects": ["केप टाउन", "प्रिटोरिया"],
    }
    word_to_id, _ = make_vocab({
        "दक्षिण": 1, "अफ्रीका": 1, "राजधानी": 1,
        "केप": 1, "टाउन": 1, "प्रिटोरिया": 1,
    })
    prompt, target = encode_group(group, word_to_id)
    assert target.count(ALT) == 1
    assert target[-1] == EOS
    assert len(target) == 6


def test_collate_masks_prompt_and_padding_from_loss():
    word_to_id, _ = make_vocab({"विषय": 1, "संबंध": 1, "उत्तर": 1})
    examples = [
        encode_group({"subject": "विषय", "relation": "संबंध", "objects": ["उत्तर"]}, word_to_id),
        encode_group({"subject": "विषय", "relation": "संबंध", "objects": ["उत्तर", "विषय"]}, word_to_id),
    ]
    x, y, loss_mask, word_mask = collate(examples)
    assert x.shape == y.shape == loss_mask.shape == word_mask.shape
    assert loss_mask.dtype == torch.bool
    assert word_mask.dtype == torch.bool
    assert torch.all(y[~loss_mask] == -100)
    assert torch.all(word_mask <= loss_mask)
    assert len(SPECIAL) == 6
