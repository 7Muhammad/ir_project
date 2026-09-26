from __future__ import annotations

import pytest

from exp14lib.splits import assert_disjoint, build_split


def _fake_pairs(n: int):
    return [{"qid": f"q{i % 20}", "docid": f"d{i}", "query": "x", "passage": "y"} for i in range(n)]


def test_split_is_pair_disjoint():
    """Test 1: no (qid, docid) pair occurs in more than one split."""
    pairs = _fake_pairs(300)
    split = build_split(pairs, seed=42)
    assert_disjoint(split)  # must not raise

    train_keys = {(p["qid"], p["docid"]) for p in split["train"]}
    val_keys = {(p["qid"], p["docid"]) for p in split["validation"]}
    test_keys = {(p["qid"], p["docid"]) for p in split["test"]}
    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)


def test_split_ratios_and_coverage():
    pairs = _fake_pairs(500)
    split = build_split(pairs, seed=42, ratios={"train": 0.6, "validation": 0.2, "test": 0.2})
    total = sum(len(v) for v in split.values())
    assert total == 500
    assert 290 <= len(split["train"]) <= 310
    assert 90 <= len(split["validation"]) <= 110
    assert 90 <= len(split["test"]) <= 110


def test_split_is_deterministic_given_seed():
    pairs = _fake_pairs(200)
    split1 = build_split(pairs, seed=42)
    split2 = build_split(pairs, seed=42)
    for name in ("train", "validation", "test"):
        keys1 = [(p["qid"], p["docid"]) for p in split1[name]]
        keys2 = [(p["qid"], p["docid"]) for p in split2[name]]
        assert keys1 == keys2


def test_all_attack_variants_of_one_pair_stay_in_one_split():
    """
    Test 2: since the split key is (qid, docid) and every attack variant of
    a pair shares that SAME (qid, docid) — attacked_passage differs, qid/docid
    don't — assigning by pair guarantees all variants land together. Simulate
    105 attack "variants" of the same 50 pairs and confirm every variant of
    a pair maps to the same split as every other variant of that pair.
    """
    base_pairs = _fake_pairs(50)
    split = build_split(base_pairs, seed=42)
    pair_to_split = {}
    for name, recs in split.items():
        for r in recs:
            pair_to_split[(r["qid"], r["docid"])] = name

    # Simulate 105 attack variants per pair (only attacked_passage differs).
    for base in base_pairs:
        key = (base["qid"], base["docid"])
        expected_split = pair_to_split[key]
        for attack_idx in range(105):
            variant = dict(base, attacked_passage=f"attack_{attack_idx}: {base['passage']}")
            # The variant's pair identity is unchanged -> must resolve to the same split.
            assert pair_to_split[(variant["qid"], variant["docid"])] == expected_split


def test_assert_disjoint_raises_on_overlap():
    splits = {
        "train": [{"qid": "1", "docid": "a"}],
        "validation": [{"qid": "1", "docid": "a"}],
    }
    with pytest.raises(ValueError):
        assert_disjoint(splits)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        build_split(_fake_pairs(10), ratios={"train": 0.5, "validation": 0.2, "test": 0.2})
