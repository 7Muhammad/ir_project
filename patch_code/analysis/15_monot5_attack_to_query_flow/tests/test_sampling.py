"""K: deterministic seed-42 sampling, cap 50, successful instances only."""

from __future__ import annotations

import random

import pytest

from conftest import EXP1_ATTACKS

from exp15lib.sampling import load_pool, sample_attack, successful_instances
from src.patching import SKIP_EPSILON


def _synthetic(n_success, n_fail):
    pool = []
    for i in range(n_success + n_fail):
        ok = i < n_success
        c = -3.0
        a = c + (1.0 + i if ok else -0.5 if i % 2 else SKIP_EPSILON / 2)
        pool.append({"qid": str(i), "docid": "d", "query": "q", "passage": "p", "attacked_passage": "x p",
                     "control_score": c, "attack_score": a, "attack_delta_vs_control": a - c})
    random.Random(1).shuffle(pool)
    return pool


def test_K_deterministic_and_identical_twice():
    pool = load_pool(EXP1_ATTACKS, "relevant_start_5")
    s1 = sample_attack(pool, "relevant_start_5", 42, 50)
    s2 = sample_attack(load_pool(EXP1_ATTACKS, "relevant_start_5"), "relevant_start_5", 42, 50)
    assert [(e["qid"], e["docid"]) for e in s1] == [(e["qid"], e["docid"]) for e in s2]
    assert len(s1) == 50
    s3 = sample_attack(pool, "relevant_start_5", 43, 50)
    assert [e["qid"] + e["docid"] for e in s3] != [e["qid"] + e["docid"] for e in s1]


def test_K_cap_all_when_fewer_and_success_only():
    pool = _synthetic(20, 15)
    s = sample_attack(pool, "x", 42, 50)
    assert len(s) == 20
    assert all(e["attack_score"] - e["control_score"] > SKIP_EPSILON for e in s)
    s = sample_attack(_synthetic(80, 5), "x", 42, 50)
    assert len(s) == 50 and len({e["qid"] for e in s}) == 50
    assert all(e["attack_score"] - e["control_score"] > SKIP_EPSILON for e in s)


def test_K_matches_exp13_scheme_prefix():
    """Same per-attack seeding as Exp 13 -> Exp 13's n=10 sample is a prefix of ours."""
    from exp13lib.run_utils import select_examples_for_attack
    ours = sample_attack(load_pool(EXP1_ATTACKS, "true_end_2"), "true_end_2", 42, 50)
    theirs = select_examples_for_attack("true_end_2", EXP1_ATTACKS, 10, 42)
    assert [(e["qid"], e["docid"]) for e in ours[:10]] == [(e["qid"], e["docid"]) for e in theirs]


def test_K_inconsistent_cached_delta_fails_loudly():
    pool = _synthetic(3, 0)
    pool[0]["attack_delta_vs_control"] += 1.0
    with pytest.raises(ValueError):
        successful_instances(pool)
