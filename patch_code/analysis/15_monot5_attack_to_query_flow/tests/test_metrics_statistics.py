"""L (metrics) + aggregation/bootstrap/sign-flip/BH unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from exp15lib.metrics import recovery_metrics
from exp15lib.statistics import benjamini_hochberg, bootstrap_within_attack, equal_weight_mean, sign_flip_test


def test_L_forward_reverse_combined_no_clipping():
    m = recovery_metrics(score_control=-5.0, score_attack=-1.0, score_fwd_patched=-3.0, score_rev_patched=-4.0)
    assert m["delta"] == 4.0
    assert m["raw_fwd"] == 2.0 and m["e_fwd"] == 0.5
    assert m["raw_rev"] == 3.0 and m["e_rev"] == 0.75
    assert m["e_combined"] == min(m["e_fwd"], m["e_rev"]) == 0.5
    m = recovery_metrics(-5.0, -1.0, 3.0, 2.0)          # beyond attack / beyond control
    assert m["e_fwd"] == 2.0 and m["e_rev"] == -0.75 and m["e_combined"] == -0.75
    m = recovery_metrics(-5.0, -1.0, -6.0, -1.0)
    assert m["e_fwd"] == -0.25 and m["e_rev"] == 0.0 and m["e_combined"] == -0.25


def test_L_unsuccessful_example_rejected():
    with pytest.raises(ValueError):
        recovery_metrics(1.0, 1.0 + 1e-5, 1.0, 1.0)


def test_equal_weight_is_mean_of_attack_means():
    g = [np.array([1.0] * 50), np.array([0.0] * 5)]
    assert equal_weight_mean(g) == 0.5            # pooled mean would be ~0.91


def test_bootstrap_deterministic_and_within_attack():
    g = [np.array([0.0, 1.0, 2.0]), np.array([5.0, 5.0])]
    a = bootstrap_within_attack(g, 2000, 0.95, 42, 0)
    b = bootstrap_within_attack(g, 2000, 0.95, 42, 0)
    assert a == b
    # attack 2 is constant, so the CI only reflects attack 1's resampling: mean in [2.5, 3.5]
    assert 2.5 <= a["ci_low"] <= 3.0 <= a["ci_high"] <= 3.5
    const = bootstrap_within_attack([np.array([2.0, 2.0])] * 3, 500, 0.95, 1, 0)
    assert const["ci_low"] == const["ci_high"] == 2.0


def test_sign_flip_properties():
    strong = sign_flip_test(np.full(20, 0.3), 5000, 42)
    assert strong["p_one_sided"] == pytest.approx(1 / 5001, rel=0.5)
    null = sign_flip_test(np.random.default_rng(0).normal(0, 1, 105), 5000, 42)
    assert 0.02 < null["p_one_sided"] < 0.98
    neg = sign_flip_test(np.full(20, -0.3), 2000, 42)
    assert neg["p_one_sided"] > 0.99
    assert sign_flip_test(np.full(5, 0.3), 1000, 7) == sign_flip_test(np.full(5, 0.3), 1000, 7)


def test_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    r = benjamini_hochberg(p, 0.05)
    # sorted p: .005 .01 .03 .04 -> p*m/k: .02 .02 .04 .04
    assert r["q"] == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert r["reject"] == [True, True, True, True]
    r = benjamini_hochberg([0.5, 0.9], 0.05)
    assert r["reject"] == [False, False] and max(r["q"]) <= 1.0
