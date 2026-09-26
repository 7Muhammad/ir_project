from __future__ import annotations

import pytest

from exp14lib.metrics import (
    clean_damage_summary,
    global_recovery,
    raw_movement_toward_control,
    recovery,
    recovery_summary,
)


def test_recovery_full_recovery_is_one():
    # score_defended == score_control -> full recovery
    assert recovery(score_attack=5.0, score_control=1.0, score_defended=1.0) == pytest.approx(1.0)


def test_recovery_no_effect_is_zero():
    # score_defended == score_attack -> no defensive effect
    assert recovery(score_attack=5.0, score_control=1.0, score_defended=5.0) == pytest.approx(0.0)


def test_recovery_can_be_negative():
    # defense pushes score further from control than the attack itself
    r = recovery(score_attack=5.0, score_control=1.0, score_defended=9.0)
    assert r < 0


def test_recovery_is_never_clipped_even_when_defense_overshoots():
    # Under the literal formula (|S_a-S_c| - |S_d-S_c|) / |S_a-S_c|, recovery is
    # mathematically bounded above by 1 (since |S_d-S_c| >= 0) but NOT clipped by
    # this implementation -- it always returns the exact computed ratio, however
    # far negative an overshoot pushes it (see DECISIONS.md item 12).
    r = recovery(score_attack=5.0, score_control=1.0, score_defended=0.0)
    assert r == pytest.approx((4.0 - 1.0) / 4.0)  # = 0.75, not clamped/rounded
    r_far_overshoot = recovery(score_attack=5.0, score_control=1.0, score_defended=20.0)
    assert r_far_overshoot == pytest.approx((4.0 - 19.0) / 4.0)  # far negative, not clipped at e.g. -1


def test_recovery_raises_on_zero_denominator():
    with pytest.raises(ValueError):
        recovery(score_attack=1.0, score_control=1.0, score_defended=0.5)


def test_raw_movement_toward_control_sign():
    assert raw_movement_toward_control(5.0, 1.0, 1.0) > 0  # moved toward control
    assert raw_movement_toward_control(5.0, 1.0, 9.0) < 0  # moved away


def test_global_recovery_matches_manual_computation():
    triples = [(5.0, 1.0, 1.0), (3.0, 1.0, 3.0), (10.0, 0.0, 5.0)]
    expected_num = (4 - 0) + (2 - 2) + (10 - 5)
    expected_den = 4 + 2 + 10
    assert global_recovery(triples) == pytest.approx(expected_num / expected_den)


def test_global_recovery_is_denominator_pooled_not_a_simple_mean():
    """
    Many tiny-|attack-control|-gap examples with "perfect" recovery should NOT
    dominate global_recovery the way they dominate a naive mean of per-example
    normalized recovery (task spec section 9's rationale for this metric).
    """
    tiny_perfect = [(1.0001, 1.0, 1.0)] * 9   # denom=0.0001 each, recovery=1.0 each
    one_big_failure = [(10.0, 0.0, 10.0)]      # denom=10, defended==attack -> recovery=0.0
    triples = tiny_perfect + one_big_failure

    gr = global_recovery(triples)
    per_example_mean = sum(recovery(a, c, d) for a, c, d in triples) / len(triples)

    # naive mean is misleadingly high: 9 "perfect" tiny examples dominate the average
    assert per_example_mean == pytest.approx(0.9)
    # global recovery correctly reflects that the one large-delta example (the
    # only one that matters in absolute terms) saw ZERO recovery
    assert gr == pytest.approx(0.0009 / 10.0009, abs=1e-6)
    assert gr < per_example_mean


def test_recovery_summary_basic_stats():
    values = [1.0, 0.5, 0.0, -0.5, 1.5]
    stats = recovery_summary(values)
    assert stats["n"] == 5
    assert stats["mean_recovery"] == pytest.approx(0.5)
    assert stats["frac_recovery_gt_0"] == pytest.approx(3 / 5)
    assert stats["frac_recovery_ge_1"] == pytest.approx(2 / 5)


def test_recovery_summary_empty():
    stats = recovery_summary([])
    assert stats["n"] == 0
    assert stats["mean_recovery"] is None


def test_clean_damage_summary_thresholds():
    pairs = [(1.0, 1.02), (2.0, 2.5), (3.0, 3.0)]
    stats = clean_damage_summary(pairs, thresholds=[0.05, 0.3])
    assert stats["n"] == 3
    assert stats["mean_abs_change"] == pytest.approx((0.02 + 0.5 + 0.0) / 3)
    assert stats["frac_abs_change_gt_0.05"] == pytest.approx(1 / 3)  # only the 0.5 change exceeds 0.05
    assert stats["frac_abs_change_gt_0.3"] == pytest.approx(1 / 3)
