"""
exp14lib/metrics.py
=====================
Recovery and clean-damage metrics (task spec sections 9-10).

Recovery = (|S_a - S_c| - |S_d - S_c|) / |S_a - S_c|
  1   = full recovery to control
  0   = no defensive effect
  < 0 = defense made the attack worse
  > 1 = intervention moved beyond the control score (task spec's stated
        interpretation; note the literal formula above is mathematically
        bounded above by 1 since |S_d - S_c| >= 0 can't be negative -- see
        DECISIONS.md item 12. Implemented exactly as specified regardless.)
NEVER clipped (explicit instruction, task spec section 9 + 21).

GlobalRecovery = sum_i(|S_a,i - S_c,i| - |S_d,i - S_c,i|) / sum_i(|S_a,i - S_c,i|)
  A denominator-pooled aggregate so tiny per-example |attack - control| gaps
  don't dominate the mean of the normalized metric.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence, Tuple


def recovery(score_attack: float, score_control: float, score_defended: float) -> float:
    denom = abs(score_attack - score_control)
    if denom == 0.0:
        raise ValueError(
            "recovery() denominator |score_attack - score_control| is exactly 0 — "
            "this example should have been filtered out by the success threshold "
            "before reaching this function."
        )
    return (denom - abs(score_defended - score_control)) / denom


def raw_movement_toward_control(score_attack: float, score_control: float, score_defended: float) -> float:
    """Unnormalized score movement: positive = moved toward control."""
    return abs(score_attack - score_control) - abs(score_defended - score_control)


def global_recovery(triples: Sequence[Tuple[float, float, float]]) -> float:
    """triples: (score_attack, score_control, score_defended) per example."""
    num = sum(abs(a - c) - abs(d - c) for a, c, d in triples)
    den = sum(abs(a - c) for a, c, _ in triples)
    if den == 0.0:
        raise ValueError("global_recovery() denominator is 0 — empty or degenerate example set.")
    return num / den


def recovery_summary(values: List[float]) -> Dict[str, float]:
    """Per-condition aggregate recovery statistics (task spec section 9)."""
    if not values:
        return {
            "n": 0, "mean_recovery": None, "median_recovery": None, "std_recovery": None,
            "frac_recovery_gt_0": None, "frac_recovery_ge_1": None,
        }
    return {
        "n": len(values),
        "mean_recovery": statistics.mean(values),
        "median_recovery": statistics.median(values),
        "std_recovery": statistics.stdev(values) if len(values) > 1 else 0.0,
        "frac_recovery_gt_0": sum(1 for v in values if v > 0) / len(values),
        "frac_recovery_ge_1": sum(1 for v in values if v >= 1) / len(values),
    }


def clean_damage_summary(
    clean_pairs: List[Tuple[float, float]],
    thresholds: List[float],
) -> Dict[str, float]:
    """
    clean_pairs: (score_clean, score_clean_defended) per example.
    Reports mean/median absolute change, mean signed change, std, and the
    fraction of examples whose |change| exceeds each configured threshold.
    """
    if not clean_pairs:
        result = {"n": 0, "mean_abs_change": None, "median_abs_change": None,
                  "mean_signed_change": None, "std_abs_change": None}
        for t in thresholds:
            result[f"frac_abs_change_gt_{t}"] = None
        return result

    abs_changes = [abs(d - c) for c, d in clean_pairs]
    signed_changes = [d - c for c, d in clean_pairs]
    result = {
        "n": len(clean_pairs),
        "mean_abs_change": statistics.mean(abs_changes),
        "median_abs_change": statistics.median(abs_changes),
        "mean_signed_change": statistics.mean(signed_changes),
        "std_abs_change": statistics.stdev(abs_changes) if len(abs_changes) > 1 else 0.0,
    }
    for t in thresholds:
        result[f"frac_abs_change_gt_{t}"] = sum(1 for v in abs_changes if v > t) / len(abs_changes)
    return result
