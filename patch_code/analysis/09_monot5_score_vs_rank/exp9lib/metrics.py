"""
exp9lib/metrics.py
=====================
Per-example and aggregate score-vs-rank metrics.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from scipy.stats import spearmanr


def success(delta_rank: int) -> int:
    """1 if the target's rank improved (moved toward rank 1) under attack."""
    return 1 if delta_rank > 0 else 0


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """
    Spearman rho between two equal-length sequences. Returns None if there
    are fewer than 2 points or either sequence is constant (undefined
    correlation) — scipy returns nan in that case, so we normalize to None.
    """
    if len(x) < 2 or len(y) < 2:
        return None
    rho, _p = spearmanr(x, y)
    if rho != rho:  # NaN check without importing math/numpy just for this
        return None
    return float(rho)
