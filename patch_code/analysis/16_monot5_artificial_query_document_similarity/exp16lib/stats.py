"""
exp16lib/stats.py
==================
Spearman / Pearson association, one-sided sign-flip test and
Benjamini-Hochberg FDR (DECISIONS 9, 32-35).

Sign-flip (H1: mean > 0): statistic = equal-weight mean of the K attack-level
values; under the symmetric null each value's sign is flipped independently
with prob 1/2 (Monte Carlo). p = (#{null >= observed} + 1) / (N + 1).
RNG: numpy Generator seeded with SeedSequence([seed, stream]); each
checkpoint uses stream = checkpoint_index. Same convention as Exp 15.

BH: standard step-up adjusted q-values (monotone, capped at 1).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as _st


def spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = _st.spearmanr(x, y)
    return {"n": int(len(x)), "rho": float(r.statistic), "p": float(r.pvalue)}


def pearson(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    r = _st.pearsonr(np.asarray(x, float), np.asarray(y, float))
    return {"r": float(r.statistic), "p": float(r.pvalue)}


def sign_flip_test(values: Sequence[float], n_flips: int, seed: int, stream: int = 0,
                   chunk: int = 10000) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("sign_flip_test needs a non-empty 1-D sample")
    observed = float(x.mean())
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
    extreme, done = 0, 0
    while done < n_flips:
        n = min(chunk, n_flips - done)
        signs = rng.integers(0, 2, size=(n, len(x))) * 2 - 1
        extreme += int(np.count_nonzero((signs * x).mean(axis=1) >= observed))
        done += n
    return {"observed": observed, "n_extreme": extreme, "n_flips": int(n_flips),
            "p_one_sided": (extreme + 1) / (n_flips + 1)}


def benjamini_hochberg(pvals: Sequence[float], alpha: float) -> Dict[str, List]:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return {"q": q.tolist(), "reject": (q <= alpha).tolist()}
