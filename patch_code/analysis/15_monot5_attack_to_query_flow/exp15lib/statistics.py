"""
exp15lib/statistics.py
=======================
Equal-weight aggregation, within-attack bootstrap CI, one-sided sign-flip
test and Benjamini-Hochberg FDR (DECISIONS.md items 25-30).

Equal weight: first average examples within each attack, then average the
attack-level means with equal weight per attack.

Bootstrap (percentile): per replicate, for every attack k resample its n_k
examples WITH replacement, take the attack mean, then average the attack
means equally. Attacks themselves are NOT resampled.

Sign-flip test (one-sided, H1: mean > 0): statistic = equal-weight mean of
the K attack-level means; under the symmetric null each attack mean's sign is
flipped independently with prob 1/2. p = (#{null >= observed} + 1) / (N + 1).

BH: standard step-up adjusted q-values (monotone, capped at 1).

All RNGs are numpy Generators seeded deterministically; `stream` lets each
head use an independent but reproducible stream (SeedSequence([seed, stream])).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def equal_weight_mean(groups: Sequence[np.ndarray]) -> float:
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError("equal_weight_mean needs non-empty groups")
    return float(np.mean([np.mean(g) for g in groups]))


def bootstrap_within_attack(
    groups: Sequence[np.ndarray], reps: int, level: float, seed: int, stream: int = 0,
) -> Dict[str, float]:
    rng = _rng(seed, stream)
    acc = np.zeros(reps, dtype=np.float64)
    for g in groups:
        g = np.asarray(g, dtype=np.float64)
        idx = rng.integers(0, len(g), size=(reps, len(g)))
        acc += g[idx].mean(axis=1)
    boot = acc / len(groups)
    alpha = 1.0 - level
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"ci_low": float(lo), "ci_high": float(hi), "boot_se": float(boot.std(ddof=1))}


def sign_flip_test(
    attack_means: Sequence[float], n_flips: int, seed: int, stream: int = 0, chunk: int = 10000,
) -> Dict[str, float]:
    x = np.asarray(attack_means, dtype=np.float64)
    observed = float(x.mean())
    rng = _rng(seed, stream)
    extreme = 0
    done = 0
    while done < n_flips:
        n = min(chunk, n_flips - done)
        signs = rng.integers(0, 2, size=(n, len(x))) * 2 - 1
        null = (signs * x).mean(axis=1)
        extreme += int(np.count_nonzero(null >= observed))
        done += n
    return {
        "observed": observed,
        "n_extreme": extreme,
        "n_flips": n_flips,
        "p_one_sided": (extreme + 1) / (n_flips + 1),
    }


def benjamini_hochberg(pvals: Sequence[float], alpha: float) -> Dict[str, List]:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return {"q": q.tolist(), "reject": (q <= alpha).tolist()}
