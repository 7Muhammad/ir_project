"""
exp16lib/activation_stats.py
=============================
Stage 24: activation magnitude / structure statistics of the stage-22 per-token head outputs
(pre-o_proj, 64 dims per head; no forward pass). Every statistic is computed per head on one
token region X in R^(n_tokens x 64):

  l2        mean_t ||X_t||_2                                   primary "activation strength"
  mean      mean_{t,j} X[t,j]
  var       mean_t Var_j(X[t,j])                               population variance (ddof = 0) over the 64 dims
  maxabs    max_{t,j} |X[t,j]|
  eff_dim   (sum_j e_j)^2 / sum_j e_j^2,  e_j = mean_t X[t,j]^2   participation ratio, in [1, 64]
  top_share max_t p_t,  p_t = ||X_t||^2 / sum_t ||X_t||^2      in [1/n, 1]
  entropy   -sum_t p_t log p_t / log n                          normalised token-energy entropy, in [0, 1]

Edge-case conventions (all NaN = "not defined", never an invented value):
  n_tokens == 0            every statistic NaN
  n_tokens == 1            top_share = 1 (the formula), entropy = NaN (0/0: log 1 = 0)
  zero total energy        eff_dim, top_share, entropy NaN (p and e undefined); l2/mean/var/maxabs = 0
  p_t == 0                 contributes 0 to the entropy (lim p log p = 0)

Token regions (masks from metrics_screen.region_masks, never re-derived here):
  query  query_mask                                  (query text only)
  doc    doc_mask                                    (full document: attacked = passage + injected tokens)
  orig   doc_mask & ~inserted_mask                   (attacked: original passage; clean/control: = doc)
  ins    doc_mask &  inserted_mask                   (attacked only; clean/control: NaN)
Control insertion slots are masked pads outside doc_mask, so they never enter any region.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np

from .metrics_screen import region_masks

STATS = ["l2", "mean", "var", "maxabs", "eff_dim", "top_share", "entropy"]
REGIONS = ["query", "doc", "orig", "ins"]
LATE_LAYERS = (9, 10, 11)


def region_stats(X: np.ndarray) -> np.ndarray:
    """X [n_tokens, n_heads, d] (any float dtype) -> [len(STATS), n_heads] float64 (NaN where undefined)."""
    X = np.asarray(X, dtype=np.float64)
    n, nh = X.shape[0], X.shape[1]
    out = np.full((len(STATS), nh), np.nan)
    if n == 0:
        return out
    sq = X ** 2
    tok_e = sq.sum(-1)                                        # [n, h]  ||X_t||^2
    out[0] = np.sqrt(tok_e).mean(0)
    out[1] = X.mean((0, 2))
    out[2] = X.var(-1).mean(0)
    out[3] = np.abs(X).max((0, 2))
    e = sq.mean(0)                                            # [h, d]  per-dimension energy
    tot = e.sum(-1)
    ok = tot > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out[4] = np.where(ok, tot ** 2 / (e ** 2).sum(-1), np.nan)
        E = tok_e.sum(0)                                      # [h]
        p = tok_e / np.where(E > 0, E, np.nan)                # [n, h]
        out[5] = p.max(0)
        if n > 1:
            plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
            out[6] = np.where(E > 0, -plogp.sum(0) / np.log(n), np.nan)
    return out


def sequence_stats(H: np.ndarray, query_mask, doc_mask, inserted_mask, kind: str) -> Dict[str, Optional[np.ndarray]]:
    """
    H [T, n_heads, d] -> {region: [len(STATS), n_heads] or None (region not defined for this kind)}.
    Also returns the token count per region under key f"n_{region}".
    """
    reg = region_masks(query_mask, doc_mask, inserted_mask, kind)
    masks = {"query": np.asarray(query_mask, bool), "doc": reg["full"], "orig": reg["orig"], "ins": reg["ins"]}
    out: Dict = {}
    for r, m in masks.items():
        out[f"n_{r}"] = 0 if m is None else int(m.sum())
        out[r] = None if m is None else region_stats(H[m])
    return out


def pooled_late_doc(heads: np.ndarray, mask) -> np.ndarray:
    """
    "Late-layer pooled document head representation": heads [T, 12, 12, 64] -> mean over the masked
    tokens (fp32) of layers 9-11, flattened layer-major, head-major -> [3 * 12 * 64 = 2304].
    NOT a residual-stream representation (pre-o_proj head outputs only).
    """
    m = np.asarray(mask, bool)
    return heads[m][:, list(LATE_LAYERS)].astype(np.float32).mean(0).reshape(-1)


def size_matched_stats(X: np.ndarray, k: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Mean over `n_draws` random k-token subsets (without replacement) of X of region_stats -> [len(STATS), h]."""
    n = X.shape[0]
    if k <= 0 or n < k:
        return np.full((len(STATS), X.shape[1]), np.nan)
    acc = np.stack([region_stats(X[np.sort(rng.choice(n, k, replace=False))]) for _ in range(n_draws)])
    with warnings.catch_warnings():                           # all-NaN rows (entropy at k = 1) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(acc, 0)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """(mean a - mean b) / pooled SD (ddof 1); NaNs dropped. Positive = larger in a."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def paired_dz(delta: np.ndarray) -> float:
    """Paired standardised effect mean(delta) / SD(delta, ddof 1); NaNs dropped."""
    d = delta[~np.isnan(delta)]
    if len(d) < 2:
        return np.nan
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else np.nan
