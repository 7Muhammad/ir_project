"""
exp16lib/metrics_screen.py
===========================
Stage 23: alternative query/document similarity metrics computed from the stage-22
per-token head outputs (no forward pass). All metrics are per head (144 heads).

Notation for one sequence and one head: query tokens q_1..q_n (query-TEXT mask), document
tokens d_1..d_m of region R, q̄ / d̄_R = means (fp32), cos in float64.

  cos      cos(q̄, d̄_R)                              baseline (= TokenSample.pooled_cosine for R = full)
  ccos     cos(q̄ - μ, d̄_R - μ)                      centered; μ = per-head background vector = mean of all
                                                   query-text + document token vectors of CLEAN sequences of
                                                   TRAINING queries (label-free, query-fold cross-validated)
  dot      q̄ · d̄_R                                  unnormalised
  qnorm    ||q̄||,  dnorm ||d̄_R||
  ms_mean  mean_i  max_j cos(q_i, d_j),  ms_median = median_i,  ms_top3 = mean of the k = min(3, n) largest
Regions R:
  full  every document token (control: the original passage; attacked: passage + injected tokens)
  orig  document tokens that are not inserted  (attacked sequences; = full for clean/control)
  ins   inserted attack tokens inside the document field (attacked sequences only)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

REGIONS = ["full", "orig", "ins"]
REGION_METRICS = ["cos", "ccos", "dot", "dnorm", "ms_mean", "ms_median", "ms_top3"]
COLUMNS = [f"{m}_{r}" for r in REGIONS for m in REGION_METRICS] + ["qnorm"]
K_TOP = 3
EPS = 1e-12


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), EPS)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def region_masks(query_mask, doc_mask, inserted_mask, kind: str) -> Dict[str, Optional[np.ndarray]]:
    """Boolean token masks per region (None = region not defined for this sequence kind)."""
    doc = doc_mask.astype(bool)
    ins = inserted_mask.astype(bool) & doc
    if kind == "attack":
        return {"full": doc, "orig": doc & ~ins, "ins": ins}
    if ins.any():
        raise ValueError(f"{kind} sequence has inserted tokens inside its document pool")
    return {"full": doc, "orig": doc, "ins": None}


def sequence_metrics(H: np.ndarray, query_mask, regions: Dict[str, Optional[np.ndarray]],
                     mu: np.ndarray) -> np.ndarray:
    """
    H [T, n_heads, d] (any float dtype), mu [n_heads, d] -> [len(COLUMNS), n_heads] float64 (NaN where undefined).
    """
    H = H.astype(np.float32)
    qm = np.asarray(query_mask, bool)
    Q = H[qm]                                             # [n, h, d]
    qbar = Q.mean(0)
    Qn = _unit(Q.astype(np.float64))
    nh = H.shape[1]
    out = {c: np.full(nh, np.nan) for c in COLUMNS}
    out["qnorm"] = np.linalg.norm(qbar.astype(np.float64), axis=-1)
    for r, m in regions.items():
        if m is None or not np.any(m):
            continue
        D = H[np.asarray(m, bool)]
        dbar = D.mean(0)
        out[f"cos_{r}"] = _cos(qbar, dbar)
        out[f"ccos_{r}"] = _cos(qbar - mu, dbar - mu)
        out[f"dot_{r}"] = (qbar.astype(np.float64) * dbar.astype(np.float64)).sum(-1)
        out[f"dnorm_{r}"] = np.linalg.norm(dbar.astype(np.float64), axis=-1)
        S = np.einsum("ihk,jhk->hij", Qn, _unit(D.astype(np.float64)))     # [h, n, m]
        mx = S.max(2)                                                         # [h, n]  max over doc tokens
        k = min(K_TOP, mx.shape[1])
        out[f"ms_mean_{r}"] = mx.mean(1)
        out[f"ms_median_{r}"] = np.median(mx, 1)
        out[f"ms_top3_{r}"] = -np.sort(-mx, 1)[:, :k].mean(1)
    return np.stack([out[c] for c in COLUMNS])
