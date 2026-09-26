"""
exp10lib/geometry.py
=====================
Rank-structure (PCA/SVD) and cross-attack direction-consistency utilities.

Mean-centering decision (see DECISIONS.md for full rationale)
---------------------------------------------------------------
SVD is computed on the RAW (uncentered) per-attack diff matrix, not on the
matrix after subtracting its own mean. The question this experiment asks —
"is there a single direction you could add to the encoder output as a
steering-based defense" — is a question about the diff vectors THEMSELVES,
not about how they vary around their mean. Mean-centering first would
answer a different question (is the *residual after removing the mean* low
rank), and could actively hide a strong, consistent shared direction (if
every diff vector points almost the same way, centering removes exactly
that signal before SVD ever sees it). The per-attack mean diff vector is
still computed and reported (``mean_direction``) — it is itself the
Experiment-2-style candidate steering vector — and used only to fix the
sign ambiguity of the top singular vector.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def compute_pca_summary(diffs: np.ndarray, k_values: Sequence[int] = (1, 3, 10)) -> Dict:
    """
    SVD-based rank-structure summary for one (attack, layer) diff matrix.

    Parameters
    ----------
    diffs : (n_positions, d_model) float array. Rows are individual
        per-position diff vectors, pooled across all examples for one
        attack (see engine.py / scripts/02_extract_diffs.py for how rows
        from variable-length examples are concatenated without padding or
        truncation bias — DECISIONS.md).

    Returns
    -------
    dict with n_positions, variance_explained_top{k} for each k in
    k_values, effective_rank (participation ratio), top_direction (unit
    vector, sign-aligned to the mean diff vector), mean_direction (raw,
    NOT unit-normalised — its own norm is meaningful).
    """
    n_positions = diffs.shape[0]
    if n_positions == 0:
        nan_row = {f"variance_explained_top{k}": float("nan") for k in k_values}
        return {
            "n_positions": 0, "effective_rank": float("nan"),
            "top_direction": None, "mean_direction": None, **nan_row,
        }

    # full_matrices=False: economy SVD, singular values sorted descending.
    _, singular_values, vt = np.linalg.svd(diffs, full_matrices=False)
    s2 = singular_values ** 2
    total_var = float(s2.sum())

    result: Dict = {"n_positions": n_positions}
    for k in k_values:
        k_eff = min(k, len(singular_values))
        result[f"variance_explained_top{k}"] = (
            float(s2[:k_eff].sum() / total_var) if total_var > 0 else float("nan")
        )

    sum_s2_sq = float((s2 ** 2).sum())
    result["effective_rank"] = float((total_var ** 2) / sum_s2_sq) if sum_s2_sq > 0 else float("nan")

    mean_direction = diffs.mean(axis=0)
    top_direction = vt[0].copy()
    # Sign-ambiguity fix: PCA/SVD directions have arbitrary sign; align to
    # have positive dot product with this attack's own mean diff vector
    # (the natural, task-specified convention — see DECISIONS.md).
    if np.dot(top_direction, mean_direction) < 0:
        top_direction = -top_direction

    result["top_direction"] = top_direction
    result["mean_direction"] = mean_direction
    return result


def cosine_similarity_matrix(directions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pairwise cosine similarity between already sign-aligned unit (or
    non-unit, will be normalised here) direction vectors.

    Parameters
    ----------
    directions : {attack_name: vector}. Order of the dict determines the
        row/column order of the returned DataFrame — callers should pass
        an already-ordered dict (e.g. by attack strength) for a readable
        heatmap.

    Returns
    -------
    Square DataFrame, index/columns = attack names, symmetric, diagonal 1.0.
    """
    names = list(directions.keys())
    vecs = np.stack([directions[n] for n in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    sim = unit @ unit.T
    return pd.DataFrame(sim, index=names, columns=names)
