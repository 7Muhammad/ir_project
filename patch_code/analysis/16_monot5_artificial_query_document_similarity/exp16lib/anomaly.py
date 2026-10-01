"""
exp16lib/anomaly.py
====================
Interpretable head-count anomaly detector on the 36 layer-9-11 encoder-head
query/document similarity features (stage 13). No trained model:

  normal reference : clean genuinely relevant documents (qrel 2/3) of TRAINING queries
  per head h       : mu_h, sigma_h (ddof=1) on that reference only
  test example     : z_h = (sim_h - mu_h) / sigma_h
  abnormal_h       : |z_h| > k            (primary k = 2, fixed in advance)
  anomaly score    : abnormal_head_count = sum_h abnormal_h  (0..36)
  secondary score  : mean_abs_z = mean_h |z_h|

Held-out evaluation: deterministic query-level K-fold (random.Random(seed)
shuffle of the sorted common qids; round-robin into K folds). A query never
contributes to mu/sigma of the fold in which it is tested; attack data never
contributes to mu/sigma at all.

Detector threshold T (flag if count >= T) is chosen WITHOUT test data by an
inner loop over the other folds: for inner fold g != f, mu/sigma are fitted on
the remaining training folds and both classes of fold g are scored; T maximises
balanced accuracy on those pooled inner held-out counts (ties -> smallest T).

Heads whose training sigma < SIGMA_MIN are reported and excluded from the count
for that fold (never silently divided by ~0).
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

SIGMA_MIN = 1e-8


def query_folds(qids: Sequence[str], n_folds: int, seed: int) -> Dict[str, int]:
    """qid -> fold (0..n_folds-1); deterministic, balanced sizes."""
    q = sorted(set(map(str, qids)))
    if len(q) < n_folds:
        raise ValueError(f"{len(q)} queries < {n_folds} folds")
    random.Random(seed).shuffle(q)
    return {qid: i % n_folds for i, qid in enumerate(q)}


def fit_reference(X_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu, sigma, valid) from the genuine reference rows [n, H]."""
    if X_ref.ndim != 2 or len(X_ref) < 2:
        raise ValueError("need >= 2 reference rows")
    mu = X_ref.mean(0)
    sigma = X_ref.std(0, ddof=1)
    return mu, sigma, sigma >= SIGMA_MIN


def zscores(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[n, H]; NaN for heads with sigma < SIGMA_MIN."""
    z = np.full(X.shape, np.nan)
    z[:, valid] = (X[:, valid] - mu[valid]) / sigma[valid]
    return z


def abnormal_counts(z: np.ndarray, k: float) -> Dict[str, np.ndarray]:
    """Per-row counts (NaN heads never count as abnormal)."""
    zz = np.nan_to_num(z, nan=0.0)
    low, high = zz < -k, zz > k
    return {"abnormal_head_count": (low | high).sum(1), "n_low_abnormal": low.sum(1), "n_high_abnormal": high.sum(1),
            "mean_abs_z": np.nanmean(np.abs(z), 1)}


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)  (Mann-Whitney, tie-corrected ranks)."""
    s = np.concatenate([scores_pos, scores_neg]).astype(float)
    r = rankdata(s)
    n1, n0 = len(scores_pos), len(scores_neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_metrics(count_pos: np.ndarray, count_neg: np.ndarray, T: int) -> Dict[str, float]:
    """Flag = count >= T; positives = successful attacks, negatives = genuine relevant."""
    tp = int((count_pos >= T).sum()); fn = len(count_pos) - tp
    fp = int((count_neg >= T).sum()); tn = len(count_neg) - fp
    tpr = tp / len(count_pos) if len(count_pos) else np.nan
    fpr = fp / len(count_neg) if len(count_neg) else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {"T": int(T), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr_recall": tpr, "fpr": fpr,
            "specificity": 1 - fpr, "precision": prec,
            "accuracy": (tp + tn) / (tp + tn + fp + fn), "balanced_accuracy": (tpr + 1 - fpr) / 2}


def choose_threshold(count_pos: np.ndarray, count_neg: np.ndarray, t_max: int) -> int:
    """argmax_T balanced accuracy over T = 0..t_max (ties -> smallest T)."""
    best = max(range(t_max + 1), key=lambda T: (round(threshold_metrics(count_pos, count_neg, T)["balanced_accuracy"], 12), -T))
    return int(best)


def inner_threshold(Xg: np.ndarray, fold_g: np.ndarray, Xa: np.ndarray, fold_a: np.ndarray,
                    train_folds: List[int], k: float) -> int:
    """Choose T using only the training folds (inner held-out fold scores)."""
    cp, cn = [], []
    for g in train_folds:
        fit_rows = np.isin(fold_g, [f for f in train_folds if f != g])
        mu, sd, ok = fit_reference(Xg[fit_rows])
        cn.append(abnormal_counts(zscores(Xg[fold_g == g], mu, sd, ok), k)["abnormal_head_count"])
        cp.append(abnormal_counts(zscores(Xa[fold_a == g], mu, sd, ok), k)["abnormal_head_count"])
    return choose_threshold(np.concatenate(cp), np.concatenate(cn), Xg.shape[1])


def inner_head_ranking(Xg: np.ndarray, fold_g: np.ndarray, Xa: np.ndarray, fold_a: np.ndarray,
                       train_folds: List[int], k: float) -> np.ndarray:
    """
    Head order (best first) by excess attack abnormality rate, computed ONLY from the
    training folds: for each inner fold g, the genuine reference is fitted on the other
    training folds and fold g is scored; rates are pooled over the inner held-out rows.
    Ties -> lower head index first.
    """
    za, zg = [], []
    for g in train_folds:
        mu, sd, ok = fit_reference(Xg[np.isin(fold_g, [f for f in train_folds if f != g])])
        zg.append(zscores(Xg[fold_g == g], mu, sd, ok))
        za.append(zscores(Xa[fold_a == g], mu, sd, ok))
    ab = lambda z: (np.abs(np.nan_to_num(np.concatenate(z), nan=0.0)) > k).mean(0)  # noqa: E731
    excess = ab(za) - ab(zg)
    return np.lexsort((np.arange(len(excess)), -excess))


# ---- generalisation for the paired extension (stages 17-21) ---------------------------------
# Above, the genuine reference and the negative class are the same rows (Xg). In the paired
# design the negatives are each positive's own padded control, which for qrel-0 documents is
# NOT the (qrel 2/3) reference. These helpers take reference, positives and negatives
# separately; with negatives == reference they reproduce inner_head_ranking / inner_threshold
# exactly (tests/test_paired.py).

def inner_heldout_z(X_ref: np.ndarray, fold_ref: np.ndarray, Xs: List[np.ndarray], folds: List[np.ndarray],
                    train_folds: List[int]) -> List[np.ndarray]:
    """Inner CV over the TRAINING folds only: for every g in train_folds the reference is refitted on
    the other training folds and the fold-g rows of each X in Xs are z-scored; concatenated over g."""
    out = [[] for _ in Xs]
    for g in train_folds:
        mu, sd, ok = fit_reference(X_ref[np.isin(fold_ref, [f for f in train_folds if f != g])])
        for i, (X, fo) in enumerate(zip(Xs, folds)):
            out[i].append(zscores(X[fo == g], mu, sd, ok))
    return [np.concatenate(o) for o in out]


def head_ranking_from_z(z_pos: np.ndarray, z_neg: np.ndarray, k: float) -> np.ndarray:
    """Head order by excess abnormality rate (pos - neg), ties -> lower head index (as inner_head_ranking)."""
    ab = lambda z: (np.abs(np.nan_to_num(z, nan=0.0)) > k).mean(0)  # noqa: E731
    excess = ab(z_pos) - ab(z_neg)
    return np.lexsort((np.arange(len(excess)), -excess))


def inner_head_ranking_ref(X_ref, fold_ref, X_pos, fold_pos, X_neg, fold_neg, train_folds: List[int],
                           k: float) -> np.ndarray:
    zp, zn = inner_heldout_z(X_ref, fold_ref, [X_pos, X_neg], [fold_pos, fold_neg], train_folds)
    return head_ranking_from_z(zp, zn, k)


def inner_threshold_ref(X_ref, fold_ref, X_pos, fold_pos, X_neg, fold_neg, train_folds: List[int], k: float,
                        heads=None) -> int:
    """T chosen on inner held-out training folds, optionally counting only the given head indices."""
    zp, zn = inner_heldout_z(X_ref, fold_ref, [X_pos, X_neg], [fold_pos, fold_neg], train_folds)
    if heads is not None:
        zp, zn = zp[:, heads], zn[:, heads]
    return choose_threshold(abnormal_counts(zp, k)["abnormal_head_count"],
                            abnormal_counts(zn, k)["abnormal_head_count"], zp.shape[1])
