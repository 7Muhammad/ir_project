"""Head-count anomaly detector: folds, reference fit, z/abnormal counts, AUROC, thresholds."""

from __future__ import annotations

import numpy as np

from exp16lib.anomaly import (SIGMA_MIN, abnormal_counts, auroc, choose_threshold, fit_reference, inner_threshold,
                              query_folds, threshold_metrics, zscores)


def test_query_folds_deterministic_balanced_disjoint():
    q = [f"q{i}" for i in range(42)]
    f1, f2 = query_folds(q, 5, 42), query_folds(list(reversed(q)), 5, 42)
    assert f1 == f2                                              # seed 42, order-independent
    sizes = np.bincount(list(f1.values()))
    assert sizes.tolist() in ([9, 9, 8, 8, 8],) and set(f1) == set(q)
    assert query_folds(q, 5, 7) != f1


def test_reference_zscores_and_counts():
    X = np.array([[0.0, 1.0, 5.0], [2.0, 1.0, 7.0], [4.0, 1.0, 9.0]])
    mu, sd, ok = fit_reference(X)
    assert np.allclose(mu, [2, 1, 7]) and np.allclose(sd[[0, 2]], 2) and not ok[1]   # constant head -> invalid
    z = zscores(np.array([[8.0, 1.0, 7.0], [-3.0, 50.0, 2.0]]), mu, sd, ok)
    assert np.isnan(z[:, 1]).all() and np.allclose(z[:, [0, 2]], [[3.0, 0.0], [-2.5, -2.5]])
    c = abnormal_counts(z, 2.0)
    assert c["abnormal_head_count"].tolist() == [1, 2]            # NaN head never counts
    assert c["n_high_abnormal"].tolist() == [1, 0] and c["n_low_abnormal"].tolist() == [0, 2]
    assert np.allclose(c["mean_abs_z"], [1.5, 2.5])
    assert abnormal_counts(z, 3.0)["abnormal_head_count"].tolist() == [0, 0]   # strict |z| > k
    assert SIGMA_MIN > 0


def test_auroc_and_threshold_metrics():
    assert auroc(np.array([3, 4]), np.array([1, 2])) == 1.0
    assert auroc(np.array([1, 2]), np.array([3, 4])) == 0.0
    assert auroc(np.array([2, 2]), np.array([2, 2])) == 0.5       # ties count half
    m = threshold_metrics(np.array([0, 3, 5]), np.array([0, 0, 1, 4]), 3)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
    assert np.isclose(m["tpr_recall"], 2 / 3) and np.isclose(m["fpr"], 0.25) and np.isclose(m["precision"], 2 / 3)
    assert threshold_metrics(np.array([1]), np.array([0]), 0)["tpr_recall"] == 1.0   # T=0 flags everything
    assert choose_threshold(np.array([5, 6, 7]), np.array([0, 1, 2]), 36) == 3        # smallest T separating


def test_inner_threshold_never_uses_test_fold():
    rng = np.random.default_rng(0)
    Xg = rng.normal(0, 1, (60, 5)); fg = np.repeat(np.arange(5), 12)
    Xa = rng.normal(3, 1, (50, 5)); fa = np.repeat(np.arange(5), 10)
    T1 = inner_threshold(Xg, fg, Xa, fa, [1, 2, 3, 4], 2.0)
    Xa2, Xg2 = Xa.copy(), Xg.copy()
    Xa2[fa == 0] = 1e6; Xg2[fg == 0] = -1e6                          # corrupt the held-out fold 0
    assert inner_threshold(Xg2, fg, Xa2, fa, [1, 2, 3, 4], 2.0) == T1


def test_inner_head_ranking_uses_training_folds_only():
    from exp16lib.anomaly import inner_head_ranking
    rng = np.random.default_rng(1)
    Xg = rng.normal(0, 1, (100, 4)); fg = np.repeat(np.arange(5), 20)
    Xa = rng.normal(0, 1, (100, 4)); fa = np.repeat(np.arange(5), 20)
    Xa[:, 2] -= 4.0; Xa[:, 0] -= 2.0                              # head 2 most abnormal, then head 0
    order = inner_head_ranking(Xg, fg, Xa, fa, [1, 2, 3, 4], 2.0)
    assert order[:2].tolist() == [2, 0]
    Xg2, Xa2 = Xg.copy(), Xa.copy()
    Xa2[fa == 0, 1] -= 100.0; Xg2[fg == 0] += 50.0                # corrupt test fold 0 only
    assert inner_head_ranking(Xg2, fg, Xa2, fa, [1, 2, 3, 4], 2.0).tolist() == order.tolist()
