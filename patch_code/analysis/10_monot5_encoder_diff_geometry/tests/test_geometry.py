"""PCA/SVD rank-structure and cross-attack cosine similarity, on synthetic
diff matrices with known rank structure (real hidden states aren't needed
to test the linear-algebra logic itself)."""

from __future__ import annotations

import numpy as np

from exp10lib.geometry import compute_pca_summary, cosine_similarity_matrix
from conftest import make_low_rank_diffs


def test_rank1_signal_has_high_top1_variance_and_low_effective_rank():
    diffs = make_low_rank_diffs(n=200, d=64, direction_seed=0, noise_scale=0.01, n_directions=1)
    summary = compute_pca_summary(diffs, k_values=(1, 3, 10))
    assert summary["variance_explained_top1"] > 0.95
    assert summary["effective_rank"] < 2.0


def test_isotropic_noise_has_low_top1_variance_and_high_effective_rank():
    rng = np.random.default_rng(0)
    diffs = rng.normal(size=(200, 64))
    summary = compute_pca_summary(diffs, k_values=(1, 3, 10))
    assert summary["variance_explained_top1"] < 0.2
    assert summary["effective_rank"] > 10.0


def test_variance_explained_is_monotonic_and_bounded():
    diffs = make_low_rank_diffs(n=100, d=32, direction_seed=1, n_directions=3)
    summary = compute_pca_summary(diffs, k_values=(1, 3, 10))
    assert summary["variance_explained_top1"] <= summary["variance_explained_top3"] <= summary["variance_explained_top10"]
    assert summary["variance_explained_top10"] <= 1.0 + 1e-9


def test_top_direction_sign_aligned_to_mean():
    diffs = make_low_rank_diffs(n=100, d=32, direction_seed=2, n_directions=1)
    summary = compute_pca_summary(diffs, k_values=(1,))
    assert np.dot(summary["top_direction"], summary["mean_direction"]) >= 0


def test_empty_matrix_returns_nan_not_crash():
    diffs = np.zeros((0, 32))
    summary = compute_pca_summary(diffs, k_values=(1, 3, 10))
    assert summary["n_positions"] == 0
    assert np.isnan(summary["effective_rank"])


def test_cosine_similarity_matrix_identical_directions():
    v = np.array([1.0, 0.0, 0.0])
    mat = cosine_similarity_matrix({"a": v, "b": v, "c": -v})
    assert np.isclose(mat.loc["a", "b"], 1.0)
    assert np.isclose(mat.loc["a", "c"], -1.0)
    assert np.allclose(np.diag(mat.values), 1.0)


def test_cosine_similarity_matrix_orthogonal_directions():
    mat = cosine_similarity_matrix({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert np.isclose(mat.loc["a", "b"], 0.0)
