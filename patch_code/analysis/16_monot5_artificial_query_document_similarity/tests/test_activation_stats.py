"""Stage 24 activation statistics: metric definitions, edge cases, masking, pooled representation."""

from __future__ import annotations

import numpy as np
import pytest

from exp16lib import activation_stats as AS

S = {s: i for i, s in enumerate(AS.STATS)}


def naive(X):
    """Reference per-head loop implementation of the documented formulas (X [n, d])."""
    X = X.astype(np.float64)
    n = len(X)
    e = (X ** 2).mean(0)
    E = (X ** 2).sum(1)
    p = E / E.sum()
    ent = np.nan if n == 1 else -sum(v * np.log(v) for v in p if v > 0) / np.log(n)
    return {"l2": np.mean([np.linalg.norm(x) for x in X]), "mean": X.mean(), "var": np.mean([x.var() for x in X]),
            "maxabs": np.abs(X).max(), "eff_dim": e.sum() ** 2 / (e ** 2).sum(), "top_share": p.max(), "entropy": ent}


def test_matches_naive_per_head():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5, 64)).astype(np.float16)
    out = AS.region_stats(X)
    assert out.shape == (len(AS.STATS), 5)
    for h in range(5):
        ref = naive(X[:, h])
        for s, i in S.items():
            assert out[i, h] == pytest.approx(ref[s], rel=1e-10, abs=1e-12), (s, h)


def test_effective_dim():
    one_hot = np.zeros((4, 1, 64)); one_hot[:, 0, 3] = 2.0
    assert AS.region_stats(one_hot)[S["eff_dim"]][0] == pytest.approx(1.0)
    iso = np.ones((3, 1, 64))
    assert AS.region_stats(iso)[S["eff_dim"]][0] == pytest.approx(64.0)
    k8 = np.zeros((5, 1, 64)); k8[:, 0, :8] = np.random.default_rng(1).choice([-1, 1], size=(5, 8))
    assert AS.region_stats(k8)[S["eff_dim"]][0] == pytest.approx(8.0)
    rnd = np.random.default_rng(2).normal(size=(50, 3, 64))
    ed = AS.region_stats(rnd)[S["eff_dim"]]
    assert np.all((ed >= 1) & (ed <= 64))
    # scale invariance of the ratio statistics
    r1, r2 = AS.region_stats(rnd), AS.region_stats(10 * rnd)
    for s in ("eff_dim", "top_share", "entropy"):
        np.testing.assert_allclose(r1[S[s]], r2[S[s]], rtol=1e-12)


def test_token_concentration():
    uni = np.ones((6, 1, 64))                                  # equal token energies
    r = AS.region_stats(uni)
    assert r[S["top_share"]][0] == pytest.approx(1 / 6) and r[S["entropy"]][0] == pytest.approx(1.0)
    spike = np.zeros((6, 1, 64)); spike[2, 0, :] = 1.0          # all energy on one token
    r = AS.region_stats(spike)
    assert r[S["top_share"]][0] == pytest.approx(1.0) and r[S["entropy"]][0] == pytest.approx(0.0)
    two = np.zeros((4, 1, 64)); two[[0, 3], 0, 0] = 1.0         # two of four tokens: H = log 2 / log 4
    r = AS.region_stats(two)
    assert r[S["top_share"]][0] == pytest.approx(0.5) and r[S["entropy"]][0] == pytest.approx(0.5)


def test_zero_and_one_token_regions():
    zero = AS.region_stats(np.zeros((0, 3, 64)))
    assert zero.shape == (len(AS.STATS), 3) and np.isnan(zero).all()
    one = AS.region_stats(np.random.default_rng(3).normal(size=(1, 3, 64)))
    assert np.all(one[S["top_share"]] == 1.0)
    assert np.isnan(one[S["entropy"]]).all()
    assert not np.isnan(np.delete(one, S["entropy"], 0)).any()


def test_zero_energy_region():
    r = AS.region_stats(np.zeros((4, 2, 64)))
    for s in ("l2", "mean", "var", "maxabs"):
        assert np.all(r[S[s]] == 0)
    for s in ("eff_dim", "top_share", "entropy"):
        assert np.isnan(r[S[s]]).all()


def _toy_sequence():
    # [template q q q template d d I I d template pad pad]
    qm = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    dm = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], bool)
    im = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], bool)
    H = np.arange(13, dtype=np.float64)[:, None, None] * np.ones((13, 2, 64)) + 1
    return H, qm, dm, im


def test_masking_attack_regions():
    H, qm, dm, im = _toy_sequence()
    out = AS.sequence_stats(H, qm, dm, im, "attack")
    assert (out["n_query"], out["n_doc"], out["n_orig"], out["n_ins"]) == (3, 5, 3, 2)
    # l2 per token = value * 8 (sqrt 64); region mean of token values
    assert out["query"][S["l2"], 0] == pytest.approx(8 * np.mean([2, 3, 4]))
    assert out["orig"][S["l2"], 0] == pytest.approx(8 * np.mean([6, 7, 10]))
    assert out["ins"][S["l2"], 0] == pytest.approx(8 * np.mean([8, 9]))
    assert out["doc"][S["l2"], 0] == pytest.approx(8 * np.mean([6, 7, 8, 9, 10]))


def test_masking_control_excludes_padded_slots():
    H, qm, dm, im = _toy_sequence()
    # control: insertion slots are masked pads OUTSIDE the document pool
    dm_c = dm & ~im
    out = AS.sequence_stats(H, qm, dm_c, im, "control")
    assert out["ins"] is None and out["n_ins"] == 0
    assert out["n_doc"] == out["n_orig"] == 3
    np.testing.assert_array_equal(out["doc"], out["orig"])
    assert out["doc"][S["l2"], 0] == pytest.approx(8 * np.mean([6, 7, 10]))
    # a control whose insertion slots leak into the document pool is rejected
    with pytest.raises(ValueError):
        AS.sequence_stats(H, qm, dm, im, "control")


def test_pooled_late_doc_is_2304_and_layer_major():
    rng = np.random.default_rng(4)
    heads = rng.normal(size=(9, 12, 12, 64)).astype(np.float16)
    m = np.zeros(9, bool); m[[2, 5, 6]] = True
    v = AS.pooled_late_doc(heads, m)
    assert v.shape == (2304,)
    ref = heads[m].astype(np.float32).mean(0)
    np.testing.assert_allclose(v[:64], ref[9, 0]); np.testing.assert_allclose(v[64:128], ref[9, 1])
    np.testing.assert_allclose(v[768:832], ref[10, 0]); np.testing.assert_allclose(v[-64:], ref[11, 11])


def test_size_matched_and_effects():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 2, 64))
    r = AS.size_matched_stats(X, 1, 4, np.random.default_rng(0))
    assert np.all(r[S["top_share"]] == 1.0) and np.isnan(r[S["entropy"]]).all()
    assert np.isnan(AS.size_matched_stats(X, 30, 4, np.random.default_rng(0))).all()
    full = AS.size_matched_stats(X, 20, 3, np.random.default_rng(0))       # k = n: every draw is the full region
    np.testing.assert_allclose(full, AS.region_stats(X))
    assert AS.cohen_d(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 0
    assert AS.cohen_d(np.array([3.0, 4, 5]), np.array([1.0, 2, 3])) > 0
    assert AS.paired_dz(np.array([1.0, 2, 3])) == pytest.approx(2.0)
    assert np.isnan(AS.paired_dz(np.array([np.nan, 1.0])))
