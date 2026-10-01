"""Unit tests for exp17lib.attention_stats (synthetic attention; CPU, no model)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from exp17lib import attention_stats as AT  # noqa: E402


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def _stats(A, key_mask, n_tokens=None):
    A = torch.tensor(A, dtype=torch.float32)
    if A.ndim == 2:
        A = A[None, None]
    km = torch.tensor(key_mask, dtype=torch.bool)[None]
    nt = torch.tensor([n_tokens or A.shape[-1]])
    return {k: v[0, 0].numpy() for k, v in AT.row_stats(A, km, nt).items()}


def test_uniform_gives_max_entropy():
    S = 10
    km = np.ones(S, bool)
    km[[2, 7]] = False                                    # 8 allowed keys
    A = np.full((S, S), 1 / S)
    st = _stats(A, km)
    np.testing.assert_allclose(st["entropy_norm"], 1.0, atol=1e-5)
    np.testing.assert_allclose(st["neff_norm"], 1.0, atol=1e-5)
    np.testing.assert_allclose(st["neff"], 8.0, atol=1e-4)
    np.testing.assert_allclose(st["max_attn"], 1 / 8, atol=1e-6)
    np.testing.assert_allclose(st["top3"], 3 / 8, atol=1e-6)


def test_one_hot_gives_zero_entropy():
    S = 9
    A = np.zeros((S, S))
    A[np.arange(S), (np.arange(S) + 3) % S] = 1.0
    st = _stats(A, np.ones(S, bool))
    np.testing.assert_allclose(st["entropy_norm"], 0.0, atol=1e-6)
    np.testing.assert_allclose(st["max_attn"], 1.0)
    np.testing.assert_allclose(st["top3"], 1.0)
    np.testing.assert_allclose(st["neff"], 1.0, atol=1e-6)
    np.testing.assert_allclose(st["dist"], np.abs(np.arange(S) - (np.arange(S) + 3) % S))


def test_single_key_entropy_nan_and_topk_capped():
    S = 6
    km = np.zeros(S, bool)
    km[4] = True
    st = _stats(_softmax(np.random.default_rng(0).normal(size=(S, S))), km)
    assert np.isnan(st["entropy_norm"]).all()
    np.testing.assert_allclose(st["entropy"], 0.0, atol=1e-6)
    np.testing.assert_allclose(st["top5"], 1.0, atol=1e-6)
    np.testing.assert_allclose(st["neff_norm"], 1.0, atol=1e-6)


def test_no_keys_all_nan():
    S = 5
    st = _stats(np.full((S, S), 0.2), np.zeros(S, bool))
    for m in AT.METRICS:
        assert np.isnan(st[m]).all(), m


def test_torch_matches_numpy_reference():
    rng = np.random.default_rng(1)
    S, n_tok = 23, 20                                     # 3 trailing batch-padding positions
    for trial in range(10):
        km = rng.random(S) > 0.3
        km[n_tok:] = False
        A = _softmax(rng.normal(scale=2.0, size=(S, S)) + np.where(km, 0, -1e9)[None, :])
        st = _stats(A, km, n_tok)
        for i in range(n_tok):
            ref = AT.row_stats_np(A[i], km, i, n_tok)
            for m in AT.METRICS:
                np.testing.assert_allclose(st[m][i], ref[m], rtol=1e-4, atol=1e-5, err_msg=f"{m} row {i}")


def test_topk_ordering_and_ranges():
    rng = np.random.default_rng(2)
    A = _softmax(rng.normal(scale=3.0, size=(4, 3, 30, 30)))
    km = torch.ones(4, 30, dtype=torch.bool)
    km[0, 5:] = False                                     # 5 keys
    km[1, 2:] = False                                     # 2 keys (< k)
    st = AT.row_stats(torch.tensor(A, dtype=torch.float32), km, torch.tensor([30] * 4))
    assert (st["max_attn"] <= st["top3"] + 1e-6).all() and (st["top3"] <= st["top5"] + 1e-6).all()
    assert (st["top5"] <= 1 + 1e-5).all()
    en = st["entropy_norm"]
    assert (en >= -1e-5).all() and (en <= 1 + 1e-5).all()
    np.testing.assert_allclose(st["top3"][1].numpy(), 1.0, atol=1e-5)      # k_eff = min(k, N) = 2
    nn = st["neff_norm"]
    assert (nn <= 1 + 1e-5).all() and (st["neff"] >= 1 - 1e-5).all()


def test_local_mass_and_dist_norm():
    S = 12
    A = np.zeros((S, S))
    A[:, 0] = 0.5
    A[:, S - 1] = 0.5
    st = _stats(A, np.ones(S, bool))
    i = 3
    np.testing.assert_allclose(st["dist"][i], 0.5 * 3 + 0.5 * 8, atol=1e-6)
    np.testing.assert_allclose(st["dist_norm"][i], (0.5 * 3 + 0.5 * 8) / 11, atol=1e-6)
    np.testing.assert_allclose(st["local_mass"][i], 0.5, atol=1e-6)       # |3-0| <= 5, |3-11| > 5


def test_shared_scope_renormalises_without_inserted_keys():
    S = 8
    A = np.zeros((S, S))
    A[:, 2] = 0.6                                         # 0.6 on an inserted key
    A[:, 5] = 0.3
    A[:, 6] = 0.1
    am = np.ones(S, int)
    ins = np.zeros(S, int)
    ins[2] = 1
    doc = np.ones(S, int)
    m = AT.batch_masks(am[None], np.zeros((1, S), int), doc[None], ins[None])
    st = AT.row_stats(torch.tensor(A, dtype=torch.float32)[None, None], m["shared_key"], torch.tensor([S]))
    np.testing.assert_allclose(st["max_attn"][0, 0, 0].item(), 0.75, atol=1e-6)


def test_region_mass_uniform_is_one_and_raw_correct():
    S = 16
    vis = np.ones(S, bool)
    vis[[3, 9]] = False                                   # masked (control slots / padding)
    A = np.where(vis[None, :], 1.0 / vis.sum(), 0.0) * np.ones((S, 1))
    src = np.zeros(S, bool); src[:4] = True
    tgt = np.zeros(S, bool); tgt[10:14] = True
    raw, norm = AT.region_mass(torch.tensor(A, dtype=torch.float32)[None, None], torch.tensor(vis)[None],
                               torch.tensor(src)[None], torch.tensor(tgt)[None])
    np.testing.assert_allclose(norm.item(), 1.0, atol=1e-5)
    np.testing.assert_allclose(raw.item(), 4 / 14, atol=1e-6)
    r2, n2 = AT.region_mass_np(A, src, tgt, vis)
    np.testing.assert_allclose([raw.item(), norm.item()], [r2, n2], atol=1e-6)


def test_region_mass_empty_target_nan():
    S = 6
    A = torch.full((1, 1, S, S), 1 / S)
    raw, norm = AT.region_mass(A, torch.ones(1, S, dtype=torch.bool), torch.ones(1, S, dtype=torch.bool),
                               torch.zeros(1, S, dtype=torch.bool))
    assert torch.isnan(raw).all() and torch.isnan(norm).all()


def test_region_mean_empty_and_nan_rows():
    stat = torch.tensor([[[1.0, 3.0, float("nan"), 5.0]]])
    np.testing.assert_allclose(AT.region_mean(stat, torch.tensor([[True, True, True, False]])).item(), 2.0)
    assert torch.isnan(AT.region_mean(stat, torch.zeros(1, 4, dtype=torch.bool))).all()


def test_batch_masks_control_slots_never_visible_or_source():
    # [tmpl, q, q, tmpl, d, SLOT, d, tmpl]; control slot = pad, attention 0, inserted 1, outside doc_mask
    am = np.array([[1, 1, 1, 1, 1, 0, 1, 1]])
    q = np.array([[0, 1, 1, 0, 0, 0, 0, 0]])
    d = np.array([[0, 0, 0, 0, 1, 0, 1, 0]])
    ins = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])
    m = AT.batch_masks(am, q, d, ins)
    assert not m["full_visible"][0, 5] and not m["shared_key"][0, 5]
    assert torch.equal(m["full_visible"], m["shared_key"])
    assert not any(m[r][0, 5] for r in AT.REGIONS)
    assert m["ins"].sum() == 0 and torch.equal(m["orig"], m["doc"])
    assert m["full_visible"][0, 0] and m["full_visible"][0, 7]            # template tokens stay keys
    assert not any(m[r][0, 0] or m[r][0, 7] for r in AT.REGIONS)          # but are no source region


def test_batch_statistics_shapes_and_scope_equality_without_insertion():
    rng = np.random.default_rng(3)
    B, H, S, L = 2, 3, 10, 2
    atts = [torch.tensor(_softmax(rng.normal(size=(B, H, S, S))), dtype=torch.float32) for _ in range(L)]
    am = np.ones((B, S), int)
    q = np.zeros((B, S), int); q[:, 1:3] = 1
    d = np.zeros((B, S), int); d[:, 4:9] = 1
    ins = np.zeros((B, S), int); ins[1, 6:8] = 1                         # sequence 1 = attacked
    m = AT.batch_masks(am, q, d, ins)
    R, M = AT.batch_statistics(atts, m, torch.tensor([S, S]))
    assert R.shape == (B, len(AT.REGIONS), len(AT.SCOPES), len(AT.METRICS), L, H)
    assert M.shape == (B, len(AT.MASS_DIRS), 2, L, H)
    np.testing.assert_array_equal(R[0, :, 0], R[0, :, 1])                # no insertion -> scopes identical
    assert np.isnan(R[0, AT.REGIONS.index("ins")]).all() and not np.isnan(R[1, AT.REGIONS.index("ins")]).any()
    assert not np.allclose(R[1, 0, 0], R[1, 0, 1])                       # attacked -> scopes differ
    assert np.isnan(M[0, AT.MASS_NAMES.index("query_to_ins")]).all()


def test_residualize_removes_linear_trend():
    x = np.arange(20, dtype=float)
    y = 2 * x + 1 + np.sin(x)
    r = AT.residualize(y, x)
    assert abs(np.corrcoef(r, x)[0, 1]) < 1e-8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
