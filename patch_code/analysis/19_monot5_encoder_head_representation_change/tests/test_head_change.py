"""Exp 19 helpers: head slicing (== EncoderHeadCapture), per-head metrics, CKA accumulator, causal comparisons."""

import numpy as np
import pytest
import torch

from exp16lib.heads import EncoderHeadCapture, all_encoder_heads
from exp16lib.pooling import pooled_cosine
from exp18lib import repr_change as R
from exp19lib import head_change as HC


def _batch(B=3, S=9, n_heads=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    inner = torch.randn(2 * B, S, n_heads * d, generator=g)
    reg = torch.zeros(B, S, 3)
    reg[:, 1:3, 0] = 1                   # query
    reg[:, [4, 6, 7], 1] = 1             # orig doc (position 5 = "inserted", excluded)
    reg[:, [0, 1, 2, 3, 4, 6, 7, 8], 2] = 1
    return inner, reg


def test_split_heads_matches_encoderheadcapture_slices():
    inner, _ = _batch()
    h = HC.split_heads(inner, 4)
    for k in range(4):
        assert torch.equal(h[:, :, k], inner[..., k * 8:(k + 1) * 8])


def test_head_metrics_identical_inputs_zero():
    inner, reg = _batch()
    B = reg.shape[0]
    inner[B:] = inner[:B]
    m = HC.head_metrics(inner, reg, 4)
    for k in HC.METRICS:
        assert torch.allclose(m[k], torch.zeros_like(m[k]), atol=1e-6), k


def test_head_metrics_values_and_inserted_excluded():
    inner, reg = _batch()
    B = reg.shape[0]
    m = HC.head_metrics(inner, reg, 4)
    h = HC.split_heads(inner, 4)
    b, k = 1, 2
    xq = h[b, 1:3, k].mean(0).double()
    yq = h[B + b, 1:3, k].mean(0).double()
    cos = float(xq @ yq / xq.norm() / yq.norm())
    assert float(m["one_minus_cosine"][b, 0, k]) == pytest.approx(1 - cos, abs=1e-6)
    assert float(m["normalized_l2"][b, 0, k]) == pytest.approx(float((yq - xq).norm() / xq.norm()), rel=1e-6)
    # changing only the inserted position (5) in the attack changes nothing in query / orig_doc
    inner2 = inner.clone()
    inner2[B:, 5] += 100
    m2 = HC.head_metrics(inner2, reg, 4)
    for key in HC.METRICS:
        assert torch.allclose(m[key], m2[key])
    # token-wise mean over the doc region
    t = [1 - torch.nn.functional.cosine_similarity(h[b, p, k], h[B + b, p, k], dim=0) for p in (4, 6, 7)]
    assert float(m["tw_mean_omc"][b, 1, k]) == pytest.approx(float(sum(t) / 3), abs=1e-6)


def test_capture_matches_exp16_encoderheadcapture(tiny_model):
    enc = tiny_model.encoder
    torch.manual_seed(1)
    B, S = 2, 7
    ids = torch.randint(2, 100, (2 * B, S))
    att = torch.ones(2 * B, S, dtype=torch.long)
    reg = torch.zeros(B, S, 3)
    reg[:, 1:3, 0] = 1
    reg[:, 3:6, 1] = 1
    reg[:, :, 2] = 1
    got = {}
    heads = all_encoder_heads(range(3))
    heads = [h for h in heads if h.head_idx < 4]
    with HC.HeadCapture(enc, lambda L, m: got.__setitem__(L, m), keep_inputs=True) as cap, \
            EncoderHeadCapture(enc, heads, keep_inputs=True) as ec, torch.inference_mode():
        cap.set_batch(reg)
        qm = torch.cat([reg[..., 0]] * 2).long()
        dm = torch.cat([reg[..., 1]] * 2).long()
        ec.set_masks(qm, dm)
        enc(input_ids=ids, attention_mask=att)
    for L in range(3):
        assert torch.equal(cap.inputs[L], ec.inputs[L])
        for h in range(4):
            x = HC.split_heads(cap.inputs[L], 4)[:, :, h]
            assert torch.allclose(pooled_cosine(x, qm, dm), ec.cos[f"L{L}H{h}"])


def test_accumulator_matches_direct_cka():
    torch.manual_seed(2)
    N, H, d, Q = 80, 3, 5, 4
    X = torch.randn(N, 2, H, d, dtype=torch.float64)
    Y = X + 0.4 * torch.randn_like(X)
    cell = torch.randint(0, 4, (N,))
    q = torch.randint(0, Q, (N,))
    acc = HC.HeadCKAAccumulator(1, H, d, Q, "cpu")
    for s in range(0, N, 30):
        acc.update(0, X[s:s + 30], Y[s:s + 30], cell[s:s + 30], q[s:s + 30])
        acc.add_counts(cell[s:s + 30], q[s:s + 30])
    st = acc.state()
    for sg, rg in [("all", "all"), ("successful", "nonrelevant")]:
        cells = R.cells_of(sg, rg)
        mk = np.isin(cell.numpy(), cells)
        tab = {(r["head"], r["region"]): r for r in HC.cka_table_from_state(st, cells)}
        for h in range(H):
            for ri, reg in enumerate(HC.REGIONS):
                x, y = X[mk][:, ri, h].numpy(), Y[mk][:, ri, h].numpy()
                assert tab[(h, reg)]["cka_raw"] == pytest.approx(R.linear_cka(x, y))
                xc, yc = R.query_center(x, q.numpy()[mk]), R.query_center(y, q.numpy()[mk])
                ref = np.linalg.norm(yc.T @ xc) ** 2 / (np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc))
                assert tab[(h, reg)]["cka_query_centered"] == pytest.approx(ref)
    # identical control / attack -> CKA 1
    acc2 = HC.HeadCKAAccumulator(1, H, d, Q, "cpu")
    acc2.update(0, X, X, cell, q)
    acc2.add_counts(cell, q)
    assert all(abs(r["cka_raw"] - 1) < 1e-12 for r in HC.cka_table_from_state(acc2.state(), [0, 1, 2, 3]))


def test_topk_enrichment_and_within_layer():
    v = np.arange(20, dtype=float)
    causal = np.zeros(20, bool)
    causal[[19, 18, 0]] = True
    e = {r["k"]: r for r in HC.topk_enrichment(v, causal, ks=(2, 5))}
    assert e[2]["n_causal_in_topk"] == 2 and e[2]["expected"] == pytest.approx(2 * 3 / 20)
    assert e[5]["p_hypergeom_ge"] == pytest.approx((3 * 680 + 136) / 15504)   # exact P(X >= 2), N=20 K=3 k=5
    layers = np.repeat([0, 1], 10)
    pct = HC.within_layer_percentile(v, layers)
    assert pct[0] == 0 and pct[9] == 1 and pct[10] == 0 and pct[19] == 1
    obs, p, null = HC.layer_stratified_permutation(v, layers, causal, n_perm=2000)
    assert obs == pytest.approx((1 + 8 / 9 + 0) / 3) and null == pytest.approx(0.5, abs=0.03)
