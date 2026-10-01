"""Exp 18 helpers: regions, pooling, metrics, linear CKA (direct + streaming), query centring, groups."""

import numpy as np
import pytest
import torch

from exp18lib import repr_change as R
from exp16lib.inputs import encode_attack_and_control, encode_clean


# ---- regions ---------------------------------------------------------------------------------

def test_pair_regions_excludes_inserted_and_maps_clean_layout():
    # clean: [T T q q T | d d d | 4 tail] -> n_clean = 12; attack inserts 2 tokens at positions 6, 7
    n, A = 14, [6, 7]
    reg = R.pair_regions(n, A, (2, 4), n_passage=3)
    assert reg.query == [2, 3]
    assert reg.orig_doc == [5, 8, 9]                       # passage tokens around the insertion
    assert reg.shared == [i for i in range(n) if i not in A]
    assert not set(A) & set(reg.orig_doc) and not set(A) & set(reg.shared)


def test_pair_regions_rejects_insert_in_query():
    with pytest.raises(ValueError):
        R.pair_regions(14, [3], (2, 4), n_passage=4)


@pytest.mark.parametrize("attacked,shift", [("hello world this is a passage . relevant relevant", False),
                                             ("relevant hello world this is a passage .", False),
                                             ("hello world relevant this is a passage .", False)])
def test_regions_on_real_encoding(tokenizer, attacked, shift):
    q, p = "what is a passage", "hello world this is a passage ."
    atk, ctl, info = encode_attack_and_control(tokenizer, q, p, attacked, 512)
    reg = R.regions_from_info(info)
    clean = encode_clean(tokenizer, q, p, 512)
    R.check_pair(atk, ctl, reg, clean_ids=clean.input_ids)
    # orig_doc tokens == clean passage tokens, query tokens == clean query tokens
    d0, d1 = clean.meta["doc_span"]
    assert [atk.input_ids[i] for i in reg.orig_doc] == clean.input_ids[d0:d1]
    q0, q1 = clean.meta["query_span"]
    assert [ctl.input_ids[i] for i in reg.query] == clean.input_ids[q0:q1]
    assert reg.orig_doc == [i for i in range(reg.n) if ctl.doc_mask[i]]     # no boundary shift here
    # full visible (attack) = shared + inserted; control full visible == shared
    assert sorted(reg.shared + reg.inserted) == list(range(reg.n))
    assert reg.shared == [i for i in range(reg.n) if ctl.attention_mask[i]]


def test_collate_and_check_batch(tokenizer):
    q, p = "what is a passage", "hello world this is a passage ."
    pairs = []
    for att in ("hello world this is a passage . relevant", "relevant relevant hello world this is a passage ."):
        atk, ctl, info = encode_attack_and_control(tokenizer, q, p, att, 512)
        pairs.append((atk, ctl, R.regions_from_info(info)))
    b = R.collate_pairs(pairs, tokenizer.pad_token_id, torch.device("cpu"))
    R.check_batch(b)
    assert b["input_ids"].shape[0] == 4
    # inserted positions: visible in attack, not in any region of the aligned comparison
    ins = pairs[1][2].inserted
    assert b["attack_full_mask"][1, ins].eq(1).all() and b["region_mask"][1, ins].eq(0).all()
    bad = {k: v.clone() for k, v in b.items()}
    bad["input_ids"][2, pairs[0][2].orig_doc[0]] += 1                       # corrupt one attack token
    with pytest.raises(AssertionError):
        R.check_batch(bad)


# ---- pooling + metrics -----------------------------------------------------------------------

def test_masked_mean():
    h = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    m = torch.tensor([[[1, 0], [1, 1], [0, 1], [0, 0]]], dtype=torch.float32)
    out = R.masked_mean(h, m)
    assert torch.allclose(out[0, 0], h[0, :2].mean(0)) and torch.allclose(out[0, 1], h[0, 1:3].mean(0))
    with pytest.raises(ValueError):
        R.masked_mean(h, torch.zeros(1, 4, 1))


def test_cosine_and_normalized_l2():
    x = torch.tensor([[1.0, 0.0], [3.0, 4.0], [1.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([[0.0, 2.0], [6.0, 8.0], [-1.0, -1.0]], dtype=torch.float64)
    assert torch.allclose(R.one_minus_cosine(x, y), torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64))
    assert torch.allclose(R.normalized_l2(x, y), torch.tensor([np.sqrt(5), 1.0, 2.0], dtype=torch.float64))
    # stable for tiny angles: 1 - cos ~ theta^2 / 2
    th = 1e-5
    a = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[np.cos(th), np.sin(th)]], dtype=torch.float64)
    assert abs(float(R.one_minus_cosine(a, b)) - th ** 2 / 2) < 1e-15
    # near-zero control norm does not blow up
    assert torch.isfinite(R.normalized_l2(torch.zeros(1, 2), torch.ones(1, 2))).all()


def test_state_metrics_identical_and_masked():
    torch.manual_seed(0)
    B, S, d = 2, 7, 5
    hc = torch.randn(B, S, d)
    ha = hc.clone()
    ha[:, 3] = torch.randn(B, d) * 10                       # inserted position 3 differs
    reg = torch.zeros(B, S, 3)
    reg[:, 1, 0] = 1                                        # query
    reg[:, [4, 5], 1] = 1                                   # orig doc
    reg[:, [0, 1, 2, 4, 5, 6], 2] = 1                       # shared (excludes 3)
    full = torch.ones(B, S)
    m = R.state_metrics(torch.cat([hc, ha]), {"region_mask": reg, "attack_full_mask": full})
    assert torch.allclose(m["one_minus_cosine"][:, :3], torch.zeros(B, 3, dtype=torch.float64), atol=1e-7)
    assert (m["one_minus_cosine"][:, 3] > 0).all()          # full-visible includes the inserted token
    assert torch.allclose(m["tw_mean_omc"], torch.zeros(B, 3), atol=1e-7)
    # the control full-visible pool == shared pool
    assert torch.allclose(m["X"][:, 2], hc[:, [0, 1, 2, 4, 5, 6]].mean(1).double(), atol=1e-6)


def test_masked_quantiles():
    v = torch.tensor([[1.0, 2.0, 3.0, 100.0], [5.0, 6.0, 7.0, 8.0]])
    m = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    q = R.masked_quantiles(v, m)
    assert torch.allclose(q[1], torch.tensor([2.0, 6.5]))


# ---- linear CKA ------------------------------------------------------------------------------

def test_linear_cka_sanity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20))
    assert R.linear_cka(X, X) == pytest.approx(1.0)
    assert R.linear_cka(X, 7.3 * X) == pytest.approx(1.0)                 # isotropic scaling
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert R.linear_cka(X, X @ Q + 3.0) == pytest.approx(1.0)             # orthogonal transform + shift
    Y = rng.normal(size=(500, 20))
    assert R.linear_cka(X, Y) < 0.1                                       # unrelated
    assert R.linear_cka(X, Y) < R.linear_cka(X, X + 0.5 * Y) < 1.0


def test_linear_cka_matches_kernel_form():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(60, 8)), rng.normal(size=(60, 5)) + rng.normal(size=(60, 1))
    H = np.eye(60) - 1 / 60
    K, L = H @ X @ X.T @ H, H @ Y @ Y.T @ H
    hsic = lambda A, B: np.trace(A @ B)
    assert R.linear_cka(X, Y) == pytest.approx(hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L)))


def test_cka_from_stats_raw_and_query_centered():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(5), [10, 20, 1, 7, 12])                     # includes a singleton query
    X = rng.normal(size=(len(g), 6)) + 5 * rng.normal(size=(5, 6))[g]   # strong query offsets
    Y = X @ rng.normal(size=(6, 6)) + rng.normal(size=(len(g), 6)) + 5 * rng.normal(size=(5, 6))[g]
    G = lambda A, B: A.T @ B
    raw = R.cka_from_stats(G(X, X), G(Y, Y), G(X, Y), X.sum(0), Y.sum(0), len(g))
    assert raw == pytest.approx(R.linear_cka(X, Y))
    sx = np.stack([X[g == q].sum(0) for q in range(5)])
    sy = np.stack([Y[g == q].sum(0) for q in range(5)])
    n = np.bincount(g)
    qc = R.cka_from_stats(G(X, X), G(Y, Y), G(X, Y), sx, sy, n)
    Xc, Yc = R.query_center(X, g), R.query_center(Y, g)
    assert np.abs(Xc[g == 2]).max() == 0                                # singleton -> zero row
    num = np.linalg.norm(Yc.T @ Xc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))
    assert qc == pytest.approx(num)


def test_query_center_does_not_mix_conditions():
    X = np.array([[1.0], [3.0], [10.0], [14.0]])
    g = np.array([0, 0, 1, 1])
    assert np.allclose(R.query_center(X, g).ravel(), [-1, 1, -2, 2])
    # centring Y uses Y's own query means only
    Y = X + 100
    assert np.allclose(R.query_center(Y, g), R.query_center(X, g))


def test_accumulator_matches_direct():
    torch.manual_seed(3)
    N, d, Q = 90, 6, 4
    X = torch.randn(N, 3, d, dtype=torch.float64)
    Y = X[:, R.ATK_OF_REGION[:3]].clone()
    Y = torch.cat([Y + 0.3 * torch.randn_like(Y), torch.randn(N, 1, d, dtype=torch.float64)], 1)
    cell = torch.randint(0, 4, (N,))
    q = torch.randint(0, Q, (N,))
    acc = R.CKAAccumulator(1, Q, d, "cpu")
    for s in range(0, N, 25):                                 # several batches
        acc.update(0, X[s:s + 25], Y[s:s + 25], cell[s:s + 25], q[s:s + 25])
        acc.add_counts(cell[s:s + 25], q[s:s + 25])
    for sg, rg in [("all", "all"), ("successful", "relevant"), ("unsuccessful", "all")]:
        cells = R.cells_of(sg, rg)
        m = np.isin(cell.numpy(), cells)
        tab = {r["region"]: r for r in acc.cka_table(cells)}
        for r, reg in enumerate(R.REGIONS):
            x = X[m][:, R.CTL_OF_REGION[r]].numpy()
            y = Y[m][:, R.ATK_OF_REGION[r]].numpy()
            assert tab[reg]["cka_raw"] == pytest.approx(R.linear_cka(x, y))
            xc, yc = R.query_center(x, q.numpy()[m]), R.query_center(y, q.numpy()[m])
            ref = np.linalg.norm(yc.T @ xc) ** 2 / (np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc))
            assert tab[reg]["cka_query_centered"] == pytest.approx(ref)
            assert tab[reg]["n_instances"] == m.sum()


# ---- groups ----------------------------------------------------------------------------------

def test_groups_consistent():
    succ = np.array([1, 1, 0, 0, 1], bool)
    rel = np.array(["relevant", "nonrelevant", "relevant", "nonrelevant", "relevant"])
    assert R.group_mask(succ, rel, "successful", "all").tolist() == [1, 1, 0, 0, 1]
    assert R.group_mask(succ, rel, "unsuccessful", "relevant").tolist() == [0, 0, 1, 0, 0]
    assert R.group_mask(succ, rel, "all", "nonrelevant").tolist() == [0, 1, 0, 1, 0]
    cell = 2 * succ.astype(int) + (rel == "relevant")
    for sg in R.SUCCESS_GROUPS:
        for rg in R.RELEVANCE_GROUPS:
            assert (np.isin(cell, R.cells_of(sg, rg)) == R.group_mask(succ, rel, sg, rg)).all()
    with pytest.raises(ValueError):
        R.group_mask(succ, rel, "weird", "all")


def test_query_bootstrap_mean():
    sums = np.array([10.0, 20.0, 30.0])
    counts = np.array([10, 10, 10])
    m, lo, hi = R.query_bootstrap_mean(sums, counts, 500, 0)
    assert m == pytest.approx(2.0) and lo <= m <= hi
    m, lo, hi = R.query_bootstrap_mean(np.array([5.0, 5.0]), np.array([5, 5]), 100, 0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_boundary_shift_instance_orig_doc_is_passage(tokenizer):
    """A real DECISIONS-40 boundary-shift instance: orig_doc = exactly the passage (control doc mask has +1 ':')."""
    import json, gzip, pathlib
    import exp18lib
    mdir = exp18lib.EXP16_DIR / "outputs" / "17_paired_manifest"
    if not (mdir / "base_pairs.jsonl.gz").exists():
        pytest.skip("stage-17 manifest not present")
    base = {json.loads(l)["pair_id"]: json.loads(l) for l in gzip.open(mdir / "base_pairs.jsonl.gz", "rt")}
    rec = None
    for f in sorted((mdir / "attacks").glob("*.jsonl.gz")):
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r["alignment_boundary_shift"]:
                rec = r
                break
        if rec:
            break
    b = base[rec["pair_id"]]
    atk, ctl, info = encode_attack_and_control(tokenizer, b["query"], b["passage"], rec["attacked_passage"], 512)
    reg = R.regions_from_info(info)
    clean = encode_clean(tokenizer, b["query"], b["passage"], 512)
    R.check_pair(atk, ctl, reg, clean_ids=clean.input_ids)
    d0, d1 = clean.meta["doc_span"]
    assert [atk.input_ids[i] for i in reg.orig_doc] == clean.input_ids[d0:d1]
    ctl_doc = [i for i in range(reg.n) if ctl.doc_mask[i]]
    assert len(ctl_doc) == len(reg.orig_doc) + 1 and set(reg.orig_doc) < set(ctl_doc)
