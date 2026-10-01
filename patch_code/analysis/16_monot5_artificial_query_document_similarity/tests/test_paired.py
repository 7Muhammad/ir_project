"""Paired extension (stages 17-21): design invariants, leakage guards, pairing, resume, old-output protection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import EXP_DIR, load_script
from exp16lib import paired as P
from exp16lib.anomaly import (inner_head_ranking, inner_head_ranking_ref, inner_heldout_z, inner_threshold,
                              inner_threshold_ref, query_folds)


@pytest.fixture(autouse=True)
def _default_scope():
    """validate_cfg(configure) sets module-level scope state; every test starts and ends on the default."""
    P.configure(P.DEFAULT_SCOPE)
    yield
    P.configure(P.DEFAULT_SCOPE)


# ---- design constants ----------------------------------------------------------------------

def test_exactly_36_heads_L9_L11_in_order():
    labels = P.head_labels()
    assert len(labels) == 36 == P.N_HEADS
    assert labels[0] == "L9H0" and labels[11] == "L9H11" and labels[12] == "L10H0" and labels[-1] == "L11H11"
    assert P.head_cols("control")[0] == "control_head_L9H0" and len(P.head_cols("attack")) == 36
    assert len(P.ckpt_cols("control")) == 25


def test_config_locks():
    from exp16lib.config import load_config
    for name in ("default.yaml", "paired_smoke.yaml", "paired_all_layers.yaml", "paired_all_layers_smoke.yaml"):
        cfg = load_config(EXP_DIR / "configs" / name)
        P.validate_cfg(cfg)
    cfg = load_config(EXP_DIR / "configs" / "default.yaml")
    for key, bad in (("abnormal_z", 2.5), ("fold_seed", 7), ("n_folds", 10), ("success_threshold", 0.1), ("layers", [11])):
        c = json.loads(json.dumps(cfg))
        c["paired"][key] = bad
        with pytest.raises(ValueError):
            P.validate_cfg(c)
    assert len(cfg["attacks"]["include"]) == 105                    # one shared grid (Exp 01 multi_attack.yaml)


def test_fold_map_deterministic_seed42():
    q = [str(1000 + i) for i in range(43)]
    assert query_folds(q, 5, 42) == query_folds(list(reversed(q)), 5, 42)
    assert sorted(np.bincount(list(query_folds(q, 5, 42).values())).tolist()) == [8, 8, 9, 9, 9]


# ---- synthetic paired frame ----------------------------------------------------------------------

def _frame(n_q=10, n_docs=4, attacks=("a1", "a2", "a3"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for qi in range(n_q):
        for d in range(n_docs):
            grade = [3, 2, 0, 0][d % 4]
            for ai, a in enumerate(attacks):
                sc, sa = rng.normal(), rng.normal()
                rows.append({"attack_id": ai, "attack_name": a, "attack_token": "true", "attack_position": "start",
                             "repetitions": 1, "pair_id": f"q{qi}_d{d}", "qid": f"q{qi}", "docid": f"d{d}",
                             "qrel_grade": grade, "relevance_group": P.RELEVANT if grade else P.NONRELEVANT,
                             "score_control": sc, "score_attack": sa, "delta_score": sa - sc, "successful": sa - sc > 0})
    df = pd.DataFrame(rows)
    H = rng.normal(0.3, 0.05, (len(df), 36))
    df = pd.concat([df, pd.DataFrame(H, columns=P.head_cols("control")),
                    pd.DataFrame(H + rng.normal(0, 0.05, H.shape), columns=P.head_cols("attack"))], axis=1)
    return df


def test_check_forward_frame_rules():
    df = _frame()
    P.check_forward_frame(df)
    bad = df.copy(); bad.loc[0, "qrel_grade"] = 1
    with pytest.raises(RuntimeError, match="qrel 1"):
        P.check_forward_frame(bad)
    bad = df.copy(); bad.loc[0, "delta_score"] += 1e-3
    with pytest.raises(RuntimeError, match="delta_score"):
        P.check_forward_frame(bad)
    bad = df.copy(); i = bad.index[bad.delta_score > 0][0]
    bad.loc[i, "successful"] = False                                     # success must be strictly delta > 0
    with pytest.raises(RuntimeError, match="successful"):
        P.check_forward_frame(bad)
    bad = df.copy(); bad.loc[bad.index[0], ["score_attack", "score_control", "delta_score", "successful"]] = [1.0, 1.0, 0.0, True]
    with pytest.raises(RuntimeError, match="successful"):                # delta == 0 is NOT a success
        P.check_forward_frame(bad)
    with pytest.raises(RuntimeError, match="36"):
        P.check_forward_frame(df.drop(columns=["attack_head_L11H11"]))
    with pytest.raises(RuntimeError, match="duplicate"):
        P.check_forward_frame(pd.concat([df, df.iloc[:1]], ignore_index=True))


def test_reference_is_successful_relevant_controls_only():
    df = _frame()
    ref = P.reference_mask(df)
    assert ref.any()
    assert df.qrel_grade[ref].isin([2, 3]).all() and (df.delta_score[ref] > 0).all()
    assert not (ref & (df.relevance_group == P.NONRELEVANT).values).any()
    assert not (ref & (df.delta_score <= 0).values).any()
    assert df.attack_name[ref].nunique() == 3                          # pooled over attack configurations


def test_one_reference_shared_by_all_attacks_and_heldout_never_used():
    df = _frame()
    fold_of = query_folds(df.qid.unique(), 5, 42)
    fold = df.qid.map(fold_of).values
    ref = P.reference_mask(df)
    Xc = df[P.head_cols("control")].values
    z, fits = P.outer_zscores(Xc, fold, Xc[ref], fold[ref])
    for f in range(5):                                                 # mu = mean over ALL attacks' train controls
        assert np.allclose(fits[f][0], Xc[ref & (fold != f)].mean(0))
    # corrupting fold-0 reference rows never changes fold-0 z-scores (held-out queries never in mu/sigma)
    X2 = Xc.copy(); X2[ref & (fold == 0)] += 100.0
    z2, _ = P.outer_zscores(Xc, fold, X2[ref], fold[ref])
    assert np.allclose(z[fold == 0], z2[fold == 0])
    assert not np.allclose(z[fold == 1], z2[fold == 1])
    # relabelling attacks cannot change anything: the fit has no attack dimension
    z3, _ = P.outer_zscores(Xc, fold, Xc[ref], fold[ref])
    assert np.array_equal(z, z3)


def test_abnormal_rule_strict_and_transitions():
    z_c = np.array([[0.0, 2.0, -2.5, 3.0, np.nan]])
    z_a = np.array([[2.01, 1.0, -1.0, -3.0, 5.0]])
    sc, sa = P.states(z_c), P.states(z_a)
    assert sc.tolist() == [[0, 0, -1, 1, 0]]                          # |z| = 2 is normal; NaN = normal
    assert sa.tolist() == [[1, 0, 0, -1, 1]]
    t = P.transition_counts(sc, sa)
    assert t["n_normal_to_abnormal"].tolist() == [2] and t["n_normal_to_abnormal_high"].tolist() == [2]
    assert t["n_abnormal_to_normal"].tolist() == [1] and t["n_abnormal_low_to_normal"].tolist() == [1]
    assert t["n_abnormal_to_abnormal"].tolist() == [1] and t["n_normal_to_normal"].tolist() == [1]
    assert sum(t[k][0] for k in ("n_normal_to_normal", "n_normal_to_abnormal", "n_abnormal_to_normal",
                                 "n_abnormal_to_abnormal")) == 5


# ---- generalised inner CV == old rule, and never sees the test fold ------------------------------------

def _xy(seed=0):
    rng = np.random.default_rng(seed)
    Xr = rng.normal(0, 1, (100, 6)); fr = np.repeat(np.arange(5), 20)
    Xp = rng.normal(0, 1, (80, 6)); fp = np.repeat(np.arange(5), 16)
    Xp[:, 3] -= 4; Xp[:, 1] += 2.5
    Xn = rng.normal(0.3, 1.2, (80, 6))
    return Xr, fr, Xp, fp, Xn, fp.copy()


def test_generalised_inner_functions_reduce_to_old():
    Xr, fr, Xp, fp, _, _ = _xy()
    tr = [1, 2, 3, 4]
    assert inner_head_ranking_ref(Xr, fr, Xp, fp, Xr, fr, tr, 2.0).tolist() == inner_head_ranking(Xr, fr, Xp, fp, tr, 2.0).tolist()
    assert inner_threshold_ref(Xr, fr, Xp, fp, Xr, fr, tr, 2.0) == inner_threshold(Xr, fr, Xp, fp, tr, 2.0)


def test_heldout_fold_never_affects_ranking_threshold_or_k():
    Xr, fr, Xp, fp, Xn, fn = _xy(1)
    tr = [1, 2, 3, 4]
    zp, zn = inner_heldout_z(Xr, fr, [Xp, Xn], [fp, fn], tr)
    o1, T1, a1 = P.topk_training_selection(zp, zn, 2.0, range(1, 7))
    for arr, fo in ((Xr, fr), (Xp, fp), (Xn, fn)):
        arr[fo == 0] = 1e6 * np.sign(np.random.default_rng(3).normal(size=arr[fo == 0].shape))
    zp2, zn2 = inner_heldout_z(Xr, fr, [Xp, Xn], [fp, fn], tr)
    o2, T2, a2 = P.topk_training_selection(zp2, zn2, 2.0, range(1, 7))
    assert o1.tolist() == o2.tolist() and T1 == T2 and a1 == a2
    assert o1[:2].tolist() == [3, 1]                                     # planted heads found
    assert inner_threshold_ref(Xr, fr, Xp, fp, Xn, fn, tr, 2.0, heads=o1[:2]) == T1[2]


def test_detector_groups_are_separate_and_negative_is_own_control():
    det = load_script("20_paired_detector")
    df = _frame(n_q=15, seed=4)
    fold = df.qid.map(query_folds(df.qid.unique(), 5, 42)).values
    s = df[df.successful].reset_index(drop=True); fold = fold[df.successful.values]
    Xc, Xa = s[P.head_cols("control")].values, s[P.head_cols("attack")].values
    ref = P.reference_mask(s)
    Zc, _ = P.outer_zscores(Xc, fold, Xc[ref], fold[ref]); Za, _ = P.outer_zscores(Xa, fold, Xc[ref], fold[ref])
    m = (s.relevance_group == P.RELEVANT).values
    rows, fr, sel, ks = det.evaluate_group(Xc, Xa, Zc, Za, fold, ref, m, P.head_labels())
    assert len(rows) == 36 and (rows.n_pairs == m.sum()).all()
    assert (rows.tp + rows.fn == m.sum()).all() and (rows.fp + rows.tn == m.sum()).all()   # 1 negative per positive
    # corrupt NON-relevant attacked rows: relevant results unchanged (reference never uses attacked rows)
    Xa2, Za2 = Xa.copy(), Za.copy(); Xa2[~m] += 50; Za2[~m] += 50
    rows2, *_ = det.evaluate_group(Xc, Xa2, Zc, Za2, fold, ref, m, P.head_labels())
    pd.testing.assert_frame_equal(rows, rows2)


# ---- forward pairing on a tiny T5 -----------------------------------------------------------------------

def _seq(ids, am, qm, dm):
    from exp16lib.inputs import EncodedSeq
    return EncodedSeq(ids, am, qm, dm)


def test_forward_with_heads_matches_existing_captures(tiny_model):
    from exp16lib.engine import run_sequences
    from exp13lib.head_lists import SenderHead
    from exp16lib.heads import run_encoder_head_sequences
    hs = [SenderHead(layer=L, head_idx=h, label=f"L{L}H{h}") for L in (1, 2) for h in range(4)]
    seqs = [_seq([5, 6, 7, 8, 9, 10, 1], [1] * 7, [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0]),
            _seq([5, 6, 7, 0, 9, 1], [1, 1, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]),
            _seq([5, 6, 7, 8, 1], [1] * 5, [0, 1, 0, 0, 0], [0, 0, 1, 1, 0])]
    cpu = torch.device("cpu")
    C, H, S = P.run_paired_sequences(tiny_model, seqs, 0, cpu, 2, hs, 3, 4)
    C0, S0 = run_sequences(tiny_model, seqs, 0, cpu, 2, with_score=True, true_id=3, false_id=4)
    H0 = run_encoder_head_sequences(tiny_model, seqs, 0, cpu, 2, hs)
    assert C.shape == C0.shape and np.allclose(C, C0, atol=1e-6) and np.allclose(S, S0, atol=1e-5)
    assert np.allclose(H, H0, atol=1e-6)                                  # input order preserved (pairing)


def test_encode_instance_logs_alignment_failure_and_pairs_control(tokenizer):
    q, p = "what is a wifi router", "A router forwards packets between networks."
    atk, ctl, info = P.encode_instance(tokenizer, q, p, "relevant relevant " + p, 512)
    assert atk is not None and atk.seq_len == ctl.seq_len and info["n_inserted"] > 0
    assert all(ctl.input_ids[i] == tokenizer.pad_token_id and ctl.attention_mask[i] == 0 for i in info["inserted_positions"])
    a, c, reason = P.encode_instance(tokenizer, q, p, p, 512)             # nothing inserted -> alignment failure
    assert a is None and c is None and "alignment failed" in reason
    from exp16lib.inputs import EncodingError
    with pytest.raises(EncodingError):                                   # other errors are raised, not logged
        P.encode_instance(tokenizer, q, p, "relevant " + p, 5)


# ---- resume + old outputs --------------------------------------------------------------------------------

def test_resume_per_attack_unit(tmp_path):
    from exp16lib.run_utils import unit_is_done, write_status
    u = tmp_path / "18_paired_forward" / "per_attack" / "relevant_start_5"
    u.mkdir(parents=True)
    assert not unit_is_done(u, ["rows.csv.gz"], "b:a", False)
    (u / "rows.csv.gz").write_text("x")
    write_status(u, {"status": "success", "manifest_sha256": "b:a"})
    assert unit_is_done(u, ["rows.csv.gz"], "b:a", False)
    assert not unit_is_done(u, ["rows.csv.gz"], "b:a", True)             # --force reruns
    with pytest.raises(RuntimeError):
        unit_is_done(u, ["rows.csv.gz"], "b:CHANGED", False)            # manifest changed -> refuse to mix
    write_status(u, {"status": "running", "manifest_sha256": "b:a"})
    assert not unit_is_done(u, ["rows.csv.gz"], "b:a", False)            # interrupted unit is rerun


def test_old_output_fingerprint(tmp_path):
    (tmp_path / "13_late_encoder_heads").mkdir()
    (tmp_path / "13_late_encoder_heads" / "x.csv").write_text("a")
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "fig.png").write_text("p")
    fp = P.fingerprint_old_outputs(tmp_path)
    assert set(fp) == {"13_late_encoder_heads/x.csv", "plots/fig.png"}
    for d in P.PAIRED_OUTPUTS:                                           # paired outputs are excluded
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / "new.csv").write_text("n")
    assert P.fingerprint_old_outputs(tmp_path) == fp
    (tmp_path / "13_late_encoder_heads" / "x.csv").write_text("b")
    assert P.fingerprint_old_outputs(tmp_path) != fp


def test_cluster_se_and_doc_level():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    assert np.isclose(P.cluster_se(x, np.arange(4)), x.std(ddof=1) / 2)   # singleton clusters = classic SE
    inst = pd.DataFrame({"relevance_group": [P.RELEVANT] * 3, "qid": ["q"] * 3, "docid": ["d"] * 3, "pair_id": ["q_d"] * 3,
                         "qrel_grade": [2] * 3, "attack_name": ["a", "b", "c"], "control_abnormal_count": [0, 1, 2],
                         "attack_abnormal_count": [3, 1, 2], "abnormal_count_change": [3, 0, 0], "delta_score": [1, 2, 3]})
    d = P.doc_level(inst).iloc[0]
    assert d.n_successful_attacks == 3 and d.mean_paired_change == 1 and d.median_paired_change == 0


# ---- all-layers scope extension ------------------------------------------------------------------------

def test_all_layers_scope_is_144_heads_with_own_dirs():
    from exp16lib.config import load_config
    p = P.validate_cfg(load_config(EXP_DIR / "configs" / "paired_all_layers.yaml"))
    assert P.SCOPE == "all_layers" and P.LAYERS == list(range(12)) and P.N_HEADS == 144
    labels = P.head_labels()
    assert len(labels) == 144 == len(set(labels)) and labels[0] == "L0H0" and labels[12] == "L1H0" and labels[-1] == "L11H11"
    assert labels[108:] == [f"L{L}H{h}" for L in (9, 10, 11) for h in range(12)]      # L9-L11 block = primary scope
    assert P.FORWARD_DIR == "18_paired_forward_all_layers" and P.PLOT_DIR == "plots/paired_all_layers"
    assert P.MANIFEST_DIR == "17_paired_manifest"                                     # manifest shared
    # everything except the scope is identical to the primary configuration
    d = load_config(EXP_DIR / "configs" / "default.yaml")["paired"]
    assert {k: v for k, v in p.items() if k not in ("scope", "layers")} == {k: v for k, v in d.items() if k not in ("scope", "layers")}
    P.configure("L9_L11")
    assert P.N_HEADS == 36 and P.FORWARD_DIR == "18_paired_forward"
    assert {"18_paired_forward", "18_paired_forward_all_layers", "plots/paired", "plots/paired_all_layers"} <= set(P.PAIRED_OUTPUTS)


def test_scope_layers_locked():
    from exp16lib.config import load_config
    cfg = load_config(EXP_DIR / "configs" / "paired_all_layers.yaml")
    cfg["paired"]["layers"] = [9, 10, 11]
    with pytest.raises(ValueError):
        P.validate_cfg(cfg)
    cfg["paired"]["scope"] = "L0_L3"
    with pytest.raises(ValueError):
        P.validate_cfg(cfg)


def test_all_layers_frame_and_detector_generalise():
    base = _frame(n_q=10, seed=5)                                    # built under the default scope
    df = base.drop(columns=[c for c in base.columns if "_head_" in c])
    P.configure("all_layers")
    det = load_script("20_paired_detector")
    rng = np.random.default_rng(5)
    H = rng.normal(0.3, 0.05, (len(df), 144))
    df = pd.concat([df, pd.DataFrame(H, columns=P.head_cols("control")),
                    pd.DataFrame(H + rng.normal(0, 0.05, H.shape), columns=P.head_cols("attack"))], axis=1)
    P.check_forward_frame(df)
    with pytest.raises(RuntimeError, match="144"):
        P.check_forward_frame(df.drop(columns=["control_head_L0H0"]))
    fold = df.qid.map(query_folds(df.qid.unique(), 5, 42)).values
    s = df[df.successful].reset_index(drop=True); fold = fold[df.successful.values]
    Xc, Xa = s[P.head_cols("control")].values, s[P.head_cols("attack")].values
    ref = P.reference_mask(s)
    Zc, _ = P.outer_zscores(Xc, fold, Xc[ref], fold[ref]); Za, _ = P.outer_zscores(Xa, fold, Xc[ref], fold[ref])
    rows, *_ = det.evaluate_group(Xc, Xa, Zc, Za, fold, ref, (s.relevance_group == P.RELEVANT).values, P.head_labels())
    assert rows.k_heads.tolist() == list(range(1, 145))


# ---- stage 22 token sample --------------------------------------------------------------------------------

def test_token_sample_design_and_loader(tmp_path):
    from exp16lib import token_sample as TS
    from exp16lib.config import load_config
    grid = load_config(EXP_DIR / "configs" / "default.yaml")["attacks"]["include"]
    A = TS.DEFAULT_ATTACKS
    assert len(A) == len(set(A)) == 12 and set(A) <= set(grid)
    parts = [a.rsplit("_", 2) for a in A]
    assert {p[1] for p in parts} == {"start", "end", "random"} and {p[2] for p in parts} == {"1", "5"}
    assert len({p[0] for p in parts}) == 7                             # every token represented
    base = [{"qid": q, "docid": f"d{i}", "relevance_group": g, "pair_id": f"{q}_d{i}"}
            for q in ("1", "2") for i, g in enumerate(["relevant"] * 5 + ["nonrelevant"] * 2)]
    s1, s2 = TS.sample_pairs(base, 3, 42), TS.sample_pairs(list(reversed(base)), 3, 42)
    assert s1 == s2 and len(s1) == 2 * (3 + 2)                         # deterministic; capped per group
    # loader round trip
    T = [3, 5]
    heads = (np.arange(sum(T) * 12 * 12 * 64) % 7).astype(np.float16).reshape(sum(T), 12, 12, 64)
    np.save(tmp_path / "heads.npy", heads)
    np.savez(tmp_path / "tokens.npz", input_ids=np.arange(8), attention_mask=np.ones(8), query_mask=np.array([0, 1, 0, 0, 1, 1, 0, 0]),
             doc_mask=np.array([0, 0, 1, 0, 0, 0, 1, 1]), inserted_mask=np.zeros(8))
    pd.DataFrame({"seq_id": [0, 1], "token_offset": [0, 3], "n_tokens": T, "kind": ["clean", "control"],
                  "qid": ["1", "1"], "docid": ["a", "a"], "pair_id": ["1_a"] * 2, "attack_name": ["", "x"],
                  "attack_token": ["", "true"]}).to_csv(tmp_path / "sequences.csv", index=False)
    ts = TS.TokenSample(tmp_path)
    s = ts[1]
    assert s["heads"].shape == (5, 12, 12, 64) and np.array_equal(s["heads"], heads[3:8])
    assert s["query_mask"].tolist() == [False, True, True, False, False] and s["meta"]["kind"] == "control"
    assert len(list(ts.iter(kind="control"))) == 1
    h = s["heads"][:, 4, 5].astype(np.float64)
    q, d = h[s["query_mask"]].mean(0), h[s["doc_mask"]].mean(0)
    assert np.isclose(TS.TokenSample.pooled_cosine(s, 4, 5), q @ d / np.linalg.norm(q) / np.linalg.norm(d))


# ---- stage 23 metric screen ---------------------------------------------------------------------------------

def test_metric_screen_definitions():
    from exp16lib import metrics_screen as MS
    rng = np.random.default_rng(0)
    T, nh, d = 12, 3, 4
    H = rng.normal(size=(T, nh, d)).astype(np.float32)
    qm = np.zeros(T, bool); qm[1:3] = True                                   # 2 query tokens (< top-k 3)
    dm = np.zeros(T, bool); dm[4:11] = True
    ins = np.zeros(T, bool); ins[8:10] = True
    reg = MS.region_masks(qm, dm, ins, "attack")
    assert reg["full"].sum() == 7 and reg["ins"].tolist() == ins.tolist() and reg["orig"].sum() == 5
    with pytest.raises(ValueError):                                          # control pools may not contain inserted slots
        MS.region_masks(qm, dm, ins, "control")
    assert MS.region_masks(qm, dm & ~ins, ins, "control")["ins"] is None
    out = dict(zip(MS.COLUMNS, MS.sequence_metrics(H, qm, reg, np.zeros((nh, d), np.float32))))
    for r in ("full", "orig", "ins"):
        m = reg[r]
        for h in range(nh):
            q, dd = H[qm, h].mean(0), H[m, h].mean(0)
            assert np.isclose(out[f"cos_{r}"][h], q @ dd / np.linalg.norm(q) / np.linalg.norm(dd))
            assert np.isclose(out[f"ccos_{r}"][h], out[f"cos_{r}"][h])            # mu = 0 -> centered == plain
            assert np.isclose(out[f"dot_{r}"][h], q @ dd, rtol=1e-5)
            Qn = H[qm, h] / np.linalg.norm(H[qm, h], axis=1, keepdims=True)
            Dn = H[m, h] / np.linalg.norm(H[m, h], axis=1, keepdims=True)
            mx = (Qn @ Dn.T).max(1)
            assert np.isclose(out[f"ms_mean_{r}"][h], mx.mean()) and np.isclose(out[f"ms_median_{r}"][h], np.median(mx))
            assert np.isclose(out[f"ms_top3_{r}"][h], mx.mean())                 # only 2 query tokens -> k = 2
    mu = rng.normal(size=(nh, d)).astype(np.float32)
    out2 = dict(zip(MS.COLUMNS, MS.sequence_metrics(H, qm, reg, mu)))
    q, dd = H[qm, 0].mean(0) - mu[0], H[reg["full"], 0].mean(0) - mu[0]
    assert np.isclose(out2["ccos_full"][0], q @ dd / np.linalg.norm(q) / np.linalg.norm(dd), rtol=1e-5)
    assert np.allclose(out2["cos_full"], out["cos_full"])                   # mu only affects ccos
