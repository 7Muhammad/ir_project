"""Tests 13-16, 20, 22: delta_sim, delta_step, no success filter, equal attack / query weighting, attack-level Spearman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import EXP1_ATTACKS, load_script
from exp16lib.aggregate import add_steps, two_level_mean
from exp16lib.checkpoints import checkpoint_table
from exp16lib.tables import long_table

A4 = load_script("04_analyze")
CFG = {"statistics": {"n_sign_flips": 500, "sign_flip_seed": 42, "fdr_alpha": 0.05}}


def _attack_df(spec):
    """spec: {attack_name: [(delta_score, sim_attack[25], sim_control[25]), ...]}"""
    meta, sa, sc = [], [], []
    for k, (name, exs) in enumerate(spec.items()):
        for j, (ds, a, c) in enumerate(exs):
            meta.append({"attack_id": k, "attack_name": name, "attack_token": name.split("_")[0],
                         "attack_position": "start", "repetitions": 1, "pair_id": f"{k}_{j}",
                         "delta_score": ds})
            sa.append(a)
            sc.append(c)
    sa, sc = np.array(sa), np.array(sc)
    return long_table(meta, {"sim_attack": sa, "sim_control": sc, "delta_sim": sa - sc})


def test_13_delta_sim_definition():
    a = np.linspace(0, 1, 25)
    c = np.linspace(0, 0.5, 25)
    df = _attack_df({"x_start_1": [(1.0, a, c)]})
    assert np.allclose(df.delta_sim, df.sim_attack - df.sim_control)
    per, glob, *_ = A4.analyze_attacks(df, CFG)
    assert np.allclose(glob.global_delta_sim, a - c)


def test_14_delta_step_definition():
    rng = np.random.default_rng(0)
    a, c = rng.normal(size=25), rng.normal(size=25)
    df = _attack_df({"x_start_1": [(1.0, a, c)], "y_start_1": [(0.0, a * 2, c)]})
    _, glob, acc, *_ = A4.analyze_attacks(df, CFG)
    d = glob.global_delta_sim.values
    assert np.isnan(acc.delta_step.iloc[0])
    assert np.allclose(acc.delta_step.values[1:], d[1:] - d[:-1])
    assert np.allclose(acc.delta_step.values[1:], acc.step_attack.values[1:] - acc.step_control.values[1:])
    assert acc.transition.iloc[1] == "embedding -> L00_post_attn"


def test_15_no_success_filter():
    z = np.zeros(25)
    df = _attack_df({"x_start_1": [(2.0, z + 0.1, z), (0.0, z, z), (-3.0, z - 0.1, z)]})
    per, glob, acc, tests, assoc, brk = A4.analyze_attacks(df, CFG)
    assert (per.n_examples == 3).all()
    assert np.isclose(per.mean_delta_score.iloc[0], (2.0 + 0.0 - 3.0) / 3)
    assert np.allclose(glob.global_delta_sim, 0.0)


def test_15b_stage00_keeps_every_scored_example(tokenizer):
    """bar_end_1 has only 153 success-filtered selected examples; the manifest must keep all 500."""
    S0 = load_script("00_prepare_manifests")
    cfg = {"_config_dir": str(EXP1_ATTACKS), "_attack_grid_source": "test", "model": {"max_length": 512},
           "attacks": {"upstream_injected_dir": "/home/ghoummaid/IR/ecir24-adversarial-evaluation/runs/injected/dl19",
                       "mode": "include", "include": ["bar_end_1"]},
           "data": {"exp1_attacks_dir": str(EXP1_ATTACKS), "max_examples_per_attack": None,
                    "canonical_pairs": str(EXP1_ATTACKS.parent / "pairs" / "pairs.jsonl"),
                    "canonical_scores": str(EXP1_ATTACKS.parent / "scores" / "all_scores.csv")}}
    prov = {}
    _, canonical = S0.build_clean(cfg, tokenizer, prov)
    recs = S0.build_attacks(cfg, tokenizer, canonical, prov)
    ds = np.array([r["delta_score"] for r in recs])
    assert len(recs) == 500
    assert (ds > 0).sum() == 153 and (ds < 0).sum() > 0


def test_16_equal_attack_weighting():
    one = np.ones(25)
    # attack A: 1 example with delta 1; attack B: 3 examples with delta 0
    df = _attack_df({"a_start_1": [(0.0, one, 0 * one)],
                     "b_start_1": [(0.0, 0 * one, 0 * one)] * 3})
    _, glob, *_ = A4.analyze_attacks(df, CFG)
    assert np.allclose(glob.global_delta_sim, 0.5)          # equal attack weight
    assert not np.allclose(glob.global_delta_sim, 0.25)     # pooled example mean would be 0.25


def test_20_equal_query_weighting():
    ck = pd.DataFrame(checkpoint_table())
    rows = []
    # q1: 1 rel (0.9) / 1 non (0.1)  -> delta 0.8;  q2: 3 rel (0.5) / 3 non (0.5) -> delta 0
    for qid, n, rs, ns in (("q1", 1, 0.9, 0.1), ("q2", 3, 0.5, 0.5)):
        for j in range(n):
            for grp, s in (("relevant", rs), ("nonrelevant", ns)):
                for _, c in ck.iterrows():
                    rows.append({"qid": qid, "pair_id": f"{qid}_{grp}_{j}", "relevance_group": grp,
                                 "similarity": s, "monot5_score": 0.0, **c.to_dict()})
    pq, gen, _ = A4.analyze_qrel(pd.DataFrame(rows))
    assert np.allclose(gen.delta_sim_real, 0.4)             # (0.8 + 0) / 2, not document-weighted 0.2
    assert np.allclose(gen.sim_rel, (0.9 + 0.5) / 2)
    assert np.allclose(pq[pq.qid == "q2"].n_rel, 3)
    assert np.isnan(gen.delta_step_real.iloc[0]) and np.allclose(gen.delta_step_real.iloc[1:], 0)


def test_22_delta_sim_delta_score_uses_attack_means():
    rng = np.random.default_rng(1)
    spec = {}
    for k in range(6):
        exs = []
        for j in range(4):
            d = k * 0.1 + rng.normal(scale=0.01)
            exs.append((float(k) + rng.normal(scale=5.0), np.full(25, d), np.zeros(25)))
        spec[f"t{k}_start_1"] = exs
    df = _attack_df(spec)
    per, _, _, _, assoc, _ = A4.analyze_attacks(df, CFG)
    m = per[per.checkpoint_index == 0]
    from scipy.stats import spearmanr
    assert np.isclose(assoc.rho_attack_level.iloc[0], spearmanr(m.mean_delta_sim, m.mean_delta_score).statistic)
    assert assoc.n_attacks.iloc[0] == 6 and assoc.n_examples_secondary.iloc[0] == 24


def test_two_level_mean_and_steps_helpers():
    df = pd.DataFrame(checkpoint_table())
    d = pd.concat([df.assign(u="a", v=np.arange(25.0)), df.assign(u="b", v=2 * np.arange(25.0))])
    s = add_steps(d, ["v"], ["u"])
    assert np.allclose(s[s.u == "b"].step_v.values[1:], 2.0)
    per, glob = two_level_mean(d, "u", ["v"])
    assert np.allclose(glob.v, 1.5 * np.arange(25)) and (glob.n_units == 2).all()
    with pytest.raises(ValueError):
        long_table([{"x": 1}], {"v": np.zeros((1, 24))})
