"""Tests 21, 23, 24: Spearman, sign-flip test, BH-FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import load_script
from exp16lib.checkpoints import checkpoint_table
from exp16lib.stats import benjamini_hochberg, sign_flip_test, spearman

A4 = load_script("04_analyze")


def test_21_clean_spearman_monotonic():
    x = np.arange(20.0)
    assert np.isclose(spearman(x, np.exp(x / 5))["rho"], 1.0)
    assert np.isclose(spearman(x, -x ** 3)["rho"], -1.0)
    ck = pd.DataFrame(checkpoint_table())
    rows = []
    for i in range(10):
        for _, c in ck.iterrows():
            sim = i * 0.1 if c.checkpoint_index > 12 else (i * 7919 % 10) * 0.1
            rows.append({"pair_id": str(i), "similarity": sim, "score_clean": float(i), **c.to_dict()})
    cc = A4.analyze_clean(pd.DataFrame(rows), 0.05)
    assert len(cc) == 25 and np.allclose(cc.spearman_rho.iloc[13:], 1.0)
    assert {"spearman_p", "pearson_r", "spearman_q_bh_secondary"} <= set(cc.columns)


def test_23_sign_flip_deterministic_and_behaviour():
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.3, 105)
    r1 = sign_flip_test(pos, 5000, seed=42, stream=3)
    r2 = sign_flip_test(pos, 5000, seed=42, stream=3)
    assert r1 == r2
    assert r1["p_one_sided"] == 1 / 5001                      # clear positive -> minimum p
    null = rng.normal(0.0, 1.0, 105)
    assert sign_flip_test(null, 5000, 42)["p_one_sided"] > 0.01
    neg = -pos
    assert sign_flip_test(neg, 5000, 42)["p_one_sided"] > 0.99  # one-sided: negative is not significant
    assert np.isclose(r1["observed"], pos.mean())


def test_24_bh_known_vector():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    out = benjamini_hochberg(p, 0.05)
    # sorted: .005 .01 .03 .04 .5 -> *m/rank: .025 .025 .05 .05 .5 -> monotone from the right
    assert np.allclose(out["q"], [0.025, 0.05, 0.05, 0.025, 0.5])
    assert out["reject"] == [True, True, True, True, False]
    assert benjamini_hochberg([0.9, 0.95], 0.05)["q"] == [0.95, 0.95]
