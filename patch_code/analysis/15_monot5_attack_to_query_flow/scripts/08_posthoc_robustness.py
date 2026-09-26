#!/usr/bin/env python3
"""
scripts/08_posthoc_robustness.py
=================================
POST-HOC, DESCRIPTIVE ONLY (added after seeing the full results — see
DECISIONS.md item 42). The pre-registered headline metric, aggregation and
tests (stages 05/06) are unchanged; nothing here is tested or used for
claims of significance.

Motivation: in the successful-instance population the attack effect δ is
often tiny (median ≈ 0.07 logits; ~10% of examples below 0.01), so
per-example ratios raw/δ — and therefore equal-weight means of them — can be
dominated by a few near-zero-δ examples. These alternative summaries show
whether the sign/ranking of heads survives without that sensitivity:

  ratio_of_means_combined  per attack: min(mean raw_fwd, mean raw_rev) / mean δ,
                           then equal-weight mean over attacks
  median_example_e_combined  pooled median of per-example e_combined
  mean_e_combined_delta_ge_0.1  equal-weight mean restricted to δ ≥ 0.1
                           (attacks with no such example are skipped)
  frac_attacks_raw_min_gt0  fraction of attacks with min(mean raw_fwd, mean raw_rev) > 0
  (+ the same first two for the whole-head reference)

Output: outputs/08_posthoc/robustness_by_head.csv, delta_distribution.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exp15lib.config import load_config, output_dir  # noqa: E402
from exp15lib.heads import load_heads  # noqa: E402
from exp15lib.run_utils import cheap_stage_is_current, now, upstream_fingerprint, write_status  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 15 stage 08: post-hoc descriptive robustness")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def ratio_of_means(df, rf, rr):
    pa = df.groupby("attack_name").agg(rf=(rf, "mean"), rr=(rr, "mean"), de=("delta", "mean"))
    return float(np.mean(np.minimum(pa.rf / pa.de, pa.rr / pa.de))), float((np.minimum(pa.rf, pa.rr) > 0).mean())


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    agg, stage = out / "05_aggregate", out / "08_posthoc"
    fp = upstream_fingerprint([agg / "status.json"])
    if cheap_stage_is_current(stage, fp, ["robustness_by_head.csv"], args.force):
        print("[08] up to date; skipping")
        return
    stage.mkdir(parents=True, exist_ok=True)
    aq = pd.read_csv(agg / "raw_all_query_edges.csv", dtype={"qid": str, "docid": str})
    wh = pd.read_csv(agg / "raw_whole_head.csv", dtype={"qid": str, "docid": str})
    ex = aq.drop_duplicates(["attack_name", "example_id"])
    qs = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    with open(stage / "delta_distribution.json", "w") as fh:
        json.dump({"quantiles": {str(q): float(ex.delta.quantile(q)) for q in qs},
                   "frac_lt_0.01": float((ex.delta < 0.01).mean()), "frac_lt_0.05": float((ex.delta < 0.05).mean()),
                   "frac_lt_0.1": float((ex.delta < 0.1).mean())}, fh, indent=2)
    rows = []
    for h in load_heads(cfg):
        d, w = aq[aq.head_label == h.label], wh[wh.head_label == h.label]
        rom, frac = ratio_of_means(d, "raw_fwd", "raw_rev")
        wrom, _ = ratio_of_means(w, "whole_head_raw_fwd", "whole_head_raw_rev")
        big = d[d.delta >= 0.1]
        rows.append({
            "head_label": h.label,
            "headline_mean_e_combined": d.groupby("attack_name").e_combined.mean().mean(),
            "ratio_of_means_combined": rom,
            "median_example_e_combined": d.e_combined.median(),
            "mean_e_combined_delta_ge_0.1": big.groupby("attack_name").e_combined.mean().mean(),
            "n_attacks_with_delta_ge_0.1": big.attack_name.nunique(),
            "frac_attacks_raw_min_gt0": frac,
            "whole_head_ratio_of_means_combined": wrom,
            "whole_head_median_example_combined": w.whole_head_combined.median(),
            "edge_over_whole_head_ratio_of_means": rom / wrom if abs(wrom) > 1e-4 else np.nan,
        })
    res = pd.DataFrame(rows)
    res.to_csv(stage / "robustness_by_head.csv", index=False)
    print(res.round(4).to_string())
    write_status(stage, {"status": "success", "finished": now(), "upstream_fingerprint": fp,
                         "note": "post-hoc descriptive only; not part of the pre-registered analysis"})


if __name__ == "__main__":
    main()
