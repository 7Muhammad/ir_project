#!/usr/bin/env python3
"""
scripts/06_statistics.py
=========================
Formal inference for the PRIMARY (all-query) condition only.

Per head (exactly one primary test per head; 18 in the full run):
  * equal-weight means of e_combined / e_fwd / e_rev
  * 95% percentile bootstrap CI, resampling EXAMPLES WITHIN EACH ATTACK
    (attacks fixed, equal weight) — for e_combined (primary) and, for
    display only, e_fwd / e_rev
  * one-sided sign-flip test on the attack-level mean e_combined values
    (H1: equal-weight mean > 0), p = (extreme + 1) / (N + 1)
  * Benjamini-Hochberg across the heads at alpha = 0.05

No token-level, per-attack, or separate forward/reverse tests.

Outputs: outputs/06_statistics/head_statistics.csv, statistics_meta.json
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

import pandas as pd  # noqa: E402

from exp15lib.config import load_config, output_dir  # noqa: E402
from exp15lib.heads import load_heads  # noqa: E402
from exp15lib.run_utils import cheap_stage_is_current, now, upstream_fingerprint, write_status  # noqa: E402
from exp15lib.statistics import (  # noqa: E402
    benjamini_hochberg, bootstrap_within_attack, equal_weight_mean, sign_flip_test,
)


def parse_args():
    p = argparse.ArgumentParser(description="Exp 15 stage 06: statistics")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    s = cfg["statistics"]
    agg = out / "05_aggregate"
    stage = out / "06_statistics"
    raw_path = agg / "raw_all_query_edges.csv"
    if not (agg / "status.json").exists() or not raw_path.exists():
        raise RuntimeError("stage 05 outputs missing; run scripts/05_aggregate.py first")
    fp = upstream_fingerprint([agg / "status.json"]) + json.dumps(s, sort_keys=True)
    if cheap_stage_is_current(stage, fp, ["head_statistics.csv"], args.force):
        print("[06] statistics up to date; skipping")
        return
    if s["bootstrap_method"] != "percentile":
        raise ValueError("only the percentile bootstrap is implemented")

    heads = load_heads(cfg)
    df = pd.read_csv(raw_path)
    rows = []
    for stream, h in enumerate(heads):
        d = df[df.head_label == h.label]
        attacks = list(dict.fromkeys(d.attack_name))
        row = {"head_label": h.label, "layer": h.layer, "head": h.head_idx,
               "n_attacks": len(attacks), "n_examples": len(d)}
        for metric in ("e_combined", "e_fwd", "e_rev"):
            groups = [d.loc[d.attack_name == a, metric].to_numpy() for a in attacks]
            row[f"mean_{metric}"] = equal_weight_mean(groups)
            bs = bootstrap_within_attack(groups, int(s["bootstrap_reps"]), float(s["bootstrap_ci_level"]),
                                         int(s["bootstrap_seed"]), stream)
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bs["ci_low"], bs["ci_high"]
            row[f"{metric}_boot_se"] = bs["boot_se"]
        attack_means = d.groupby("attack_name", sort=False)["e_combined"].mean().reindex(attacks).to_numpy()
        sf = sign_flip_test(attack_means, int(s["n_sign_flips"]), int(s["sign_flip_seed"]), stream)
        row["sign_flip_observed"] = sf["observed"]
        row["p_one_sided"] = sf["p_one_sided"]
        row["n_sign_flip_extreme"] = sf["n_extreme"]
        rows.append(row)

    res = pd.DataFrame(rows)
    alpha = float(s["fdr_alpha"])
    bh = benjamini_hochberg(res.p_one_sided.tolist(), alpha)
    res["q_bh"] = bh["q"]
    res[f"reject_FDR_{alpha}"] = bh["reject"]
    stage.mkdir(parents=True, exist_ok=True)
    res.to_csv(stage / "head_statistics.csv", index=False)
    meta = {
        "n_primary_tests": len(res), "test": "one-sided sign-flip on attack-level mean e_combined (H1: mean > 0)",
        "p_value_formula": "(n_extreme + 1) / (n_sign_flips + 1)",
        "n_sign_flips": s["n_sign_flips"], "sign_flip_seed": s["sign_flip_seed"],
        "bootstrap": "percentile; examples resampled with replacement within each attack; attacks fixed; equal weight",
        "bootstrap_reps": s["bootstrap_reps"], "bootstrap_seed": s["bootstrap_seed"],
        "bootstrap_ci_level": s["bootstrap_ci_level"],
        "rng": "numpy default_rng(SeedSequence([seed, head_index_in_config_order]))",
        "multiple_comparisons": f"Benjamini-Hochberg, alpha={alpha}",
    }
    with open(stage / "statistics_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(res[["head_label", "mean_e_combined", "e_combined_ci_low", "e_combined_ci_high",
               "mean_e_fwd", "mean_e_rev", "p_one_sided", "q_bh", f"reject_FDR_{alpha}"]].to_string())
    write_status(stage, {"status": "success", "finished": now(), "upstream_fingerprint": fp})


if __name__ == "__main__":
    main()
