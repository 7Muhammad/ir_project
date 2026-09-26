#!/usr/bin/env python3
"""
scripts/03_aggregate.py
=========================
Aggregates the raw per-(attack, example, sender, receiver) rows written by
scripts/02_run_sweep.py into the tables described in the task spec. No
importance threshold is applied anywhere here -- all continuous values are
saved as-is; thresholding is left for interpretation after inspecting the
distribution.

Outputs (outputs/aggregates/):
  per_attack_sender_receiver.csv   -- Table 1: one row per (attack, A, B)
  global_sender_receiver_long.csv  -- Table 2: attack-balanced + example-weighted, long form
  global_matrix_combined.csv       -- Table 2 as an 18x31 wide matrix (attack-balanced mean_combined)
  global_matrix_forward.csv        -- same, mean_forward
  global_matrix_reverse.csv        -- same, mean_reverse
  sender_summary.csv               -- Table 3
  receiver_summary.csv             -- Table 4
  path_stability.csv               -- Table 5 (folded into global_sender_receiver_long.csv too)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(EXP_DIR))

from exp13lib.head_lists import load_receivers, load_senders  # noqa: E402
from exp13lib.run_utils import load_config, resolve_cfg_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 13 -- aggregation.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def load_raw(raw_dir: pathlib.Path) -> pd.DataFrame:
    csvs = sorted(raw_dir.glob("*/paths.csv"))
    if not csvs:
        sys.exit(f"[aggregate] no raw CSVs found under {raw_dir} -- run scripts/02_run_sweep.py first.")
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"[aggregate] loaded {len(df)} rows from {len(csvs)} attacks")
    return df


def per_attack_sender_receiver(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["attack_name", "sender_name", "receiver_name", "sender_layer", "sender_head",
                     "receiver_layer", "receiver_head"])
    agg = g.agg(
        mean_forward=("path_forward", "mean"),
        mean_reverse=("path_reverse", "mean"),
        mean_combined=("path_combined", "mean"),
        median_combined=("path_combined", "median"),
        std_combined=("path_combined", "std"),
        fraction_positive_forward=("path_forward", lambda s: float((s > 0).mean())),
        fraction_positive_reverse=("path_reverse", lambda s: float((s > 0).mean())),
        fraction_positive_combined=("path_combined", lambda s: float((s > 0).mean())),
        n_examples=("path_combined", "size"),
    ).reset_index()
    agg["std_combined"] = agg["std_combined"].fillna(0.0)
    return agg


def global_sender_receiver(per_attack: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    Attack-balanced global mean: average the per-attack means (Table 1)
    across attacks where the pair has valid examples, so an attack with
    more qualifying examples does not dominate merely by row count.
    Also computes the example-weighted (flat) version as a secondary
    diagnostic, plus n_attacks / total_examples per pair.
    """
    balanced = per_attack.groupby(["sender_name", "receiver_name", "sender_layer", "sender_head",
                                    "receiver_layer", "receiver_head"]).agg(
        attack_balanced_mean_forward=("mean_forward", "mean"),
        attack_balanced_mean_reverse=("mean_reverse", "mean"),
        attack_balanced_mean_combined=("mean_combined", "mean"),
        attack_balanced_median_combined=("mean_combined", "median"),
        n_attacks=("attack_name", "nunique"),
        total_examples=("n_examples", "sum"),
        n_attacks_positive_combined=("mean_combined", lambda s: int((s > 0).sum())),
    ).reset_index()
    balanced["fraction_attacks_positive_combined"] = (
        balanced["n_attacks_positive_combined"] / balanced["n_attacks"]
    )

    weighted = raw.groupby(["sender_name", "receiver_name"]).agg(
        example_weighted_mean_forward=("path_forward", "mean"),
        example_weighted_mean_reverse=("path_reverse", "mean"),
        example_weighted_mean_combined=("path_combined", "mean"),
    ).reset_index()

    merged = balanced.merge(weighted, on=["sender_name", "receiver_name"], how="left")
    return merged


def wide_matrix(global_df: pd.DataFrame, senders, receivers, value_col: str) -> pd.DataFrame:
    sender_order = [s.label for s in senders]
    receiver_order = [r.label for r in receivers]
    pivot = global_df.pivot(index="sender_name", columns="receiver_name", values=value_col)
    pivot = pivot.reindex(index=sender_order, columns=receiver_order)
    return pivot


def sender_summary(global_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sender_name, sub in global_df.groupby("sender_name"):
        best = sub.loc[sub["attack_balanced_mean_combined"].idxmax()]
        rows.append({
            "sender_name": sender_name,
            "mean_path_effect_across_receivers": sub["attack_balanced_mean_combined"].mean(),
            "max_receiver_path_effect": sub["attack_balanced_mean_combined"].max(),
            "best_receiver": best["receiver_name"],
            "n_receivers_positive": int((sub["attack_balanced_mean_combined"] > 0).sum()),
            "n_receivers_total": len(sub),
            "fraction_receivers_positive": float((sub["attack_balanced_mean_combined"] > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_path_effect_across_receivers", ascending=False)


def receiver_summary(global_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for receiver_name, sub in global_df.groupby("receiver_name"):
        best = sub.loc[sub["attack_balanced_mean_combined"].idxmax()]
        rows.append({
            "receiver_name": receiver_name,
            "mean_path_effect_across_senders": sub["attack_balanced_mean_combined"].mean(),
            "max_sender_path_effect": sub["attack_balanced_mean_combined"].max(),
            "best_sender": best["sender_name"],
            "n_senders_positive": int((sub["attack_balanced_mean_combined"] > 0).sum()),
            "n_senders_total": len(sub),
            "fraction_senders_positive": float((sub["attack_balanced_mean_combined"] > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_path_effect_across_senders", ascending=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    raw_dir = resolve_cfg_path(cfg, cfg["outputs"]["raw_dir"])
    agg_dir = resolve_cfg_path(cfg, cfg["outputs"]["aggregates_dir"])
    agg_dir.mkdir(parents=True, exist_ok=True)

    senders = load_senders()
    receivers = load_receivers()

    raw = load_raw(raw_dir)

    per_attack = per_attack_sender_receiver(raw)
    per_attack.to_csv(agg_dir / "per_attack_sender_receiver.csv", index=False)
    print(f"[aggregate] wrote per_attack_sender_receiver.csv ({len(per_attack)} rows)")

    global_df = global_sender_receiver(per_attack, raw)
    global_df.to_csv(agg_dir / "global_sender_receiver_long.csv", index=False)
    print(f"[aggregate] wrote global_sender_receiver_long.csv ({len(global_df)} rows)")

    for name, col in [
        ("global_matrix_combined.csv", "attack_balanced_mean_combined"),
        ("global_matrix_forward.csv", "attack_balanced_mean_forward"),
        ("global_matrix_reverse.csv", "attack_balanced_mean_reverse"),
        ("global_matrix_stability.csv", "fraction_attacks_positive_combined"),
    ]:
        wide_matrix(global_df, senders, receivers, col).to_csv(agg_dir / name)
        print(f"[aggregate] wrote {name}")

    sender_summary(global_df).to_csv(agg_dir / "sender_summary.csv", index=False)
    receiver_summary(global_df).to_csv(agg_dir / "receiver_summary.csv", index=False)
    print("[aggregate] wrote sender_summary.csv, receiver_summary.csv")

    stability_cols = ["sender_name", "receiver_name", "n_attacks", "n_attacks_positive_combined",
                       "fraction_attacks_positive_combined", "attack_balanced_mean_combined"]
    global_df[stability_cols].to_csv(agg_dir / "path_stability.csv", index=False)
    print("[aggregate] wrote path_stability.csv")

    n_pairs_seen = global_df.shape[0]
    print(f"[aggregate] {n_pairs_seen} / {len(senders)*len(receivers)} (sender, receiver) pairs have raw data")


if __name__ == "__main__":
    main()
