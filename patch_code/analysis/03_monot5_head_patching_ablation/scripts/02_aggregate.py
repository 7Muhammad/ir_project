#!/usr/bin/env python3
"""
scripts/02_aggregate.py
=======================
Derived summaries for Experiment 3 — mean ± std per (head, attack) and per
head.  This is a DERIVED view; the per-example CSVs written by script 01 are
the primary data and are never modified here.

For each run present in the outputs directory:

  outputs/{run}/aggregated/head_attack_summary.csv
      one row per (attack_name, layer, component, head_idx):
      mean/std/n of every effect / score-drop metric.

  outputs/{run}/aggregated/head_summary.csv
      one row per (layer, component, head_idx), pooled over all attacks
      AND examples (per-example rows weighted equally).

Usage:
  python scripts/02_aggregate.py --config configs/default.yaml [--run grid_a]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))  # for src.* (imported via headlib.run_utils)
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from headlib.run_utils import load_config, resolve_cfg_path

RUN_NAMES = ["grid_a", "grid_b", "clean_a", "clean_b"]

GRID_METRICS = [
    "fwd_effect", "rev_effect", "combined_effect",
    "score_drop_zero", "score_drop_mean",
]
CLEAN_METRICS = ["score_drop_zero", "score_drop_mean"]

HEAD_KEYS = ["layer", "component", "head_idx"]


def load_run_rows(run_dir: pathlib.Path) -> pd.DataFrame:
    """Concatenate every per-attack head_results.csv of one run."""
    csvs = sorted(run_dir.glob("attacks/*/head_results.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(csvs)} attack CSV(s) in {run_dir.name}/")
    return df


def summarize(df: pd.DataFrame, keys: list, metrics: list) -> pd.DataFrame:
    """mean/std/n per key group for each metric, flat column names."""
    agg = df.groupby(keys, dropna=False).agg(
        n=("qid", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).reset_index()
    return agg.sort_values(keys).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate per-example head results.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--run", choices=RUN_NAMES, default=None,
                   help="Aggregate a single run (default: every run with outputs).")
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    runs = [args.run] if args.run else RUN_NAMES

    for run_name in runs:
        run_dir = outputs_base / run_name
        if not run_dir.exists():
            print(f"[02_aggregate] {run_name}: no outputs — skipping.")
            continue
        print(f"[02_aggregate] {run_name}:")
        df = load_run_rows(run_dir)
        if df.empty:
            print(f"  no rows — skipping.")
            continue

        metrics = GRID_METRICS if run_name.startswith("grid") else CLEAN_METRICS
        metrics = [m for m in metrics if m in df.columns]

        agg_dir = run_dir / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)

        by_head_attack = summarize(df, ["attack_name"] + HEAD_KEYS, metrics)
        by_head_attack.to_csv(agg_dir / "head_attack_summary.csv", index=False)
        by_head = summarize(df, HEAD_KEYS, metrics)
        by_head.to_csv(agg_dir / "head_summary.csv", index=False)
        print(f"  wrote {len(by_head_attack)} (head, attack) rows and "
              f"{len(by_head)} head rows → {agg_dir}")


if __name__ == "__main__":
    main()
