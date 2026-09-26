#!/usr/bin/env python3
"""
scripts/03_aggregate.py
========================
Derived summaries for Experiment 6 — mean ± std per (layer, condition,
attack), separately for encoder_self_attn and decoder_cross_attn. This is a
DERIVED view; the per-example CSVs written by script 02 are the primary
data and are never modified here.

Outputs (per run: grid, canonical):
  outputs/{run}/aggregated/layer_condition_attack_summary.csv
      one row per (region, attack_name, layer, condition):
      mean/std/n of fwd_effect, rev_effect, combined_effect.
  outputs/{run}/aggregated/layer_condition_summary.csv
      same, pooled over all attacks (for the grid run) / over all
      examples (for canonical).
  outputs/{run}/aggregated/attention_summary.csv
      one row per (attack_name, layer): mean/std attention_q_to_d /
      attention_d_to_q, clean and attack (encoder rows only).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from exp6lib.run_utils import load_config, resolve_cfg_path

RUN_NAMES = ["grid", "canonical"]
EFFECT_METRICS = ["fwd_effect", "rev_effect", "combined_effect"]
ATTENTION_METRICS = ["attention_q_to_d_clean", "attention_d_to_q_clean",
                      "attention_q_to_d_attack", "attention_d_to_q_attack"]


def load_run_rows(run_dir: pathlib.Path) -> pd.DataFrame:
    csvs = sorted(run_dir.glob("attacks/*/results.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(csvs)} attack CSV(s) in {run_dir.name}/")
    return df


def summarize(df: pd.DataFrame, keys: list, metrics: list) -> pd.DataFrame:
    agg = df.groupby(keys, dropna=False).agg(
        n=("qid", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).reset_index()
    return agg.sort_values(keys).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate Experiment 6 per-example results.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--run", choices=RUN_NAMES, default=None)
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    runs = [args.run] if args.run else RUN_NAMES

    for run_name in runs:
        run_dir = outputs_base / run_name
        if not run_dir.exists():
            print(f"[03_aggregate] {run_name}: no outputs — skipping.")
            continue
        print(f"[03_aggregate] {run_name}:")
        df = load_run_rows(run_dir)
        if df.empty:
            print("  no rows — skipping.")
            continue

        agg_dir = run_dir / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)

        by_layer_cond_attack = summarize(df, ["region", "attack_name", "layer", "condition"], EFFECT_METRICS)
        by_layer_cond_attack.to_csv(agg_dir / "layer_condition_attack_summary.csv", index=False)

        by_layer_cond = summarize(df, ["region", "layer", "condition"], EFFECT_METRICS)
        by_layer_cond.to_csv(agg_dir / "layer_condition_summary.csv", index=False)

        enc_df = df[df["region"] == "encoder_self_attn"].drop_duplicates(subset=["qid", "docid", "attack_name", "layer"])
        attn_summary = summarize(enc_df, ["attack_name", "layer"], ATTENTION_METRICS)
        attn_summary.to_csv(agg_dir / "attention_summary.csv", index=False)

        print(f"  wrote {len(by_layer_cond_attack)} (region,attack,layer,condition) rows, "
              f"{len(by_layer_cond)} (region,layer,condition) rows, "
              f"{len(attn_summary)} attention rows → {agg_dir}")


if __name__ == "__main__":
    main()
