#!/usr/bin/env python3
"""
scripts/03_aggregate.py
=========================
Derived summaries for Experiment 7 — the per-example CSVs written by
script 02 are the primary data and are never modified here.

Outputs (per run: grid, canonical):
  outputs/{run}/aggregated/composition_summary.csv
      whole-block rows only (head is null): mean % per bucket, per
      (run_type, layer, attack_name).
  outputs/{run}/aggregated/attack_token_trajectory.csv
      whole-block rows only: mean/std attack_token_rank / attack_token_logit,
      per (run_type, layer, attack_name) — the promotion curve data.
  outputs/{run}/aggregated/per_head_token_frequency.csv
      per-head rows (plus whole-block rows at the SAME layers, for
      comparison): how often each top-5 token appears across examples, per
      (run_type, layer, head_or_none, token) — feeds Plot 3.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from exp7lib.run_utils import load_config, resolve_cfg_path

RUN_NAMES = ["grid", "canonical"]
BUCKETS = ["attack", "query", "document", "stopword", "other"]


def load_run_rows(run_dir: pathlib.Path) -> pd.DataFrame:
    csvs = sorted(run_dir.glob("attacks/*/results.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(csvs)} attack CSV(s) in {run_dir.name}/")
    return df


def build_composition_summary(whole_block: pd.DataFrame) -> pd.DataFrame:
    comps = whole_block["bucket_composition"].apply(json.loads).apply(pd.Series)
    comps = comps.reindex(columns=BUCKETS, fill_value=0.0)
    keyed = pd.concat([whole_block[["run_type", "layer", "attack_name"]], comps], axis=1)
    return keyed.groupby(["run_type", "layer", "attack_name"]).mean().reset_index()


def build_attack_token_trajectory(whole_block: pd.DataFrame) -> pd.DataFrame:
    return whole_block.groupby(["run_type", "layer", "attack_name"]).agg(
        n=("qid", "size"),
        attack_token_rank_mean=("attack_token_rank", "mean"),
        attack_token_rank_std=("attack_token_rank", "std"),
        attack_token_logit_mean=("attack_token_logit", "mean"),
        attack_token_logit_std=("attack_token_logit", "std"),
    ).reset_index()


def build_per_head_token_frequency(df: pd.DataFrame, top_k_col: str = "top_k5_tokens") -> pd.DataFrame:
    flagged_layers = sorted(df.loc[df["head"].notna(), "layer"].unique().tolist())
    sub = df[(df["head"].notna()) | (df["layer"].isin(flagged_layers) & df["head"].isna())]

    records = []
    for (run_type, layer, head), group in sub.groupby(["run_type", "layer", "head"], dropna=False):
        counter = Counter()
        for cell in group[top_k_col]:
            counter.update(json.loads(cell))
        n = len(group)
        for token, count in counter.most_common(10):
            records.append({
                "run_type": run_type, "layer": layer, "head": head,
                "token": token, "frequency_pct": 100.0 * count / n, "n_examples": n,
            })
    return pd.DataFrame(records)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate Experiment 7 per-example results.")
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

        whole_block = df[df["head"].isna()]

        comp = build_composition_summary(whole_block)
        comp.to_csv(agg_dir / "composition_summary.csv", index=False)

        traj = build_attack_token_trajectory(whole_block)
        traj.to_csv(agg_dir / "attack_token_trajectory.csv", index=False)

        freq = build_per_head_token_frequency(df)
        freq.to_csv(agg_dir / "per_head_token_frequency.csv", index=False)

        print(f"  wrote {len(comp)} composition rows, {len(traj)} trajectory rows, "
              f"{len(freq)} token-frequency rows → {agg_dir}")


if __name__ == "__main__":
    main()
