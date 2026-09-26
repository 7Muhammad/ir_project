#!/usr/bin/env python3
"""
scripts/03_aggregate.py
=========================
Derived summaries for Experiment 2 — the per-example CSVs written by
script 02 are the primary data and are never modified here.

For each tier (grid_a, grid_b):
  outputs/interventions/{tier}/aggregated/defense_by_head_scale_attack.csv
      mean/std/n of delta_toward_control per (layer, component, head_idx,
      scale, attack_name) — "which scale, at which head, best defends
      against which attack."
  outputs/interventions/{tier}/aggregated/defense_by_head_scale.csv
      same, pooled over attacks (one row per head x scale).
  outputs/interventions/{tier}/aggregated/sufficiency_by_head_scale_attack.csv
  outputs/interventions/{tier}/aggregated/sufficiency_by_head_scale.csv
      analogous, for delta_toward_attack (2b).
  outputs/interventions/{tier}/aggregated/best_scale_by_head.csv
      the scale with the highest mean delta_toward_control per head — used
      by plot 2's "best scale only" breadth check.

Separately, direction-norm vs Experiment-3 causal-ranking cross-reference
(Part 1, only meaningful for decoder heads — Experiment 3 never scored
encoder heads):
  outputs/directions/direction_norm_vs_exp3_effect.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd  # noqa: E402

from exp2lib.run_utils import load_config, resolve_cfg_path  # noqa: E402

TIER_NAMES = ["grid_a", "grid_b"]
HEAD_KEYS = ["layer", "component", "head_idx"]


def load_tier_rows(tier_dir: pathlib.Path) -> pd.DataFrame:
    csvs = sorted(tier_dir.glob("attacks/*/rows.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(csvs)} attack CSV(s)")
    return df


def summarize(df: pd.DataFrame, keys: list, metric: str) -> pd.DataFrame:
    agg = df.groupby(keys, dropna=False).agg(
        n=("qid", "size"),
        **{f"{metric}_mean": (metric, "mean")},
        **{f"{metric}_std": (metric, "std")},
    ).reset_index()
    return agg.sort_values(keys).reset_index(drop=True)


def aggregate_tier(tier: str, outputs_base: pathlib.Path) -> None:
    tier_dir = outputs_base / "interventions" / tier
    if not tier_dir.exists():
        print(f"[03_aggregate] {tier}: no outputs — skipping.")
        return
    print(f"[03_aggregate] {tier}:")
    df = load_tier_rows(tier_dir)
    if df.empty:
        print("  no rows — skipping.")
        return

    agg_dir = tier_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    defense = df[df["intervention_type"] == "subtract"].copy()
    if not defense.empty:
        by_hsa = summarize(defense, HEAD_KEYS + ["scale", "attack_name"], "delta_toward_control")
        by_hsa.to_csv(agg_dir / "defense_by_head_scale_attack.csv", index=False)
        by_hs = summarize(defense, HEAD_KEYS + ["scale"], "delta_toward_control")
        by_hs.to_csv(agg_dir / "defense_by_head_scale.csv", index=False)

        best_idx = by_hs.groupby(HEAD_KEYS, dropna=False)["delta_toward_control_mean"].idxmax()
        best_scale = by_hs.loc[best_idx].reset_index(drop=True)
        best_scale.to_csv(agg_dir / "best_scale_by_head.csv", index=False)
        print(f"  defense: {len(by_hsa)} (head,scale,attack) rows, "
              f"{len(by_hs)} (head,scale) rows, {len(best_scale)} best-scale rows")

    sufficiency = df[df["intervention_type"] == "add"].copy()
    if not sufficiency.empty:
        by_hsa = summarize(sufficiency, HEAD_KEYS + ["scale", "attack_name"], "delta_toward_attack")
        by_hsa.to_csv(agg_dir / "sufficiency_by_head_scale_attack.csv", index=False)
        by_hs = summarize(sufficiency, HEAD_KEYS + ["scale"], "delta_toward_attack")
        by_hs.to_csv(agg_dir / "sufficiency_by_head_scale.csv", index=False)
        print(f"  sufficiency: {len(by_hsa)} (head,scale,attack) rows, {len(by_hs)} (head,scale) rows")


def cross_reference_direction_norms(cfg: dict, outputs_base: pathlib.Path) -> None:
    """
    Correlate Part 1's per-head direction norm (grid_a tier) against
    Experiment 3's causal combined_effect_mean ranking. Decoder heads only —
    Experiment 3 never scored encoder heads, so there is no ranking to
    correlate encoder_self_attn directions against.
    """
    norms_path = outputs_base / "directions" / "grid_a_direction_norms.csv"
    if not norms_path.exists():
        print("[03_aggregate] no grid_a_direction_norms.csv — skipping cross-reference.")
        return
    norms = pd.read_csv(norms_path)
    norms = norms[
        (norms["granularity"] == "per_head")
        & (norms["component"].isin(["decoder_self_attn", "decoder_cross_attn"]))
    ]

    exp3_csv = resolve_cfg_path(cfg, cfg["flagged_heads"]["source_csv"])
    exp3 = pd.read_csv(exp3_csv)

    merged = norms.merge(exp3[HEAD_KEYS + ["combined_effect_mean"]], on=HEAD_KEYS, how="inner")
    if merged.empty:
        print("[03_aggregate] no overlap between direction norms and Exp3 heads.")
        return

    pearson = merged["direction_norm"].corr(merged["combined_effect_mean"], method="pearson")
    spearman = merged["direction_norm"].corr(merged["combined_effect_mean"], method="spearman")

    out_path = outputs_base / "directions" / "direction_norm_vs_exp3_effect.csv"
    merged.to_csv(out_path, index=False)
    print(f"[03_aggregate] direction-norm vs Exp3 combined-effect: "
          f"pearson={pearson:.3f} spearman={spearman:.3f} (n={len(merged)}) -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate Experiment 2 per-example results.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--tier", choices=TIER_NAMES, default=None,
                    help="Aggregate a single tier (default: every tier with outputs).")
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    tiers = [args.tier] if args.tier else TIER_NAMES

    for tier in tiers:
        aggregate_tier(tier, outputs_base)

    cross_reference_direction_norms(cfg, outputs_base)


if __name__ == "__main__":
    main()
