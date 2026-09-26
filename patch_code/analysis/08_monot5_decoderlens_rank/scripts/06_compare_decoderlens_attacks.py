#!/usr/bin/env python3
"""
scripts/06_compare_decoderlens_attacks.py
=========================================
Cross-attack comparison.  Collects every attack's layerwise rank summary and
produces the comparison table + heatmaps + bar plots.

Reads
-----
  outputs/attacks/{attack_name}/ranks/layerwise_rank_summary.csv
  outputs/attacks/{attack_name}/status.json            (if present)
  outputs/attacks/{attack_name}/pairs/alignment_report.json
  outputs/attacks/{attack_name}/sanity/final_layer_equivalence.json

Writes
------
  outputs/attack_comparison/decoderlens_summary.csv
  outputs/attack_comparison/rank_gain_heatmap.png
  outputs/attack_comparison/score_delta_heatmap.png
  outputs/attack_comparison/success_rate_heatmap.png
  outputs/attack_comparison/top_attacks_rank_gain.png
  outputs/attack_comparison/top_attacks_score_delta.png
  outputs/attack_comparison/layer_of_first_positive_effect.png
  outputs/attack_comparison/mean_rank_gain_trajectory.png
  outputs/attack_comparison/mean_score_delta_trajectory.png
  outputs/attack_comparison/attack_vs_control_rank_trajectory.png

Usage
-----
  python scripts/06_compare_decoderlens_attacks.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.comparison import (
    build_summary_row,
    plot_attack_vs_control_trajectory,
    plot_heatmap,
    plot_layer_of_first_positive_effect,
    plot_mean_trajectory,
    plot_top_attacks_bar,
    _pivot,
)
from src.utils import comparison_output_dir, load_config


def _load_json(path: pathlib.Path) -> Optional[Dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _gather_metadata(attack_dir: pathlib.Path, summary: pd.DataFrame) -> Dict:
    """Assemble the metadata/counts dict for build_summary_row."""
    status = _load_json(attack_dir / "status.json") or {}
    align = _load_json(attack_dir / "pairs" / "alignment_report.json") or {}
    first = summary.iloc[0] if not summary.empty else {}

    def _get(key, default=None):
        return status.get(key, align.get(key, default))

    meta = {
        "attack_name": _get("attack_name", attack_dir.name),
        "token": status.get("token", first.get("token") if len(first) else None),
        "position": status.get("position", first.get("position") if len(first) else None),
        "repetitions": status.get("repetitions", first.get("repetitions") if len(first) else None),
        "n_examples_used": _get("n_examples_used"),
        "n_alignment_success": _get("n_alignment_success"),
        "n_alignment_failed": _get("n_alignment_failed"),
        "candidate_top_k": status.get("candidate_top_k", align.get("candidate_top_k")),
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-attack DecoderLens comparison.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    attacks_dir = base_dir / "attacks"
    comp_dir = comparison_output_dir(base_dir)
    top_n = int(cfg["comparison"]["top_attacks_n"])

    if not attacks_dir.is_dir():
        raise SystemExit(f"No attacks directory at {attacks_dir}. Run the pipeline first.")

    long_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []

    for attack_dir in sorted(attacks_dir.iterdir()):
        if not attack_dir.is_dir():
            continue
        summary_csv = attack_dir / "ranks" / "layerwise_rank_summary.csv"
        if not summary_csv.exists():
            print(f"[06] Skipping {attack_dir.name}: no rank summary.")
            continue
        summary = pd.read_csv(summary_csv)
        if summary.empty:
            print(f"[06] Skipping {attack_dir.name}: empty summary.")
            continue
        long_frames.append(summary)

        meta = _gather_metadata(attack_dir, summary)
        sanity = _load_json(attack_dir / "sanity" / "final_layer_equivalence.json")
        summary_rows.append(build_summary_row(summary, meta, sanity))

    if not summary_rows:
        raise SystemExit("[06] No attack summaries found — nothing to compare.")

    long_df = pd.concat(long_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("attack_name")

    summary_path = comp_dir / "decoderlens_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[06] Wrote {len(summary_df)} attack rows -> {summary_path}")

    # --- Heatmaps ------------------------------------------------------------
    plot_heatmap(
        _pivot(long_df, "median_rank_gain_vs_control"),
        comp_dir / "rank_gain_heatmap.png",
        title="Median rank gain vs control (attack) by encoder layer",
        cbar_label="median(control_rank − attack_rank)",
    )
    plot_heatmap(
        _pivot(long_df, "mean_score_delta_vs_control"),
        comp_dir / "score_delta_heatmap.png",
        title="Mean score delta vs control (attack) by encoder layer",
        cbar_label="mean(attack_score − control_score)",
    )
    plot_heatmap(
        _pivot(long_df, "success_rate_vs_control"),
        comp_dir / "success_rate_heatmap.png",
        title="Attack success rate (attack_rank < control_rank) by encoder layer",
        cbar_label="success rate",
        cmap="viridis",
        center_zero=False,
    )

    # --- Top-attack bar plots (final layer) ----------------------------------
    plot_top_attacks_bar(
        summary_df, "final_layer_median_rank_gain",
        comp_dir / "top_attacks_rank_gain.png",
        title=f"Top {top_n} attacks by final-layer median rank gain",
        xlabel="final-layer median rank gain vs control",
        top_n=top_n,
    )
    plot_top_attacks_bar(
        summary_df, "final_layer_mean_score_delta",
        comp_dir / "top_attacks_score_delta.png",
        title=f"Top {top_n} attacks by final-layer mean score delta",
        xlabel="final-layer mean score delta vs control",
        top_n=top_n,
    )

    # --- Aggregate mean/median trajectory across all attacks -----------------
    plot_mean_trajectory(
        long_df, comp_dir / "mean_rank_gain_trajectory.png",
        value_col="mean_rank_gain_vs_control",
        title="Rank gain vs control, averaged across all attacks, by encoder layer",
        ylabel="rank gain vs control (control_rank − attack_rank)",
    )
    plot_mean_trajectory(
        long_df, comp_dir / "mean_score_delta_trajectory.png",
        value_col="mean_score_delta_vs_control",
        title="Score delta vs control, averaged across all attacks, by encoder layer",
        ylabel="score delta vs control",
    )
    plot_attack_vs_control_trajectory(
        long_df, comp_dir / "attack_vs_control_rank_trajectory.png",
        value_col="mean_rank",
        title="Attack vs. padded-control rank, averaged across all attacks, by encoder layer",
        ylabel="mean rank (1 = best) within top-100 candidates",
    )

    # --- Layer of first positive effect --------------------------------------
    plot_layer_of_first_positive_effect(
        summary_df,
        comp_dir / "layer_of_first_positive_effect.png",
        column="first_layer_positive_score_delta",
        title="First encoder layer with a positive attack score delta",
    )

    print("[06] Cross-attack comparison complete.")


if __name__ == "__main__":
    main()
