#!/usr/bin/env python3
"""
scripts/04_plot_results.py
===========================
Stage 04: Generate all plots from the patching results.

Input
-----
  outputs/scores/all_scores.csv                      (from Stage 01)
  outputs/scores/selected_examples.jsonl             (from Stage 01)
  outputs/patching/layer_patching_results.csv        (from Stage 03)

Outputs
-------
  outputs/plots/attack_delta_hist.png
  outputs/plots/forward_layer_component_heatmap.png
  outputs/plots/reverse_layer_component_heatmap.png
  outputs/plots/combined_layer_component_heatmap.png

See src/plotting.py for detailed explanation of each plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.plotting import (
    plot_attack_delta_histogram,
    plot_combined_heatmap,
    plot_patching_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 04: plot patching results.")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    p.add_argument(
        "--attack-name",
        default=None,
        help="If set, read/write outputs under outputs/attacks/{attack-name}/.",
    )
    return p.parse_args()


def _resolve_base_dir(cfg: dict, attack_name: str | None) -> pathlib.Path:
    base = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    if attack_name:
        return base / "attacks" / attack_name
    return base


def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()
    print(f"[04_plot_results] Loading config from: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    selection_cfg = cfg["selection"]

    base_dir   = _resolve_base_dir(cfg, args.attack_name)
    plots_dir  = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_scores_path   = base_dir / "scores" / "all_scores.csv"
    selected_path     = base_dir / "scores" / "selected_examples.jsonl"
    agg_results_path  = base_dir / "patching" / "layer_patching_results.csv"

    # -------------------------------------------------------------------------
    # Plot 1: Attack delta histogram
    # -------------------------------------------------------------------------
    if not selected_path.exists():
        print(f"WARNING: {selected_path} not found, skipping histogram.")
    else:
        selected = []
        with open(selected_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    selected.append(json.loads(line))
        print(f"[04_plot_results] Loaded {len(selected)} selected examples for histogram.")

        plot_attack_delta_histogram(
            scored_pairs=selected,
            output_path=plots_dir / "attack_delta_hist.png",
            min_attack_delta=selection_cfg["min_attack_delta"],
        )

    # -------------------------------------------------------------------------
    # Plots 2-4: Patching heatmaps
    # -------------------------------------------------------------------------
    if not agg_results_path.exists():
        print(
            f"WARNING: {agg_results_path} not found.\n"
            "Skipping heatmap plots.  Run Stage 03 first."
        )
        return

    agg_df = pd.read_csv(agg_results_path)
    print(f"[04_plot_results] Loaded aggregated results: {len(agg_df)} rows.")

    # Forward patching heatmap
    plot_patching_heatmap(
        agg_df=agg_df,
        direction="forward",
        output_path=plots_dir / "forward_layer_component_heatmap.png",
    )

    # Reverse patching heatmap
    plot_patching_heatmap(
        agg_df=agg_df,
        direction="reverse",
        output_path=plots_dir / "reverse_layer_component_heatmap.png",
    )

    # Combined importance heatmap
    plot_combined_heatmap(
        agg_df=agg_df,
        output_path=plots_dir / "combined_layer_component_heatmap.png",
    )

    print(f"[04_plot_results] All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
