#!/usr/bin/env python3
"""
scripts/05_plot_per_attack_results.py
=====================================
Produce the four per-attack DecoderLens plots from layerwise_rank_summary.csv:

  plots/rank_over_layers.png
  plots/rank_gain_over_layers.png
  plots/score_delta_over_layers.png
  plots/success_rate_over_layers.png

Usage
-----
  python scripts/05_plot_per_attack_results.py --config configs/default.yaml \
      --attack-name relevant_start_5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.attack_registry import AttackSpec, discover_attacks
from src.plotting import plot_all_per_attack
from src.utils import attack_output_dirs, load_config


def plot_attack(spec: AttackSpec, cfg: dict, base_dir: pathlib.Path) -> None:
    out_dirs = attack_output_dirs(base_dir, spec.attack_name)
    summary_csv = out_dirs["ranks"] / "layerwise_rank_summary.csv"
    if not summary_csv.exists():
        print(f"[05] {spec.attack_name}: missing {summary_csv}, skipping.")
        return
    summary = pd.read_csv(summary_csv)
    if summary.empty:
        print(f"[05] {spec.attack_name}: empty summary, skipping.")
        return
    plot_all_per_attack(summary, out_dirs["plots"], spec.attack_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-attack DecoderLens results.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--attack-name", default=None,
                   help="Single attack to plot (default: all discovered).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    attacks = discover_attacks(cfg["attacks"])
    if args.attack_name:
        attacks = [a for a in attacks if a.attack_name == args.attack_name]
        if not attacks:
            raise SystemExit(f"Attack '{args.attack_name}' not found.")
    for spec in attacks:
        plot_attack(spec, cfg, base_dir)


if __name__ == "__main__":
    main()
