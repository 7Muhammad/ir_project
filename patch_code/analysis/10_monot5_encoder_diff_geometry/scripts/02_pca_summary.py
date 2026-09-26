#!/usr/bin/env python3
"""
scripts/02_pca_summary.py
============================
Experiment 10, stage 2: per (attack, layer) SVD-based rank-structure
summary — variance explained by the top 1/3/10 components, effective rank
(participation ratio), plus the sign-aligned top-1 direction and mean
direction (saved for stage 3's cross-attack cosine comparison).

Reads outputs/diffs/{attack}/diffs.npz (stage 1).
Writes:
  outputs/geometry/{attack}/pca_summary.csv        (one row per layer)
  outputs/geometry/{attack}/top_directions.npz      (per-layer top/mean directions)
  outputs/geometry/pca_summary_all.csv              (concatenated, all attacks)
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))  # src.*
sys.path.insert(0, str(EXP6_DIR))  # exp6lib.*
sys.path.insert(0, str(EXP_DIR))   # exp10lib.*

import numpy as np

from exp10lib.geometry import compute_pca_summary
from exp10lib.run_utils import load_config, resolve_cfg_path, select_attacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 10 stage 2: PCA/SVD rank-structure summary.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    layers = cfg["layers"]
    k_values = cfg["pca_k_values"]
    diffs_dir = outputs_base / "diffs"
    geometry_dir = outputs_base / "geometry"

    attack_names = select_attacks(cfg)
    k_cols = [f"variance_explained_top{k}" for k in k_values]
    all_rows: List[Dict] = []

    for attack_name in attack_names:
        npz_path = diffs_dir / attack_name / "diffs.npz"
        if not npz_path.exists():
            print(f"  [SKIP] {attack_name}: no diffs.npz (run scripts/01_extract_diffs.py first).")
            continue
        data = np.load(npz_path, allow_pickle=True)

        attack_dir = geometry_dir / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict] = []
        direction_kwargs: Dict[str, np.ndarray] = {}

        for L in layers:
            diffs = data[f"diff_layer{L}"]
            summary = compute_pca_summary(diffs, k_values=k_values)
            row = {
                "attack_name": attack_name, "layer": L,
                "n_positions": summary["n_positions"],
                **{col: summary[col] for col in k_cols},
                "effective_rank": summary["effective_rank"],
            }
            rows.append(row)
            all_rows.append(row)
            if summary["top_direction"] is not None:
                direction_kwargs[f"top_direction_layer{L}"] = summary["top_direction"]
                direction_kwargs[f"mean_direction_layer{L}"] = summary["mean_direction"]

        csv_path = attack_dir / "pca_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["attack_name", "layer", "n_positions", *k_cols, "effective_rank"])
            writer.writeheader()
            writer.writerows(rows)

        if direction_kwargs:
            np.savez_compressed(attack_dir / "top_directions.npz", **direction_kwargs)

        print(f"  [OK] {attack_name}: " + ", ".join(
            f"L{r['layer']}: top1={r['variance_explained_top1']:.3f} eff_rank={r['effective_rank']:.2f}"
            for r in rows
        ))

    combined_path = geometry_dir / "pca_summary_all.csv"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["attack_name", "layer", "n_positions", *k_cols, "effective_rank"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[02_pca_summary] Wrote {len(all_rows)} rows -> {combined_path}")
    print(f"[02_pca_summary] Next: python scripts/03_cross_attack_cosine.py --config {args.config}")


if __name__ == "__main__":
    main()
