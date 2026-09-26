#!/usr/bin/env python3
"""
scripts/03_cross_attack_cosine.py
====================================
Experiment 10, stage 3: pairwise cosine similarity of each attack's
sign-aligned top-1 principal direction (from stage 2), one matrix per
layer. Rows/columns ordered by attack strength (Experiment 1's
mean_attack_delta, descending — the order select_attacks() already
returns) for interpretability.

Writes outputs/geometry/cross_attack_cosine_matrix_layer{L}.csv for each
configured layer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))  # src.*
sys.path.insert(0, str(EXP6_DIR))  # exp6lib.*
sys.path.insert(0, str(EXP_DIR))   # exp10lib.*

import numpy as np

from exp10lib.geometry import cosine_similarity_matrix
from exp10lib.run_utils import load_config, resolve_cfg_path, select_attacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 10 stage 3: cross-attack cosine similarity.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    layers = cfg["layers"]
    geometry_dir = outputs_base / "geometry"

    attack_names = select_attacks(cfg)

    for L in layers:
        directions = {}
        for attack_name in attack_names:
            npz_path = geometry_dir / attack_name / "top_directions.npz"
            if not npz_path.exists():
                print(f"  [SKIP] {attack_name}: no top_directions.npz (run scripts/02_pca_summary.py first).")
                continue
            data = np.load(npz_path)
            key = f"top_direction_layer{L}"
            if key not in data:
                continue
            directions[attack_name] = data[key]

        if len(directions) < 2:
            print(f"  [SKIP] layer {L}: fewer than 2 attacks with a direction.")
            continue

        mat = cosine_similarity_matrix(directions)
        out_path = geometry_dir / f"cross_attack_cosine_matrix_layer{L}.csv"
        mat.to_csv(out_path)
        off_diag = mat.values[~np.eye(len(mat), dtype=bool)]
        print(f"  [OK] layer {L}: {len(directions)} attacks -> {out_path} "
              f"(mean off-diag cosine={off_diag.mean():.3f}, "
              f"min={off_diag.min():.3f}, max={off_diag.max():.3f})")

    print(f"\n[03_cross_attack_cosine] Next: python scripts/04_position_concentration.py --config {args.config}")


if __name__ == "__main__":
    main()
