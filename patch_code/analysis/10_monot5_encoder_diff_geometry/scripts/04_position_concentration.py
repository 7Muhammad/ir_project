#!/usr/bin/env python3
"""
scripts/04_position_concentration.py
=======================================
Experiment 10, stage 4: mean L2 norm of the diff vector, broken out by
position tag (injected_attack_token / other_document / query /
connective_template) and layer, per attack. Tests whether the
attack-control shift stays localized to the injected tokens or has already
spread into other-document / query positions by the given layer.

Writes outputs/geometry/position_concentration.csv
  (one row per attack, layer, position_tag: mean_diff_norm, std, n_positions).
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

from exp10lib.tagging import ALL_TAGS
from exp10lib.run_utils import load_config, resolve_cfg_path, select_attacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 10 stage 4: positional concentration of the diff.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    layers = cfg["layers"]
    diffs_dir = outputs_base / "diffs"
    geometry_dir = outputs_base / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    attack_names = select_attacks(cfg)
    rows: List[Dict] = []

    for attack_name in attack_names:
        npz_path = diffs_dir / attack_name / "diffs.npz"
        if not npz_path.exists():
            print(f"  [SKIP] {attack_name}: no diffs.npz (run scripts/01_extract_diffs.py first).")
            continue
        data = np.load(npz_path, allow_pickle=True)
        tags = data["tags"]

        for L in layers:
            diffs = data[f"diff_layer{L}"]
            norms = np.linalg.norm(diffs, axis=1)
            for tag in ALL_TAGS:
                mask = tags == tag
                n = int(mask.sum())
                if n == 0:
                    continue
                tag_norms = norms[mask]
                rows.append({
                    "attack_name": attack_name, "layer": L, "position_tag": tag,
                    "mean_diff_norm": float(tag_norms.mean()),
                    "std_diff_norm": float(tag_norms.std()),
                    "n_positions": n,
                })
        print(f"  [OK] {attack_name}")

    out_path = geometry_dir / "position_concentration.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["attack_name", "layer", "position_tag", "mean_diff_norm", "std_diff_norm", "n_positions"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[04_position_concentration] Wrote {len(rows)} rows -> {out_path}")
    print(f"[04_position_concentration] Next: python scripts/05_make_plots.py --config {args.config}")


if __name__ == "__main__":
    main()
