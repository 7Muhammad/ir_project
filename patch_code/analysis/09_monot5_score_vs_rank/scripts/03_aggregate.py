#!/usr/bin/env python3
"""
scripts/03_aggregate.py
==========================
Per-attack aggregate table (105 rows) and the global Spearman correlation
(the primary reported number — per-config n is small and noisy, so the
pooled-across-all-examples value is the headline metric).

Output:
  outputs/aggregate/per_attack_summary.csv
    attack_name, mean_delta_score, mean_delta_rank, success_rate,
    spearman_per_attack, n
  outputs/aggregate/global_spearman.json
    {"spearman_global": rho, "n": N}
"""

from __future__ import annotations

import argparse
import json
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

from exp9lib.metrics import spearman_correlation  # noqa: E402
from exp9lib.run_utils import load_config, resolve_cfg_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-example score/rank rows.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])

    per_example_dir = outputs_base / "per_example"
    csvs = sorted(per_example_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"No per-example CSVs in {per_example_dir} — run scripts/02_compute_score_rank.py first.")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    print(f"[03_aggregate] loaded {len(df)} rows from {len(csvs)} attack CSV(s)")

    rows = []
    for attack_name, group in df.groupby("attack_name"):
        rows.append({
            "attack_name": attack_name,
            "mean_delta_score": group["delta_score"].mean(),
            "mean_delta_rank": group["delta_rank"].mean(),
            "success_rate": group["success"].mean(),
            "spearman_per_attack": spearman_correlation(
                group["delta_score"].tolist(), group["delta_rank"].tolist()
            ),
            "n": len(group),
        })
    summary = pd.DataFrame(rows).sort_values("attack_name").reset_index(drop=True)

    agg_dir = outputs_base / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    summary_path = agg_dir / "per_attack_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[03_aggregate] wrote {len(summary)} per-attack rows -> {summary_path}")

    rho_global = spearman_correlation(df["delta_score"].tolist(), df["delta_rank"].tolist())
    global_path = agg_dir / "global_spearman.json"
    with open(global_path, "w", encoding="utf-8") as fh:
        json.dump({"spearman_global": rho_global, "n": len(df)}, fh, indent=2)
    print(f"[03_aggregate] global spearman rho={rho_global:.4f} (n={len(df)}) -> {global_path}")
    print(f"\n  Next: python scripts/04_make_plot.py --config {args.config}")


if __name__ == "__main__":
    main()
