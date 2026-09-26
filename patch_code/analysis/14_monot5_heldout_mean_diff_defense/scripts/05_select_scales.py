#!/usr/bin/env python3
"""
scripts/05_select_scales.py
==============================
Per candidate (decoder head, or encoder head x position-mask condition),
select the scale with the highest MEAN normalized recovery on VALIDATION
data; ties broken by the smaller scale (task spec section 8). Test data is
never touched here.

Also joins the validation recovery curve against the validation
clean-damage curve (per scale) so the recovery-vs-clean-damage trade-off
can be inspected explicitly, without baking any clean-damage threshold
into the selection rule itself (task spec section 8, explicit instruction).

Outputs:
  outputs/selected_scales.json   {candidate_key: {selected_scale, mean_recovery, ...}}
  outputs/selected_scales.csv
  outputs/validation_recovery_vs_damage_curve.csv
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

from exp14lib.run_utils import is_already_successful, load_config, resolve_cfg_path, write_status  # noqa: E402
from exp14lib.scale_selection import CANDIDATE_COLS, select_scales  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select per-candidate scale from validation recovery.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    selection_dir = outputs_base / "scale_selection"

    required = ["selected_scales.json", "selected_scales.csv", "validation_recovery_vs_damage_curve.csv"]
    if not args.force and is_already_successful(selection_dir, required):
        print("[05_select_scales] RESUME: already done — skipping.")
        return

    val = pd.read_csv(outputs_base / "validation" / "validation_results.csv")
    clean = pd.read_csv(outputs_base / "validation" / "validation_clean_damage.csv")
    clean["abs_change"] = (clean["score_clean_defended"] - clean["score_clean"]).abs()

    status = {"status": "failed"}
    try:
        # --- mean abs clean-score change per (candidate, scale), for the inspection curve ---
        damage_by_scale = (
            clean.groupby(CANDIDATE_COLS + ["scale"])["abs_change"]
            .mean()
            .reset_index()
            .rename(columns={"abs_change": "mean_abs_clean_change"})
        )
        recov_by_scale = (
            val.groupby(CANDIDATE_COLS + ["scale"])["recovery"]
            .mean()
            .reset_index()
            .rename(columns={"recovery": "mean_recovery"})
        )
        curve = recov_by_scale.merge(damage_by_scale, on=CANDIDATE_COLS + ["scale"], how="outer")
        selection_dir.mkdir(parents=True, exist_ok=True)
        curve.to_csv(selection_dir / "validation_recovery_vs_damage_curve.csv", index=False)

        # --- selection: highest mean_recovery, ties -> smaller scale (exp14lib.scale_selection) ---
        selected_map = select_scales(val)
        selected_rows = list(selected_map.values())

        with open(selection_dir / "selected_scales.json", "w", encoding="utf-8") as fh:
            json.dump(selected_map, fh, indent=2)
        pd.DataFrame(selected_rows).to_csv(selection_dir / "selected_scales.csv", index=False)

        status.update({
            "n_candidates_selected": len(selected_rows),
            "n_curve_rows": len(curve),
            "status": "success",
        })
        print(f"[05_select_scales] selected a scale for {len(selected_rows)} candidates "
              f"-> {selection_dir / 'selected_scales.json'}")
    finally:
        write_status(selection_dir, status)

    if status["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
