#!/usr/bin/env python3
"""
scripts/08_aggregate.py
==========================
Consolidate the IID (script 06) and attack-OOD (script 07) results into
final reporting tables. Every output row identifies: side, component,
layer, head_idx, condition (encoder position mask, or "decoder"),
protocol ("iid" | "ood_token" | "ood_position" | "ood_repetitions"),
scale_role, selected scale, n examples, recovery metrics, and (for the
clean-damage tables) clean-damage metrics — task spec section 14.

Also derives, for every encoder head, its single BEST validation-selected
position condition (by mean_recovery_at_selected_scale) — used by Figure A
(task spec section 15: "encoder result should use each head's best
validation-selected position condition").

Outputs:
  outputs/full_recovery_summary.csv
  outputs/full_clean_damage_summary.csv
  outputs/encoder_best_condition.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd  # noqa: E402

from exp14lib.run_utils import load_config, resolve_cfg_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate IID + OOD results into final reporting tables.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def _read_csv_or_empty(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])

    iid_summary = _read_csv_or_empty(outputs_base / "test" / "iid_summary.csv")
    if len(iid_summary):
        iid_summary["protocol"] = "iid"
    ood_summary = _read_csv_or_empty(outputs_base / "ood_summary.csv")
    if len(ood_summary):
        ood_summary["protocol"] = "ood_" + ood_summary["ood_factor"].astype(str)
        ood_summary["scale_role"] = ""  # OOD rows are already restricted to selected/reference scales per fold
        ood_summary = ood_summary.drop(columns=["ood_factor"])

    selected_scales = _read_csv_or_empty(outputs_base / "scale_selection" / "selected_scales.csv")

    recovery_summary = pd.concat([iid_summary, ood_summary], ignore_index=True, sort=False)
    if len(recovery_summary) and len(selected_scales):
        recovery_summary = recovery_summary.merge(
            selected_scales[["side", "layer", "head_idx", "condition", "selected_scale",
                              "mean_recovery_at_selected_scale"]],
            on=["side", "layer", "head_idx", "condition"], how="left",
        )
    recovery_summary.to_csv(outputs_base / "full_recovery_summary.csv", index=False)

    iid_clean = _read_csv_or_empty(outputs_base / "test" / "clean_damage_summary.csv")
    if len(iid_clean):
        iid_clean["protocol"] = "iid"
    ood_clean = _read_csv_or_empty(outputs_base / "ood_clean_damage_summary.csv")
    if len(ood_clean):
        ood_clean["protocol"] = "ood_" + ood_clean["ood_factor"].astype(str)
        ood_clean["scale_role"] = ""
        ood_clean = ood_clean.drop(columns=["ood_factor"])
    clean_summary = pd.concat([iid_clean, ood_clean], ignore_index=True, sort=False)
    clean_summary.to_csv(outputs_base / "full_clean_damage_summary.csv", index=False)

    # --- Encoder best position condition (Figure A) ---
    best_rows = []
    if len(selected_scales):
        enc = selected_scales[selected_scales["side"] == "encoder"]
        for (layer, head_idx), group in enc.groupby(["layer", "head_idx"]):
            best = group.sort_values("mean_recovery_at_selected_scale", ascending=False).iloc[0]
            best_rows.append({
                "layer": int(layer), "head_idx": int(head_idx),
                "best_condition": best["condition"], "selected_scale": best["selected_scale"],
                "mean_recovery_at_selected_scale": best["mean_recovery_at_selected_scale"],
            })
    pd.DataFrame(best_rows).to_csv(outputs_base / "encoder_best_condition.csv", index=False)

    print(f"[08_aggregate] full_recovery_summary.csv: {len(recovery_summary)} rows")
    print(f"[08_aggregate] full_clean_damage_summary.csv: {len(clean_summary)} rows")
    print(f"[08_aggregate] encoder_best_condition.csv: {len(best_rows)} encoder heads")


if __name__ == "__main__":
    main()
