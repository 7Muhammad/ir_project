#!/usr/bin/env python3
"""
scripts/05_sanity_check_outputs.py
====================================
Stage 05: Read output files and print human-readable diagnostics.

WHAT THIS SCRIPT DOES
----------------------
After running the experiment (or smoke test), this script reads the CSV and
JSONL files produced by Stages 01–03 and prints statistics that let you
quickly judge whether the results look reasonable.

It also prints warnings when something looks suspicious:
  - No examples were selected (Stage 01 filter too strict).
  - All patching effects are NaN (encoding or hook bug).
  - All patching effects are approximately zero (model was not affected).
  - Attack barely outscores control (attack may not be working on this data).

This script is NOT part of the experiment itself.  It does not modify any files.
It is safe to run multiple times.

Usage:
    python scripts/05_sanity_check_outputs.py --config configs/default.yaml
or via:
    bash bash/run_sanity_check.sh configs/default.yaml

Run with:
    conda activate advseq2seq
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import List, Optional

# Make the project src/ importable
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> Optional[float]:
    """Return the mean of a list of floats, ignoring NaNs.  None if empty."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def pct_true(flags: List[bool]) -> float:
    """Return the percentage of True values in a boolean list."""
    if not flags:
        return 0.0
    return 100.0 * sum(flags) / len(flags)


def fmt(v: Optional[float], decimals: int = 4) -> str:
    """Format a float, or return 'N/A' if None."""
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def check_warn(condition: bool, message: str) -> bool:
    """Print a warning if condition is True.  Returns True if warning was printed."""
    if condition:
        print(f"  WARNING: {message}")
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 05: Print sanity-check diagnostics for experiment outputs."
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()

    if not cfg_path.exists():
        sys.exit(f"ERROR: Config not found: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    base_dir_str = cfg.get("outputs", {}).get("base_dir", "outputs")
    base_dir = PROJECT_ROOT / base_dir_str

    # Known output paths
    pairs_path    = base_dir / "pairs" / "pairs.jsonl"
    scores_path   = base_dir / "scores" / "all_scores.csv"
    selected_path = base_dir / "scores" / "selected_examples.jsonl"
    agg_path      = base_dir / "patching" / "layer_patching_results.csv"

    any_warnings = False

    print("=" * 60)
    print(f"  Sanity Check: {base_dir_str}")
    print(f"  Config: {cfg_path.name}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Section 1: Pairs (Stage 00 output)
    # -------------------------------------------------------------------------
    print("\n--- Pairs (Stage 00) ---")
    if not pairs_path.exists():
        print(f"  MISSING: {pairs_path}")
        print("  Run Stage 00 first: bash bash/run_prepare_pairs.sh")
    else:
        pairs = []
        with open(pairs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        print(f"  Number of pairs: {len(pairs)}")
        has_attacked = sum(1 for p in pairs if "attacked_passage" in p)
        print(f"  Pairs with attacked_passage: {has_attacked}")
        if has_attacked < len(pairs):
            print(
                f"  NOTE: {len(pairs) - has_attacked} pairs lack 'attacked_passage'. "
                "Stage 01 will fail for those."
            )

    # -------------------------------------------------------------------------
    # Section 2: Scores (Stage 01 output)
    # -------------------------------------------------------------------------
    print("\n--- Scores (Stage 01) ---")
    if not scores_path.exists():
        print(f"  MISSING: {scores_path}")
        print("  Run Stage 01 first: bash bash/run_score_and_select.sh")
    else:
        original_scores = []
        control_scores  = []
        attack_scores   = []
        attack_gt_control = []

        with open(scores_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    o = float(row["original_score"])
                    c = float(row["control_score"])
                    a = float(row["attack_score"])
                    original_scores.append(o)
                    control_scores.append(c)
                    attack_scores.append(a)
                    attack_gt_control.append(a > c)
                except (KeyError, ValueError):
                    continue

        print(f"  Number of scored examples: {len(original_scores)}")
        print(f"  mean(original_score):          {fmt(mean(original_scores))}")
        print(f"  mean(control_score):           {fmt(mean(control_scores))}")
        print(f"  mean(attack_score):            {fmt(mean(attack_scores))}")

        ctrl_vs_orig = [c - o for c, o in zip(control_scores, original_scores)]
        atk_vs_orig  = [a - o for a, o in zip(attack_scores, original_scores)]
        atk_vs_ctrl  = [a - c for a, c in zip(attack_scores, control_scores)]

        print(f"  mean(control_score - original_score):  {fmt(mean(ctrl_vs_orig))}")
        print(f"  mean(attack_score  - original_score):  {fmt(mean(atk_vs_orig))}")
        print(f"  mean(attack_score  - control_score):   {fmt(mean(atk_vs_ctrl))}")

        pct_atk_wins = pct_true(attack_gt_control)
        print(f"  attack_score > control_score: {pct_atk_wins:.1f}% of examples")

        # Warnings
        any_warnings |= check_warn(
            pct_atk_wins < 50.0,
            f"Attack outscores control in only {pct_atk_wins:.1f}% of examples. "
            "The attack may not be effective on this data subset, or the wrong "
            "attacked_passage was loaded.",
        )

    # -------------------------------------------------------------------------
    # Section 3: Selected examples (Stage 01 output)
    # -------------------------------------------------------------------------
    print("\n--- Selected Examples (Stage 01) ---")
    if not selected_path.exists():
        print(f"  MISSING: {selected_path}")
    else:
        selected = []
        with open(selected_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    selected.append(json.loads(line))

        print(f"  Number of selected examples: {len(selected)}")

        any_warnings |= check_warn(
            len(selected) == 0,
            "selected_examples.jsonl is empty. "
            "Either no pairs passed the min_attack_delta filter (try lowering it "
            "in the config), or the scoring stage did not produce any attack-score > "
            "control-score pairs. In the smoke config, min_attack_delta is -999.",
        )

        if selected:
            sel_deltas = [
                ex.get("attack_delta_vs_control", float("nan")) for ex in selected
            ]
            print(f"  mean(attack_delta_vs_control) for selected: {fmt(mean(sel_deltas))}")

    # -------------------------------------------------------------------------
    # Section 4: Patching results (Stage 03 output)
    # -------------------------------------------------------------------------
    print("\n--- Patching Results (Stage 03) ---")
    if not agg_path.exists():
        print(f"  MISSING: {agg_path}")
        print("  Run Stage 03 first: bash bash/run_layer_patching.sh")
    else:
        forward_effects:  defaultdict = defaultdict(list)
        reverse_effects:  defaultdict = defaultdict(list)
        all_effects: list = []

        with open(agg_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    effect    = float(row["mean_effect"])
                    component = row["component"]
                    direction = row["direction"]
                    all_effects.append(effect)
                    if direction == "forward":
                        forward_effects[component].append(effect)
                    elif direction == "reverse":
                        reverse_effects[component].append(effect)
                except (KeyError, ValueError):
                    continue

        print(f"  Number of aggregated patching rows: {len(all_effects)}")

        # Detect degenerate outputs
        non_nan = [e for e in all_effects if not math.isnan(e)]
        all_nan = len(non_nan) == 0
        all_zero = len(non_nan) > 0 and all(abs(e) < 1e-6 for e in non_nan)

        any_warnings |= check_warn(
            all_nan,
            "All patching effects are NaN. This suggests an encoding or hook bug. "
            "Check that Stage 02 cached activations successfully.",
        )
        any_warnings |= check_warn(
            all_zero,
            "All patching effects are approximately zero. "
            "Either no examples were selected, or the patching mechanics are not "
            "transmitting the activation change to the final score.",
        )

        # Mean forward effect per component
        print("\n  Mean FORWARD effect by component (higher → carries attack signal):")
        for comp in sorted(forward_effects):
            vals = forward_effects[comp]
            print(f"    {comp:<25} mean={fmt(mean(vals))}  (n={len(vals)} layer-averages)")

        # Mean reverse effect per component
        print("\n  Mean REVERSE effect by component (higher → necessary for attack):")
        for comp in sorted(reverse_effects):
            vals = reverse_effects[comp]
            print(f"    {comp:<25} mean={fmt(mean(vals))}  (n={len(vals)} layer-averages)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    if any_warnings:
        print("  Sanity check complete — see WARNINGS above.")
    else:
        print("  Sanity check complete — no warnings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
