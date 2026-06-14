#!/usr/bin/env python3
"""
scripts/11_compare_attacks.py
==============================
Cross-attack comparison: collect results from all successful attack runs and
produce summary statistics and cross-attack plots.

Reads
-----
  outputs/attacks/{attack_name}/status.json
  outputs/attacks/{attack_name}/scores/all_scores.csv
  outputs/attacks/{attack_name}/patching/layer_patching_results.csv

Writes
------
  outputs/attack_comparison/summary.csv
  outputs/attack_comparison/summary_shared_examples.csv   (intersection of pairs)
  outputs/attack_comparison/attack_strength_bar.png
  outputs/attack_comparison/component_importance_heatmap.png
  outputs/attack_comparison/layer_profile_heatmap.png

Combined importance
-------------------
For cross-attack comparison we use:
    combined_clamped = min(max(forward, 0), max(reverse, 0))

This keeps only positive effects (negative means the patch moved the score
in the wrong direction — not evidence of importance) and measures the
weakest direction (both necessary AND sufficient).

Per-attack heatmaps (scripts/04_plot_results.py) still use the signed
min(forward, reverse) so they remain consistent with existing plots.

Layer groups (from config)
--------------------------
    early  = layers 0–3
    mid    = layers 4–7
    late   = layers 8–11
These defaults are for monoT5-base (12 layers). Adjust in configs/default.yaml
under comparison.layer_groups.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-attack comparison: summarise and plot multi-attack results."
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Layer group helpers
# ---------------------------------------------------------------------------

def _group_range(groups_cfg: dict, key: str) -> Tuple[int, int]:
    lo, hi = groups_cfg[key]
    return int(lo), int(hi)


def _mean_in_group(
    agg_df: pd.DataFrame,
    direction: str,
    component: str,
    lo: int,
    hi: int,
) -> float:
    """Mean (clamped) combined effect for a component across a layer range."""
    fwd = agg_df[(agg_df["direction"] == "forward") & (agg_df["component"] == component)]
    rev = agg_df[(agg_df["direction"] == "reverse") & (agg_df["component"] == component)]

    values = []
    for layer in range(lo, hi + 1):
        f_row = fwd[fwd["layer"] == layer]
        r_row = rev[rev["layer"] == layer]
        if f_row.empty or r_row.empty:
            continue
        f_val = f_row["mean_effect"].values[0]
        r_val = r_row["mean_effect"].values[0]
        combined = min(max(f_val, 0.0), max(r_val, 0.0))
        values.append(combined)
    return float(np.mean(values)) if values else float("nan")


def _mean_combined_late(
    agg_df: pd.DataFrame,
    component: str,
    late_min: int,
) -> float:
    """Mean clamped combined effect for late layers of a given component."""
    fwd = agg_df[(agg_df["direction"] == "forward") & (agg_df["component"] == component)]
    rev = agg_df[(agg_df["direction"] == "reverse") & (agg_df["component"] == component)]

    values = []
    for layer in fwd["layer"].unique():
        if layer < late_min:
            continue
        r_row = rev[rev["layer"] == layer]
        if r_row.empty:
            continue
        f_val = fwd[fwd["layer"] == layer]["mean_effect"].values[0]
        r_val = r_row["mean_effect"].values[0]
        values.append(min(max(f_val, 0.0), max(r_val, 0.0)))
    return float(np.mean(values)) if values else float("nan")


def _top_cell(agg_df: pd.DataFrame) -> Tuple[Optional[int], Optional[str], float]:
    """Find the (layer, component) with the highest combined clamped importance."""
    best_layer = None
    best_comp  = None
    best_val   = float("-inf")

    fwd_df = agg_df[agg_df["direction"] == "forward"]
    rev_df = agg_df[agg_df["direction"] == "reverse"]

    for _, frow in fwd_df.iterrows():
        layer = frow["layer"]
        comp  = frow["component"]
        rrow  = rev_df[(rev_df["layer"] == layer) & (rev_df["component"] == comp)]
        if rrow.empty:
            continue
        combined = min(max(frow["mean_effect"], 0.0), max(rrow["mean_effect"].values[0], 0.0))
        if combined > best_val:
            best_val   = combined
            best_layer = int(layer)
            best_comp  = comp

    return best_layer, best_comp, best_val if best_layer is not None else float("nan")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_successful_attacks(
    attacks_dir: pathlib.Path,
    attacks_cfg: dict,
) -> List[Dict]:
    """
    Walk attacks_dir and load data for every attack with status == 'success'.

    Returns a list of dicts with keys:
        spec_meta    — from status.json
        scores_df    — DataFrame from all_scores.csv (or None)
        patching_df  — DataFrame from layer_patching_results.csv (or None)
    """
    if not attacks_dir.is_dir():
        return []

    results = []
    for attack_dir in sorted(attacks_dir.iterdir()):
        status_path = attack_dir / "status.json"
        if not status_path.exists():
            continue

        with open(status_path) as fh:
            status = json.load(fh)

        if status.get("status") != "success":
            continue

        scores_path   = attack_dir / "scores" / "all_scores.csv"
        patching_path = attack_dir / "patching" / "layer_patching_results.csv"

        scores_df   = pd.read_csv(scores_path)   if scores_path.exists()   else None
        patching_df = pd.read_csv(patching_path) if patching_path.exists() else None

        results.append({
            "status":      status,
            "scores_df":   scores_df,
            "patching_df": patching_df,
        })

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _build_summary(
    attack_data: List[Dict],
    groups_cfg: dict,
    late_layer_min: int,
) -> pd.DataFrame:
    rows = []
    for ad in attack_data:
        status = ad["status"]
        name   = status["attack_name"]
        sdf    = ad["scores_df"]
        pdf    = ad["patching_df"]

        deltas = (
            sdf["attack_delta_vs_control"].dropna().values
            if sdf is not None and "attack_delta_vs_control" in sdf.columns
            else np.array([])
        )

        top_layer, top_comp, top_combined = _top_cell(pdf) if pdf is not None else (None, None, float("nan"))

        e_lo, e_hi = _group_range(groups_cfg, "early")
        m_lo, m_hi = _group_range(groups_cfg, "mid")
        l_lo, l_hi = _group_range(groups_cfg, "late")

        def _mg(comp, lo, hi):
            return _mean_in_group(pdf, "forward", comp, lo, hi) if pdf is not None else float("nan")

        late_enc = np.nanmean([
            _mg("encoder_self_attn", l_lo, l_hi),
            _mg("encoder_mlp", l_lo, l_hi),
        ])
        late_dec_self  = _mg("decoder_self_attn",  l_lo, l_hi)
        late_dec_cross = _mg("decoder_cross_attn", l_lo, l_hi)
        late_dec_mlp   = _mg("decoder_mlp",        l_lo, l_hi)

        rows.append({
            "attack_name":             name,
            "token":                   status.get("token", ""),
            "position":                status.get("position", ""),
            "repetitions":             status.get("repetitions", ""),
            "n_pairs_loaded":          status.get("n_pairs_loaded", 0),
            "n_scored":                status.get("n_scored", 0),
            "n_align_failed":          status.get("n_align_failed", 0),
            "n_selected":              status.get("n_selected", 0),
            "mean_attack_delta":       float(np.mean(deltas)) if len(deltas) else float("nan"),
            "median_attack_delta":     float(np.median(deltas)) if len(deltas) else float("nan"),
            "max_attack_delta":        float(np.max(deltas)) if len(deltas) else float("nan"),
            "top_layer":               top_layer,
            "top_component":           top_comp,
            "top_combined_effect":     top_combined,
            "late_encoder_mean":       late_enc,
            "late_decoder_self_attn_mean":  late_dec_self,
            "late_decoder_cross_attn_mean": late_dec_cross,
            "late_decoder_mlp_mean":        late_dec_mlp,
        })
    return pd.DataFrame(rows)


def _build_shared_summary(
    attack_data: List[Dict],
    groups_cfg: dict,
    late_layer_min: int,
) -> Optional[pd.DataFrame]:
    """
    Like _build_summary but restrict each attack's scores to the intersection
    of (qid, docid) pairs that appear in ALL successful attacks.
    Gives apples-to-apples cross-attack comparison.
    """
    all_ids = []
    for ad in attack_data:
        sdf = ad["scores_df"]
        if sdf is None or "qid" not in sdf.columns:
            return None
        ids = set(zip(sdf["qid"].astype(str), sdf["docid"].astype(str)))
        all_ids.append(ids)

    if not all_ids:
        return None

    shared = all_ids[0]
    for s in all_ids[1:]:
        shared = shared & s

    if not shared:
        print("[11_compare_attacks] WARNING: intersection of (qid, docid) is empty — skipping shared summary.")
        return None

    print(f"[11_compare_attacks] Shared example intersection: {len(shared)} pairs.")

    # Build modified attack_data with scores restricted to shared pairs
    filtered = []
    for ad in attack_data:
        sdf = ad["scores_df"]
        mask = [
            (str(q), str(d)) in shared
            for q, d in zip(sdf["qid"].astype(str), sdf["docid"].astype(str))
        ]
        ad_copy = {**ad, "scores_df": sdf[mask].copy()}
        filtered.append(ad_copy)

    return _build_summary(filtered, groups_cfg, late_layer_min)


# ---------------------------------------------------------------------------
# Cross-attack plots
# ---------------------------------------------------------------------------

def _plot_attack_strength_bar(
    summary_df: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    """Bar plot: mean_attack_delta per attack, sorted descending."""
    df = summary_df.dropna(subset=["mean_attack_delta"]).sort_values("mean_attack_delta", ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 5))
    bars = ax.bar(
        range(len(df)),
        df["mean_attack_delta"],
        color="#4878CF",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["attack_name"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("mean(attack_score − control_score)", fontsize=11)
    ax.set_title("Attack strength comparison\n(mean score increase, higher = stronger attack)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[11_compare_attacks] Saved: {output_path}")


def _plot_component_importance_heatmap(
    attack_data: List[Dict],
    groups_cfg: dict,
    output_path: pathlib.Path,
) -> None:
    """
    Heatmap: rows = attacks, columns = late-layer component groups.
    Values = mean clamped combined importance (late layers only).
    """
    l_lo, l_hi = _group_range(groups_cfg, "late")

    col_defs = [
        ("Late Enc (SA+MLP)", ["encoder_self_attn", "encoder_mlp"]),
        ("Late Dec Self-Attn", ["decoder_self_attn"]),
        ("Late Dec Cross-Attn", ["decoder_cross_attn"]),
        ("Late Dec MLP", ["decoder_mlp"]),
    ]

    rows_labels = []
    matrix_rows = []

    for ad in attack_data:
        pdf  = ad["patching_df"]
        name = ad["status"]["attack_name"]
        rows_labels.append(name)
        if pdf is None:
            matrix_rows.append([float("nan")] * len(col_defs))
            continue
        row = []
        for _, comps in col_defs:
            vals = [_mean_in_group(pdf, "combined", c, l_lo, l_hi) for c in comps]
            # Fall back to forward if combined column absent (use clamped formula)
            if all(np.isnan(v) for v in vals):
                vals = []
                for c in comps:
                    vals.append(_mean_in_group(pdf, "forward", c, l_lo, l_hi))
            row.append(float(np.nanmean(vals)))
        matrix_rows.append(row)

    matrix = np.array(matrix_rows, dtype=float)
    col_labels = [cd[0] for cd in col_defs]

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.5), max(4, len(rows_labels) * 0.55)))
    vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(vmax, 1.0))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean combined importance (clamped)", fontsize=9)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows_labels)))
    ax.set_yticklabels(rows_labels, fontsize=8)
    ax.set_title(
        "Component importance by attack (late layers)\n"
        "combined = min(max(fwd,0), max(rev,0))",
        fontsize=11,
    )

    for i in range(len(rows_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[11_compare_attacks] Saved: {output_path}")


def _plot_layer_profile_heatmap(
    attack_data: List[Dict],
    groups_cfg: dict,
    output_path: pathlib.Path,
) -> None:
    """
    Heatmap: rows = attacks, columns = encoder/decoder × early/mid/late groups.
    Values = mean clamped combined importance.
    """
    group_keys = ["early", "mid", "late"]

    col_defs = []
    for gk in group_keys:
        lo, hi = _group_range(groups_cfg, gk)
        col_defs.append((f"Enc {gk.capitalize()}", ["encoder_self_attn", "encoder_mlp"], gk))
    for gk in group_keys:
        lo, hi = _group_range(groups_cfg, gk)
        col_defs.append((f"Dec {gk.capitalize()}", ["decoder_self_attn", "decoder_cross_attn", "decoder_mlp"], gk))

    rows_labels = []
    matrix_rows = []

    for ad in attack_data:
        pdf  = ad["patching_df"]
        name = ad["status"]["attack_name"]
        rows_labels.append(name)
        if pdf is None:
            matrix_rows.append([float("nan")] * len(col_defs))
            continue
        row = []
        for _, comps, gk in col_defs:
            lo, hi = _group_range(groups_cfg, gk)
            vals = [_mean_in_group(pdf, "forward", c, lo, hi) for c in comps]
            row.append(float(np.nanmean(vals)))
        matrix_rows.append(row)

    matrix = np.array(matrix_rows, dtype=float)
    col_labels = [cd[0] for cd in col_defs]

    fig, ax = plt.subplots(figsize=(max(7, len(col_labels) * 1.2), max(4, len(rows_labels) * 0.55)))
    vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(vmax, 1.0))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean combined importance (clamped)", fontsize=9)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows_labels)))
    ax.set_yticklabels(rows_labels, fontsize=8)

    # Divider between encoder and decoder groups
    ax.axvline(len(group_keys) - 0.5, color="white", linewidth=2)

    ax.set_title(
        "Layer-group profile by attack\n"
        "mean clamped combined importance (forward used as proxy)",
        fontsize=11,
    )

    for i in range(len(rows_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[11_compare_attacks] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()
    print(f"[11_compare_attacks] Config: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    outputs_base    = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    attacks_dir     = outputs_base / "attacks"
    comparison_dir  = outputs_base / "attack_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_cfg = cfg.get("comparison", {})
    groups_cfg     = comparison_cfg.get("layer_groups", {"early": [0, 3], "mid": [4, 7], "late": [8, 11]})
    late_min       = comparison_cfg.get("late_layer_min", 8)

    # --- Load all successful attacks ---
    print("[11_compare_attacks] Loading attack results ...")
    attack_data = _load_successful_attacks(attacks_dir, cfg.get("attacks", {}))

    if not attack_data:
        sys.exit(
            f"ERROR: No successful attacks found in {attacks_dir}.\n"
            "Run scripts/10_run_multi_attack_pipeline.py first."
        )

    print(f"[11_compare_attacks] Found {len(attack_data)} successful attacks.")

    # --- Summary CSV ---
    summary_df = _build_summary(attack_data, groups_cfg, late_min)
    summary_path = comparison_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[11_compare_attacks] Summary → {summary_path}")
    print(summary_df[["attack_name", "n_selected", "mean_attack_delta",
                       "top_layer", "top_component", "top_combined_effect"]].to_string(index=False))

    # --- Shared-example summary ---
    shared_df = _build_shared_summary(attack_data, groups_cfg, late_min)
    if shared_df is not None:
        shared_path = comparison_dir / "summary_shared_examples.csv"
        shared_df.to_csv(shared_path, index=False)
        print(f"[11_compare_attacks] Shared-example summary → {shared_path}")

    # --- Plots ---
    _plot_attack_strength_bar(
        summary_df=summary_df,
        output_path=comparison_dir / "attack_strength_bar.png",
    )
    _plot_component_importance_heatmap(
        attack_data=attack_data,
        groups_cfg=groups_cfg,
        output_path=comparison_dir / "component_importance_heatmap.png",
    )
    _plot_layer_profile_heatmap(
        attack_data=attack_data,
        groups_cfg=groups_cfg,
        output_path=comparison_dir / "layer_profile_heatmap.png",
    )

    print(f"\n[11_compare_attacks] Done. Outputs in: {comparison_dir}")


if __name__ == "__main__":
    main()
