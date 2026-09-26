"""
src/comparison.py
=================
Cross-attack aggregation and plotting for the DecoderLens rank experiment.

Reads every attack's ``ranks/layerwise_rank_summary.csv`` (+ status.json and
sanity JSON) and produces:

    attack_comparison/decoderlens_summary.csv
    attack_comparison/rank_gain_heatmap.png
    attack_comparison/score_delta_heatmap.png
    attack_comparison/success_rate_heatmap.png
    attack_comparison/top_attacks_rank_gain.png
    attack_comparison/top_attacks_score_delta.png
    attack_comparison/layer_of_first_positive_effect.png
    attack_comparison/mean_rank_gain_trajectory.png
    attack_comparison/mean_score_delta_trajectory.png
    attack_comparison/attack_vs_control_rank_trajectory.png

Heatmaps: rows = attack_name, columns = encoder layer 0..12.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-attack summary row
# ---------------------------------------------------------------------------

def _first_positive_layer(sub: pd.DataFrame, column: str) -> Optional[int]:
    """First layer_index (ascending) where ``column`` > 0, else None."""
    pos = sub[sub[column] > 0].sort_values("layer_index")
    if pos.empty:
        return None
    return int(pos.iloc[0]["layer_index"])


def build_summary_row(
    attack_summary: pd.DataFrame,
    status: Dict,
    sanity: Optional[Dict],
) -> Dict:
    """
    Build one ``decoderlens_summary.csv`` row from an attack's rank summary.

    ``attack_summary`` is the per-attack layerwise_rank_summary.csv loaded as a
    DataFrame; we use its ``attack`` (variant == "attack") rows.
    """
    atk = attack_summary[attack_summary["variant"] == "attack"].sort_values("layer_index")
    final_layer = int(attack_summary["layer_index"].max())
    final = atk[atk["layer_index"] == final_layer]

    def _final(col: str) -> Optional[float]:
        if final.empty or col not in final:
            return None
        return float(final.iloc[0][col])

    # Best layer by score delta / rank gain.
    best_score_layer = best_score_val = None
    best_gain_layer = best_gain_val = None
    if not atk.empty:
        i_sd = atk["mean_score_delta_vs_control"].idxmax()
        best_score_layer = int(atk.loc[i_sd, "layer_index"])
        best_score_val = float(atk.loc[i_sd, "mean_score_delta_vs_control"])
        i_rg = atk["median_rank_gain_vs_control"].idxmax()
        best_gain_layer = int(atk.loc[i_rg, "layer_index"])
        best_gain_val = float(atk.loc[i_rg, "median_rank_gain_vs_control"])

    row = {
        "attack_name":  status.get("attack_name"),
        "token":        status.get("token"),
        "position":     status.get("position"),
        "repetitions":  status.get("repetitions"),
        "n_examples":   status.get("n_examples_used"),
        "n_alignment_success": status.get("n_alignment_success"),
        "n_alignment_failed":  status.get("n_alignment_failed"),
        "candidate_top_k":     status.get("candidate_top_k"),
        "final_layer_mean_score_delta":   _final("mean_score_delta_vs_control"),
        "final_layer_median_score_delta": _final("median_score_delta_vs_control"),
        "final_layer_mean_rank_gain":     _final("mean_rank_gain_vs_control"),
        "final_layer_median_rank_gain":   _final("median_rank_gain_vs_control"),
        "final_layer_success_rate_rank":  _final("success_rate_vs_control"),
        "final_layer_success_rate_score": _final("score_success_rate_vs_control"),
        "best_layer_by_score_delta":  best_score_layer,
        "best_layer_score_delta":     best_score_val,
        "best_layer_by_rank_gain":    best_gain_layer,
        "best_layer_rank_gain":       best_gain_val,
        "first_layer_positive_score_delta": _first_positive_layer(atk, "mean_score_delta_vs_control"),
        "first_layer_positive_rank_gain":   _first_positive_layer(atk, "median_rank_gain_vs_control"),
        "final_layer_sanity_passed": (sanity or {}).get("passed"),
    }
    return row


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _pivot(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot attack × layer matrix of ``value_col`` (attack rows from each summary)."""
    atk = long_df[long_df["variant"] == "attack"]
    mat = atk.pivot_table(
        index="attack_name", columns="layer_index", values=value_col, aggfunc="mean"
    )
    return mat.sort_index()


def plot_heatmap(
    mat: pd.DataFrame,
    output_path: pathlib.Path,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
) -> None:
    """Render an attack × layer heatmap."""
    if mat.empty:
        print(f"[comparison] Skipping empty heatmap: {output_path}")
        return
    n_rows = mat.shape[0]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * n_rows + 1.5)))

    data = mat.values.astype(float)
    if center_zero:
        vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xlabel("encoder layer", fontsize=12)
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[comparison] Saved: {output_path}")


def plot_top_attacks_bar(
    summary_df: pd.DataFrame,
    value_col: str,
    output_path: pathlib.Path,
    title: str,
    xlabel: str,
    top_n: int,
) -> None:
    """Horizontal bar chart of the top-N attacks by a final-layer metric."""
    df = summary_df.dropna(subset=[value_col]).copy()
    if df.empty:
        print(f"[comparison] Skipping empty bar plot: {output_path}")
        return
    df = df.sort_values(value_col, ascending=False).head(top_n)
    df = df.iloc[::-1]  # largest at top
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df) + 1.5)))
    colors = ["#C44E52" if v >= 0 else "#4878CF" for v in df[value_col]]
    ax.barh(df["attack_name"], df[value_col], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[comparison] Saved: {output_path}")


def compute_mean_trajectory(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Per-layer aggregate of ``value_col`` across attacks (one row per attack
    per layer already; this averages/medians those attack-level values
    across all attacks at each encoder depth).
    """
    atk = long_df[long_df["variant"] == "attack"]
    grp = atk.groupby("layer_index")[value_col].agg(["mean", "median", "std", "count"])
    return grp.sort_index()


def plot_mean_trajectory(
    long_df: pd.DataFrame,
    output_path: pathlib.Path,
    value_col: str,
    title: str,
    ylabel: str,
) -> None:
    """
    Line plot: x = encoder layer (0..12), y = mean/median of ``value_col``
    averaged across all attacks at that layer, with a ±1 std band.
    """
    traj = compute_mean_trajectory(long_df, value_col)
    if traj.empty:
        print(f"[comparison] Skipping empty trajectory plot: {output_path}")
        return

    x = traj.index.to_numpy()
    mean = traj["mean"].to_numpy()
    median = traj["median"].to_numpy()
    std = traj["std"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, mean, marker="o", color="#4878CF", label="mean across attacks")
    ax.fill_between(x, mean - std, mean + std, color="#4878CF", alpha=0.15,
                     label="±1 std across attacks")
    ax.plot(x, median, marker="s", linestyle="--", color="#C44E52",
             label="median across attacks")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("encoder layer (0 = embeddings, 12 = final)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[comparison] Saved: {output_path}")


def plot_attack_vs_control_trajectory(
    long_df: pd.DataFrame,
    output_path: pathlib.Path,
    value_col: str = "mean_rank",
    title: str = "Attack vs. padded-control rank, averaged across all attacks, by encoder layer",
    ylabel: str = "mean rank (1 = best) within top-100 candidates",
) -> None:
    """
    Two-line plot: x = encoder layer, y = ``value_col`` averaged across all
    attacks, one line for the ``attack`` variant and one for the
    ``padded_control`` variant — shows the two raw trajectories whose gap
    is the rank_gain_vs_control metric plotted elsewhere.
    """
    variants = {"attack": ("#C44E52", "o", "-"), "padded_control": ("#4878CF", "s", "--")}
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for variant, (color, marker, ls) in variants.items():
        sub = long_df[long_df["variant"] == variant]
        if sub.empty:
            continue
        traj = sub.groupby("layer_index")[value_col].mean().sort_index()
        if traj.empty:
            continue
        any_data = True
        label = "attack" if variant == "attack" else "padded control"
        ax.plot(traj.index, traj.values, marker=marker, linestyle=ls, color=color, label=label)

    if not any_data:
        print(f"[comparison] Skipping empty attack-vs-control plot: {output_path}")
        plt.close(fig)
        return

    ax.set_xlabel("encoder layer (0 = embeddings, 12 = final)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()  # rank 1 (best) at top
    all_layers = sorted(long_df["layer_index"].unique())
    ax.set_xticks(all_layers)
    ax.legend(fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[comparison] Saved: {output_path}")


def plot_layer_of_first_positive_effect(
    summary_df: pd.DataFrame,
    output_path: pathlib.Path,
    column: str = "first_layer_positive_score_delta",
    title: str = "First encoder layer with a positive attack effect",
) -> None:
    """Bar chart of the first layer each attack becomes positive (NaN omitted)."""
    df = summary_df.dropna(subset=[column]).copy()
    if df.empty:
        print(f"[comparison] Skipping empty plot: {output_path}")
        return
    df = df.sort_values(column)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df) + 1.5)))
    ax.barh(df["attack_name"], df[column], color="#55A868")
    ax.set_xlabel("first encoder layer with positive effect", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[comparison] Saved: {output_path}")
