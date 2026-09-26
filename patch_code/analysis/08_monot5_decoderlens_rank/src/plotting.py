"""
src/plotting.py
===============
Per-attack DecoderLens plots.

All plots share the x-axis = encoder layer index (0..12).  The summary
DataFrame passed in is ``ranks/layerwise_rank_summary.csv`` (aggregated by
attack, variant, and layer).

Plots
-----
1. rank_over_layers.png        — median rank per variant; y-axis inverted.
2. rank_gain_over_layers.png   — median(control_rank - attack_rank); 0-line.
3. score_delta_over_layers.png — mean(attack_score - control_score); 0-line.
4. success_rate_over_layers.png— fraction of examples where attack_rank < control_rank.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / server runs
import matplotlib.pyplot as plt
import pandas as pd

VARIANT_STYLE = {
    "original":       {"color": "#888888", "marker": "o", "label": "original"},
    "padded_control": {"color": "#4878CF", "marker": "s", "label": "padded_control"},
    "attack":         {"color": "#C44E52", "marker": "^", "label": "attack"},
}


def _layer_axis(ax, summary: pd.DataFrame) -> None:
    layers = sorted(summary["layer_index"].unique())
    ax.set_xticks(layers)
    ax.set_xlabel("encoder layer", fontsize=12)


def plot_rank_over_layers(
    summary: pd.DataFrame, output_path: pathlib.Path, attack_name: str
) -> None:
    """Median rank of the target passage per variant across encoder layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ("original", "padded_control", "attack"):
        sub = summary[summary["variant"] == variant].sort_values("layer_index")
        if sub.empty:
            continue
        st = VARIANT_STYLE[variant]
        ax.plot(
            sub["layer_index"], sub["median_rank"],
            color=st["color"], marker=st["marker"], label=st["label"], linewidth=2,
        )
    _layer_axis(ax, summary)
    ax.set_ylabel("median rank of target passage", fontsize=12)
    ax.invert_yaxis()  # rank 1 (best) at the top
    ax.set_title(f"Rank over encoder layers — {attack_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved: {output_path}")


def plot_rank_gain_over_layers(
    summary: pd.DataFrame, output_path: pathlib.Path, attack_name: str
) -> None:
    """Median(control_rank - attack_rank): >0 means the attack improved rank."""
    sub = summary[summary["variant"] == "attack"].sort_values("layer_index")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not sub.empty:
        ax.plot(
            sub["layer_index"], sub["median_rank_gain_vs_control"],
            color="#C44E52", marker="^", linewidth=2, label="median rank gain",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    _layer_axis(ax, summary)
    ax.set_ylabel("median(control_rank − attack_rank)", fontsize=12)
    ax.set_title(
        f"Rank gain over encoder layers — {attack_name}\n"
        "above 0: attack improves rank   |   below 0: attack hurts rank",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved: {output_path}")


def plot_score_delta_over_layers(
    summary: pd.DataFrame, output_path: pathlib.Path, attack_name: str
) -> None:
    """Mean(attack_score - padded_control_score) across encoder layers."""
    sub = summary[summary["variant"] == "attack"].sort_values("layer_index")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not sub.empty:
        ax.plot(
            sub["layer_index"], sub["mean_score_delta_vs_control"],
            color="#4878CF", marker="o", linewidth=2, label="mean score delta",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    _layer_axis(ax, summary)
    ax.set_ylabel("mean(attack_score − padded_control_score)", fontsize=12)
    ax.set_title(f"Score delta over encoder layers — {attack_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved: {output_path}")


def plot_success_rate_over_layers(
    summary: pd.DataFrame, output_path: pathlib.Path, attack_name: str
) -> None:
    """Fraction of examples where attack_rank < control_rank, per layer."""
    sub = summary[summary["variant"] == "attack"].sort_values("layer_index")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not sub.empty:
        ax.plot(
            sub["layer_index"], sub["success_rate_vs_control"],
            color="#55A868", marker="D", linewidth=2, label="success rate",
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    _layer_axis(ax, summary)
    ax.set_ylabel("fraction with attack_rank < control_rank", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Attack success rate over encoder layers — {attack_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved: {output_path}")


def plot_all_per_attack(
    summary: pd.DataFrame, plots_dir: pathlib.Path, attack_name: str
) -> None:
    """Produce all four per-attack plots."""
    plot_rank_over_layers(summary, plots_dir / "rank_over_layers.png", attack_name)
    plot_rank_gain_over_layers(summary, plots_dir / "rank_gain_over_layers.png", attack_name)
    plot_score_delta_over_layers(summary, plots_dir / "score_delta_over_layers.png", attack_name)
    plot_success_rate_over_layers(summary, plots_dir / "success_rate_over_layers.png", attack_name)
