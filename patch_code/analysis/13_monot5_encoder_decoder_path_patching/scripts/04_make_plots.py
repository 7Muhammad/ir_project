#!/usr/bin/env python3
"""
scripts/04_make_plots.py
==========================
Main plots for Experiment 13, reading only from outputs/aggregates/
(written by scripts/03_aggregate.py). Architectural ordering (the order
senders/receivers appear in configs/heads/*.json, by descending
importance from Experiments 11/3) is used for rows/columns throughout --
NOT a value-sorted order, which would destroy the layer/head structure.
A secondary clustered version of the main heatmap is saved to
outputs/diagnostics/ for exploration only.

Plot 1: main combined heatmap        outputs/plots/path_heatmap_combined.png
Plot 2: forward matrix               outputs/plots/path_heatmap_forward.png
Plot 3: reverse matrix               outputs/plots/path_heatmap_reverse.png
Plot 4: stability (fraction of attacks positive) outputs/plots/path_stability.png
Plot 5 (optional): sparse top-K circuit diagram   outputs/plots/sparse_circuit_topk.png
Diagnostic: clustered version of Plot 1           outputs/diagnostics/path_heatmap_combined_clustered.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

EXP_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(EXP_DIR))

from exp13lib.head_lists import load_receivers, load_senders  # noqa: E402
from exp13lib.run_utils import load_config, resolve_cfg_path  # noqa: E402

TOP_K_SPARSE = 25  # visualization-only cutoff for Plot 5 -- NOT an importance threshold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 13 -- plots.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def plot_matrix(matrix: pd.DataFrame, title: str, cbar_label: str, out_path: pathlib.Path,
                 diverging: bool = True, vlim=None) -> None:
    values = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if diverging:
        finite = values[np.isfinite(values)]
        vmax = max(float(np.nanmax(np.abs(finite))), 1e-6) if finite.size else 1e-6
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap = "RdBu_r"
        im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    else:
        vmin, vmax = vlim or (0.0, 1.0)
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("decoder receiver (cross-attention head)")
    ax.set_ylabel("encoder sender (self-attention head)")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_clustered(matrix: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Diagnostic-only: rows/columns reordered by hierarchical similarity."""
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
    except ImportError:
        print("  SKIP clustered diagnostic plot -- scipy not available.")
        return
    vals = matrix.fillna(0.0).values
    row_order = leaves_list(linkage(vals, method="average")) if vals.shape[0] > 1 else list(range(vals.shape[0]))
    col_order = leaves_list(linkage(vals.T, method="average")) if vals.shape[1] > 1 else list(range(vals.shape[1]))
    clustered = matrix.iloc[row_order, col_order]
    plot_matrix(clustered, "Path-patching heatmap (clustered, diagnostic only)",
                "mean path_combined (attack-balanced)", out_path)


def plot_sparse_circuit(global_df: pd.DataFrame, senders, receivers, out_path: pathlib.Path, top_k: int) -> None:
    top = global_df.reindex(global_df["attack_balanced_mean_combined"].abs().sort_values(ascending=False).index)
    top = top.head(top_k)

    sender_order = [s.label for s in senders]
    receiver_order = [r.label for r in receivers]
    used_senders = [s for s in sender_order if s in set(top["sender_name"])]
    used_receivers = [r for r in receiver_order if r in set(top["receiver_name"])]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * max(len(used_senders), len(used_receivers)))))
    y_send = {name: i for i, name in enumerate(used_senders)}
    y_recv = {name: i for i, name in enumerate(used_receivers)}

    for _, row in top.iterrows():
        y0 = y_send[row["sender_name"]]
        y1 = y_recv[row["receiver_name"]]
        val = row["attack_balanced_mean_combined"]
        color = "#c0392b" if val < 0 else "#2166ac"
        ax.plot([0, 1], [y0, y1], color=color, alpha=min(1.0, 0.25 + abs(val) * 4), linewidth=max(0.5, abs(val) * 25))

    ax.scatter([0] * len(used_senders), range(len(used_senders)), s=40, color="#333333", zorder=3)
    ax.scatter([1] * len(used_receivers), range(len(used_receivers)), s=40, color="#333333", zorder=3)
    for name, y in y_send.items():
        ax.text(-0.03, y, name, ha="right", va="center", fontsize=7)
    for name, y in y_recv.items():
        ax.text(1.03, y, name, ha="left", va="center", fontsize=7)

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Sparse encoder->decoder path diagram\n"
                 f"(top {top_k} |mean path_combined| edges -- VISUALIZATION-ONLY cutoff, not an importance threshold)",
                 fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    agg_dir = resolve_cfg_path(cfg, cfg["outputs"]["aggregates_dir"])
    plots_dir = resolve_cfg_path(cfg, cfg["outputs"]["plots_dir"])
    diag_dir = resolve_cfg_path(cfg, cfg["outputs"]["diagnostics_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    senders = load_senders()
    receivers = load_receivers()

    combined = pd.read_csv(agg_dir / "global_matrix_combined.csv", index_col=0)
    forward = pd.read_csv(agg_dir / "global_matrix_forward.csv", index_col=0)
    reverse = pd.read_csv(agg_dir / "global_matrix_reverse.csv", index_col=0)
    stability = pd.read_csv(agg_dir / "global_matrix_stability.csv", index_col=0)
    global_df = pd.read_csv(agg_dir / "global_sender_receiver_long.csv")

    plot_matrix(combined, "Encoder sender -> decoder receiver path effect (attack-balanced mean path_combined)",
                "mean path_combined", plots_dir / "path_heatmap_combined.png")
    plot_matrix(forward, "Forward path effect (attack-balanced mean path_forward)",
                "mean path_forward", plots_dir / "path_heatmap_forward.png")
    plot_matrix(reverse, "Reverse path effect (attack-balanced mean path_reverse)",
                "mean path_reverse", plots_dir / "path_heatmap_reverse.png")
    plot_matrix(stability, "Path consistency across attacks\n(fraction of eligible attacks with positive mean path_combined)",
                "fraction of attacks positive", plots_dir / "path_stability.png",
                diverging=False, vlim=(0.0, 1.0))

    plot_sparse_circuit(global_df, senders, receivers, plots_dir / "sparse_circuit_topk.png", TOP_K_SPARSE)
    plot_clustered(combined, diag_dir / "path_heatmap_combined_clustered.png")

    print("[plots] done.")


if __name__ == "__main__":
    main()
