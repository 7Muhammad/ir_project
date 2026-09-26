#!/usr/bin/env python3
"""
scripts/05_make_plots.py
===========================
Experiment 10, stage 5: the four required plots, plus one supplementary
PCA scatter plot.

1. Scree plot (variance explained vs. component index, first ~20 components),
   canonical attack + representative others, one panel per layer.
2. Cross-attack cosine-similarity heatmap, one panel per layer, ordered by
   attack strength.
3. Grouped bar chart: mean diff-vector norm by position tag, grouped by layer.
4. Line plot: effective rank vs. layer, one line per attack.
5. (Supplementary) PCA scatter: diff vectors projected onto their own top-2
   components, colored by position tag, for a few representative attacks x
   all 3 layers -- a direct visual of the point cloud the scree plot and
   effective-rank numbers summarize, and of how position tags relate to the
   dominant direction.

Reads outputs from stages 1-4; recomputes only the full singular-value
spectrum/projection for the handful of representative attacks (cheap; stage
2 only stored variance_explained at k in {1,3,10}, not the full spectrum or
the projected coordinates).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))  # src.*
sys.path.insert(0, str(EXP6_DIR))  # exp6lib.*
sys.path.insert(0, str(EXP_DIR))   # exp10lib.*

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp10lib.run_utils import load_config, resolve_cfg_path, select_attacks
from exp10lib.tagging import ALL_TAGS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 10 stage 5: plots.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Scree plot
# ---------------------------------------------------------------------------

def pick_scree_attacks(attack_names: List[str], canonical: str, n_total: int) -> List[str]:
    picks: List[str] = [canonical] if canonical in attack_names else []
    remaining = [a for a in attack_names if a not in picks]
    n_more = max(n_total - len(picks), 0)
    if n_more > 0 and remaining:
        idx = np.linspace(0, len(remaining) - 1, num=min(n_more, len(remaining)), dtype=int)
        picks.extend(remaining[i] for i in sorted(set(idx.tolist())))
    return picks


def plot_scree(cfg: dict, diffs_dir: pathlib.Path, plots_dir: pathlib.Path, attack_names: List[str], layers: List[int]) -> None:
    plot_cfg = cfg.get("plotting", {})
    canonical = plot_cfg.get("canonical_attack", "relevant_start_5")
    n_total = plot_cfg.get("n_scree_attacks", 5)
    scree_attacks = pick_scree_attacks(attack_names, canonical, n_total)
    if not scree_attacks:
        print("  [SKIP] scree plot: no attacks available.")
        return

    n_components = 20
    fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 4.5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, L in zip(axes, layers):
        for attack_name in scree_attacks:
            npz_path = diffs_dir / attack_name / "diffs.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            diffs = data[f"diff_layer{L}"]
            singular_values = np.linalg.svd(diffs, full_matrices=False, compute_uv=False)
            s2 = singular_values ** 2
            var_explained = np.cumsum(s2) / s2.sum()
            k = min(n_components, len(var_explained))
            style = "-" if attack_name == canonical else "--"
            lw = 2.2 if attack_name == canonical else 1.4
            label = f"{attack_name}" + (" (canonical)" if attack_name == canonical else "")
            ax.plot(np.arange(1, k + 1), var_explained[:k], style, linewidth=lw, marker="o", markersize=3, label=label)
        ax.set_title(f"layer {L}", fontsize=12)
        ax.set_xlabel("component index (cumulative)", fontsize=10)
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("cumulative variance explained", fontsize=11)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Scree plot: cumulative variance explained by top-k components", fontsize=13)
    plt.tight_layout()
    out_path = plots_dir / "scree_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# Supplementary: PCA scatter (PC1 vs PC2 of the raw, uncentered diff matrix)
# ---------------------------------------------------------------------------

def _most_anti_aligned_attack(geometry_dir: pathlib.Path, layer: int, exclude: List[str]) -> str | None:
    """The attack whose top-1 direction has the lowest mean cosine similarity
    to every other attack's, at the given layer -- used to make the PCA
    scatter show a genuine outlier alongside typical/canonical attacks."""
    path = geometry_dir / f"cross_attack_cosine_matrix_layer{layer}.csv"
    if not path.exists():
        return None
    mat = pd.read_csv(path, index_col=0)
    n = len(mat)
    if n < 3:
        return None
    mean_cos = (mat.sum(axis=1) - 1.0) / (n - 1)
    mean_cos = mean_cos.drop(labels=[a for a in exclude if a in mean_cos.index], errors="ignore")
    if mean_cos.empty:
        return None
    return str(mean_cos.idxmin())


def plot_pca_scatter(cfg: dict, diffs_dir: pathlib.Path, geometry_dir: pathlib.Path, plots_dir: pathlib.Path,
                      attack_names: List[str], layers: List[int]) -> None:
    plot_cfg = cfg.get("plotting", {})
    canonical = plot_cfg.get("canonical_attack", "relevant_start_5")
    scatter_attacks = pick_scree_attacks(attack_names, canonical, n_total=3)
    outlier = _most_anti_aligned_attack(geometry_dir, layers[-1], exclude=scatter_attacks)
    if outlier is not None:
        scatter_attacks = scatter_attacks + [outlier]
    if not scatter_attacks:
        print("  [SKIP] PCA scatter: no attacks available.")
        return

    tag_colors = {
        "injected_attack_token": "#C44E52", "other_document": "#4878CF",
        "query": "#55A868", "connective_template": "#8172B2",
    }

    n_attacks = len(scatter_attacks)
    fig, axes = plt.subplots(n_attacks, len(layers), figsize=(5.0 * len(layers), 4.2 * n_attacks), squeeze=False)

    for i, attack_name in enumerate(scatter_attacks):
        npz_path = diffs_dir / attack_name / "diffs.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        tags = data["tags"]
        role = "canonical" if attack_name == canonical else ("outlier" if attack_name == outlier else "representative")
        for j, L in enumerate(layers):
            ax = axes[i][j]
            diffs = data[f"diff_layer{L}"]
            # Uncentered SVD, consistent with compute_pca_summary: projecting
            # the raw diff vectors onto their own top-2 right singular vectors
            # (not centered PCA -- see DECISIONS.md).
            _, _, vt = np.linalg.svd(diffs, full_matrices=False)
            proj = diffs @ vt[:2].T
            for tag, color in tag_colors.items():
                mask = tags == tag
                if not mask.any():
                    continue
                ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.45, color=color,
                           label=tag if (i == 0 and j == 0) else None, linewidths=0)
            ax.axhline(0.0, color="gray", linewidth=0.5)
            ax.axvline(0.0, color="gray", linewidth=0.5)
            ax.set_title(f"{attack_name} ({role}) — layer {L}", fontsize=9)
            ax.set_xlabel("PC1 (top singular direction)", fontsize=8)
            if j == 0:
                ax.set_ylabel("PC2", fontsize=8)

    axes[0][0].legend(fontsize=7, markerscale=2.5, loc="best")
    fig.suptitle("PCA scatter: per-position diff vectors projected onto their own top-2 components", fontsize=13)
    plt.tight_layout()
    out_path = plots_dir / "pca_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# 2. Cross-attack cosine heatmap
# ---------------------------------------------------------------------------

def plot_cosine_heatmaps(geometry_dir: pathlib.Path, plots_dir: pathlib.Path, layers: List[int]) -> None:
    mats = {}
    for L in layers:
        path = geometry_dir / f"cross_attack_cosine_matrix_layer{L}.csv"
        if path.exists():
            mats[L] = pd.read_csv(path, index_col=0)
    if not mats:
        print("  [SKIP] cosine heatmap: no cross_attack_cosine_matrix_layer*.csv found.")
        return

    fig, axes = plt.subplots(1, len(mats), figsize=(6.5 * len(mats), 6))
    if len(mats) == 1:
        axes = [axes]

    for ax, L in zip(axes, mats.keys()):
        mat = mats[L]
        im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=6)
        ax.set_title(f"layer {L}", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Cross-attack cosine similarity of top-1 direction (ordered by attack strength)", fontsize=13)
    plt.tight_layout()
    out_path = plots_dir / "cross_attack_cosine_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# 3. Positional concentration grouped bar chart
# ---------------------------------------------------------------------------

def plot_position_concentration(geometry_dir: pathlib.Path, plots_dir: pathlib.Path, layers: List[int]) -> None:
    path = geometry_dir / "position_concentration.csv"
    if not path.exists():
        print("  [SKIP] position concentration bar chart: no position_concentration.csv.")
        return
    df = pd.read_csv(path)
    agg = df.groupby(["layer", "position_tag"])["mean_diff_norm"].mean().reset_index()

    tags = ["injected_attack_token", "other_document", "query", "connective_template"]
    colors = ["#C44E52", "#4878CF", "#55A868", "#8172B2"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(layers))
    width = 0.8 / len(tags)

    for i, (tag, color) in enumerate(zip(tags, colors)):
        vals = [agg[(agg["layer"] == L) & (agg["position_tag"] == tag)]["mean_diff_norm"].mean() for L in layers]
        vals = [v if not np.isnan(v) else 0.0 for v in vals]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=tag, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {L}" for L in layers])
    ax.set_ylabel("mean diff-vector L2 norm (averaged across attacks)", fontsize=11)
    ax.set_title("Positional concentration of the attack-control encoder diff", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = plots_dir / "position_concentration_bars.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# 4. Effective rank vs. layer
# ---------------------------------------------------------------------------

def plot_effective_rank_vs_layer(geometry_dir: pathlib.Path, plots_dir: pathlib.Path, layers: List[int]) -> None:
    path = geometry_dir / "pca_summary_all.csv"
    if not path.exists():
        print("  [SKIP] effective rank plot: no pca_summary_all.csv.")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    attack_names = list(df["attack_name"].unique())
    for i, attack_name in enumerate(attack_names):
        sub = df[df["attack_name"] == attack_name].sort_values("layer")
        color = cmap(i / max(len(attack_names) - 1, 1))
        ax.plot(sub["layer"], sub["effective_rank"], marker="o", markersize=4,
                color=color, alpha=0.8, linewidth=1.3)

    ax.set_xticks(layers)
    ax.set_xlabel("encoder layer", fontsize=11)
    ax.set_ylabel("effective rank (participation ratio)", fontsize=11)
    ax.set_title(f"Effective rank vs. layer, one line per attack (n={len(attack_names)})", fontsize=13)
    plt.tight_layout()
    out_path = plots_dir / "effective_rank_vs_layer.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    layers = cfg["layers"]
    diffs_dir = outputs_base / "diffs"
    geometry_dir = outputs_base / "geometry"
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    attack_names = select_attacks(cfg)

    plot_scree(cfg, diffs_dir, plots_dir, attack_names, layers)
    plot_pca_scatter(cfg, diffs_dir, geometry_dir, plots_dir, attack_names, layers)
    plot_cosine_heatmaps(geometry_dir, plots_dir, layers)
    plot_position_concentration(geometry_dir, plots_dir, layers)
    plot_effective_rank_vs_layer(geometry_dir, plots_dir, layers)

    print(f"\n[05_make_plots] Done -> {plots_dir}")


if __name__ == "__main__":
    main()
