#!/usr/bin/env python3
"""
scripts/04_make_plots.py
==========================
Experiment 2 figures. Reads the DERIVED summaries written by
scripts/03_aggregate.py (run that first); never touches per-example data.

Plots (written to outputs/plots/):

1. defense_by_scale_grid_a.png
     Mean delta_toward_control per flagged head (Grid A, pooled over all 105
     attacks), one line per scale in {0.5, 1.0, 1.5} — identifies the best
     scale per head and whether it's consistent across heads.

2. defense_across_attacks_best_scale_grid_a.png
     Box plot per flagged head of delta_toward_control across the 105
     attacks, AT EACH HEAD'S BEST SCALE (from best_scale_by_head.csv) — a
     within-sample breadth check, not a held-out generalization claim.

3. direction_norm_vs_exp3_effect.png
     Scatter: Part 1's per-head direction norm (grid_a) vs Experiment 3's
     combined_effect_mean. Decoder heads only (Experiment 3 never scored
     encoder heads).

4. sufficiency_by_scale_grid_a.png
     Same layout as (1), for delta_toward_attack (2b, addition intervention).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from headlib.head_hooks import COMPONENT_SHORT  # noqa: E402
from exp2lib.run_utils import load_config, resolve_cfg_path  # noqa: E402

SCALE_COLORS = {0.5: "#1f77b4", 1.0: "#ff7f0e", 1.5: "#2ca02c"}


def head_label(row) -> str:
    return f"L{int(row['layer'])}-{COMPONENT_SHORT[row['component']]}-H{int(row['head_idx'])}"


def _load(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/03_aggregate.py first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plots 1 / 4: effectiveness by scale, one line per scale, heads on x-axis
# ---------------------------------------------------------------------------

def plot_effect_by_scale(by_head_scale: pd.DataFrame, metric: str, ylabel: str,
                          title: str, out_path: pathlib.Path) -> None:
    df = by_head_scale.copy()
    df["head"] = df.apply(head_label, axis=1)
    head_order = (df.groupby("head")[f"{metric}_mean"].mean()
                    .sort_values(ascending=False).index.tolist())
    scales = sorted(df["scale"].unique())

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(head_order)), 5))
    x = np.arange(len(head_order))
    for scale in scales:
        sub = df[df["scale"] == scale].set_index("head").reindex(head_order)
        ax.errorbar(x, sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                    marker="o", markersize=4, capsize=2, linewidth=1.2,
                    color=SCALE_COLORS.get(scale, "#888888"), label=f"scale={scale}")
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: box plot across attacks at each head's best scale
# ---------------------------------------------------------------------------

def plot_across_attacks_best_scale(by_head_scale_attack: pd.DataFrame,
                                    best_scale_by_head: pd.DataFrame,
                                    out_path: pathlib.Path) -> None:
    keys = ["layer", "component", "head_idx"]
    merged = by_head_scale_attack.merge(
        best_scale_by_head[keys + ["scale"]], on=keys + ["scale"], how="inner",
    )
    merged["head"] = merged.apply(head_label, axis=1)
    med = (merged.groupby("head")["delta_toward_control_mean"].median()
                 .sort_values(ascending=False))
    heads = med.index.tolist()
    data = [merged.loc[merged["head"] == h, "delta_toward_control_mean"].dropna().values
            for h in heads]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(heads))))
    ax.boxplot(data[::-1], vert=False, patch_artist=True, tick_labels=heads[::-1],
               medianprops=dict(color="#222222"),
               flierprops=dict(marker=".", markersize=3, alpha=0.5))
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("delta_toward_control (per-attack mean), at each head's best scale")
    ax.set_title("Defense effectiveness across 105 attacks — best scale per head\n"
                 "within-sample breadth check, not a held-out generalization claim",
                 fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: direction norm vs Exp3 combined effect
# ---------------------------------------------------------------------------

def plot_norm_vs_effect(merged: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for comp, color in [("decoder_cross_attn", "#ff7f0e"), ("decoder_self_attn", "#1f77b4")]:
        sub = merged[merged["component"] == comp]
        if sub.empty:
            continue
        ax.scatter(sub["direction_norm"], sub["combined_effect_mean"],
                   s=30, alpha=0.75, color=color, edgecolors="white",
                   linewidths=0.4, label=comp)
    pearson = merged["direction_norm"].corr(merged["combined_effect_mean"], method="pearson")
    spearman = merged["direction_norm"].corr(merged["combined_effect_mean"], method="spearman")
    ax.set_xlabel("mean-diff direction norm (Part 1, grid_a)")
    ax.set_ylabel("Experiment 3 combined_effect_mean")
    ax.set_title(f"Direction norm vs causal importance (n={len(merged)})\n"
                 f"pearson={pearson:.3f}  spearman={spearman:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 2 figures.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[04_make_plots] Plots -> {plots_dir}")

    agg_dir = outputs_base / "interventions" / "grid_a" / "aggregated"

    # 1. Defense effectiveness by scale
    try:
        by_hs = _load(agg_dir / "defense_by_head_scale.csv")
        plot_effect_by_scale(
            by_hs, "delta_toward_control", "mean delta_toward_control",
            "Defense effectiveness by scale, per flagged head (Grid A)",
            plots_dir / "defense_by_scale_grid_a.png",
        )
    except FileNotFoundError as e:
        print(f"  SKIP defense_by_scale_grid_a: {e}")

    # 2. Defense across attacks at best scale
    try:
        by_hsa = _load(agg_dir / "defense_by_head_scale_attack.csv")
        best = _load(agg_dir / "best_scale_by_head.csv")
        plot_across_attacks_best_scale(
            by_hsa, best, plots_dir / "defense_across_attacks_best_scale_grid_a.png",
        )
    except FileNotFoundError as e:
        print(f"  SKIP defense_across_attacks_best_scale_grid_a: {e}")

    # 3. Direction norm vs Exp3 combined effect
    try:
        merged = _load(outputs_base / "directions" / "direction_norm_vs_exp3_effect.csv")
        plot_norm_vs_effect(merged, plots_dir / "direction_norm_vs_exp3_effect.png")
    except FileNotFoundError as e:
        print(f"  SKIP direction_norm_vs_exp3_effect: {e}")

    # 4. Sufficiency test by scale
    try:
        suff_hs = _load(agg_dir / "sufficiency_by_head_scale.csv")
        plot_effect_by_scale(
            suff_hs, "delta_toward_attack", "mean delta_toward_attack",
            "Sufficiency effectiveness by scale, per flagged head (Grid A)",
            plots_dir / "sufficiency_by_scale_grid_a.png",
        )
    except FileNotFoundError as e:
        print(f"  SKIP sufficiency_by_scale_grid_a: {e}")

    print("[04_make_plots] Done.")


if __name__ == "__main__":
    main()
