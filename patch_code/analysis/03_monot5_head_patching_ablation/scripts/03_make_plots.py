#!/usr/bin/env python3
"""
scripts/03_make_plots.py
========================
Experiment 3 figures.  Reads the DERIVED summaries written by
scripts/02_aggregate.py (run that first); never touches per-example data.

Plots (written to outputs/plots/):

1. head_heatmap_grid_a.png
     288-head map (rows = decoder layer, cols = head; one panel per
     component), coloured by mean combined_effect over all attacks x
     examples of Grid A.  Localises candidate heads.

2. zero_vs_mean_ablation_scatter.png
     Per head: mean score_drop_zero vs mean score_drop_mean (Grid A, attack
     base).  On the diagonal = the head's entire importance is attack-
     specific signal; high-zero/low-mean = generally important head, not
     attack-specific.

3. per_head_effect_across_attacks.png
     Box plot per head of the per-attack mean combined_effect across the
     105 attacks (top-N heads shown).  A tight box high above 0 = universal
     attack head; a scattered box = attack/token-specific head.

4. clean_vs_attack_drop.png
     For candidate heads: mean score drop when ablated on ATTACK inputs
     (Grid A) vs on CLEAN inputs (side-effect run A).  A large drop on both
     = risky ablation target; attack-only drop = safe target.

5. head_heatmap_grid_b.png
     Same layout as (1) for Grid B (canonical attack, n=100) — sanity check
     against Grid A.

Colour notes: effects are polar quantities (0 = irrelevant, positive =
carries the attack), so heatmaps use a diverging CVD-safe map (RdBu_r)
centred at 0.  The two components get a fixed blue/orange pair.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))  # for src.* (imported via headlib.run_utils)
sys.path.insert(0, str(EXP_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from headlib.head_hooks import COMPONENT_SHORT, HEAD_COMPONENTS
from headlib.run_utils import load_config, resolve_cfg_path

COMPONENT_LABELS = {
    "decoder_self_attn": "Decoder Self-Attention",
    "decoder_cross_attn": "Decoder Cross-Attention",
}
COMPONENT_COLORS = {  # fixed assignment, CVD-safe pair
    "decoder_self_attn": "#1f77b4",   # blue
    "decoder_cross_attn": "#ff7f0e",  # orange
}


def head_label(row) -> str:
    """Short head name, e.g. L3-X-H7 (X = cross-attn, S = self-attn)."""
    return f"L{int(row['layer'])}-{COMPONENT_SHORT[row['component']]}-H{int(row['head_idx'])}"


def _load_summary(outputs_base: pathlib.Path, run: str, which: str) -> pd.DataFrame:
    path = outputs_base / run / "aggregated" / f"{which}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/02_aggregate.py first "
            f"(and scripts/01_run_head_grid.py --run {run} before that)."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plot 1 / 5: 288-head heatmap
# ---------------------------------------------------------------------------

def plot_head_heatmap(head_summary: pd.DataFrame, metric: str, title: str,
                      out_path: pathlib.Path) -> None:
    comps = [c for c in HEAD_COMPONENTS if c in head_summary["component"].unique()]
    pivots = {
        c: head_summary[head_summary["component"] == c]
        .pivot(index="layer", columns="head_idx", values=metric)
        .sort_index()
        for c in comps
    }
    vmax = max(np.nanmax(np.abs(p.values)) for p in pivots.values())
    vmax = max(float(vmax), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, len(comps), figsize=(5.5 * len(comps) + 1, 5.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, comp in zip(axes, comps):
        piv = pivots[comp]
        im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(COMPONENT_LABELS[comp], fontsize=12)
        ax.set_xlabel("head index")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
    axes[0].set_ylabel("decoder layer")
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(metric.replace("_", " "))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: zero vs mean ablation scatter
# ---------------------------------------------------------------------------

def plot_zero_vs_mean_scatter(head_summary: pd.DataFrame, out_path: pathlib.Path,
                              n_annotate: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for comp in HEAD_COMPONENTS:
        sub = head_summary[head_summary["component"] == comp]
        if sub.empty:
            continue
        ax.scatter(sub["score_drop_zero_mean"], sub["score_drop_mean_mean"],
                   s=28, alpha=0.75, color=COMPONENT_COLORS[comp],
                   edgecolors="white", linewidths=0.4,
                   label=COMPONENT_LABELS[comp])

    lims = np.array([ax.get_xlim(), ax.get_ylim()])
    lo, hi = lims.min(), lims.max()
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1,
            zorder=0, label="drop(zero) = drop(mean)")
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)

    # Direct-label only the most important heads (by mean-ablation drop)
    top = head_summary.nlargest(n_annotate, "score_drop_mean_mean")
    for _, row in top.iterrows():
        ax.annotate(head_label(row),
                    (row["score_drop_zero_mean"], row["score_drop_mean_mean"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7,
                    color="#333333")

    ax.set_xlabel("mean score drop — zero ablation (attack input)")
    ax.set_ylabel("mean score drop — mean ablation (attack input)")
    ax.set_title("Zero vs mean ablation per head (Grid A)\n"
                 "diagonal = attack-specific; below diagonal = generally important",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: per-head effect across attacks (box plot)
# ---------------------------------------------------------------------------

def plot_per_head_across_attacks(head_attack: pd.DataFrame, out_path: pathlib.Path,
                                 top_n: int = 30) -> None:
    ha = head_attack.copy()
    ha["head"] = ha.apply(head_label, axis=1)
    med = (ha.groupby("head")["combined_effect_mean"].median()
             .sort_values(ascending=False))
    heads = med.head(top_n).index.tolist()
    data = [ha.loc[ha["head"] == h, "combined_effect_mean"].dropna().values
            for h in heads]
    colors = [COMPONENT_COLORS[ha.loc[ha["head"] == h, "component"].iloc[0]]
              for h in heads]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(heads))))
    bp = ax.boxplot(data[::-1], vert=False, patch_artist=True,
                    tick_labels=heads[::-1],
                    medianprops=dict(color="#222222"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors[::-1]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("per-attack mean combined_effect")
    ax.set_title(f"Per-head combined effect across attacks — top {len(heads)} heads (Grid A)\n"
                 "tight box above 0 = universal attack head; wide box = token/position-specific",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COMPONENT_COLORS[c], alpha=0.55)
               for c in HEAD_COMPONENTS]
    ax.legend(handles, [COMPONENT_LABELS[c] for c in HEAD_COMPONENTS],
              fontsize=9, loc="lower right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: clean vs attack score drop for candidate heads
# ---------------------------------------------------------------------------

def plot_clean_vs_attack_drop(grid_summary: pd.DataFrame, clean_summary: pd.DataFrame,
                              out_path: pathlib.Path, top_n: int = 15) -> None:
    keys = ["layer", "component", "head_idx"]
    merged = grid_summary.merge(clean_summary, on=keys, suffixes=("_attack", "_clean"))
    cand = merged.nlargest(top_n, "combined_effect_mean").copy()
    cand["head"] = cand.apply(head_label, axis=1)

    x = np.arange(len(cand))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, method in zip(axes, ["zero", "mean"]):
        atk = cand[f"score_drop_{method}_mean_attack"]
        cln = cand[f"score_drop_{method}_mean_clean"]
        ax.bar(x - width / 2, atk, width, label="attack input",
               color="#1f77b4", edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, cln, width, label="clean input",
               color="#9aa0a6", edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cand["head"], rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{method} ablation")
    axes[0].set_ylabel("mean score drop when head ablated")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Score drop on attack vs clean inputs — top {len(cand)} candidate heads\n"
                 "attack-only drop = safe ablation target; drop on both = risky",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 3 figures.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--top-n-box", type=int, default=30,
                   help="Heads shown in the across-attacks box plot.")
    p.add_argument("--top-n-bars", type=int, default=15,
                   help="Candidate heads shown in the clean-vs-attack bar chart.")
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[03_make_plots] Plots → {plots_dir}")

    # 1. Grid A heatmap
    ga_head = _load_summary(outputs_base, "grid_a", "head_summary")
    plot_head_heatmap(
        ga_head, "combined_effect_mean",
        "Per-head combined patching effect — Grid A (all attacks)",
        plots_dir / "head_heatmap_grid_a.png",
    )

    # 2. Zero vs mean scatter
    plot_zero_vs_mean_scatter(ga_head, plots_dir / "zero_vs_mean_ablation_scatter.png")

    # 3. Per-head effect across attacks
    ga_head_attack = _load_summary(outputs_base, "grid_a", "head_attack_summary")
    plot_per_head_across_attacks(
        ga_head_attack, plots_dir / "per_head_effect_across_attacks.png",
        top_n=args.top_n_box,
    )

    # 4. Clean vs attack drop (needs clean_a)
    try:
        ca_head = _load_summary(outputs_base, "clean_a", "head_summary")
        plot_clean_vs_attack_drop(
            ga_head, ca_head, plots_dir / "clean_vs_attack_drop.png",
            top_n=args.top_n_bars,
        )
    except FileNotFoundError as e:
        print(f"  SKIP clean_vs_attack_drop: {e}")

    # 5. Grid B heatmap
    try:
        gb_head = _load_summary(outputs_base, "grid_b", "head_summary")
        gb_name = cfg["runs"]["grid_b"]["attack_name"]
        gb_n = cfg["runs"]["grid_b"]["n_examples"]
        plot_head_heatmap(
            gb_head, "combined_effect_mean",
            f"Per-head combined patching effect — Grid B ({gb_name}, n={gb_n})",
            plots_dir / "head_heatmap_grid_b.png",
        )
    except FileNotFoundError as e:
        print(f"  SKIP head_heatmap_grid_b: {e}")

    print("[03_make_plots] Done.")


if __name__ == "__main__":
    main()
