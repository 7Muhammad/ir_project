#!/usr/bin/env python3
"""
scripts/07_plot.py
===================
Figures (read-only on stage 05/06 outputs). Heads always appear in the
canonical config order (Exp 11 importance order) — never re-sorted by value —
and every head is shown, significant or not. No rows are dropped; where an
axis range is limited for readability the number of points outside the view
is printed on the figure.

  fig1_edge_combined_by_head.png    all-query e_combined, 95% bootstrap CI, FDR marker
  fig2_fwd_vs_rev_by_head.png       all-query e_fwd vs e_rev with CIs
  fig3_edge_vs_whole_head.png       edge combined vs whole-head combined (per head)
  fig4_attack_level_distribution.png distribution of the 105 attack-level means per head
  fig5_attack_head_heatmap.png      head x attack mean e_combined (sweep consistency)
  fig6_token_distribution.png       per-token e_combined distribution per head (descriptive)
  fig7_token_examples.png           heads x query tokens for a few fixed examples (descriptive;
                                    token axes are per-example, NOT aligned across queries)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from exp15lib.config import load_config, output_dir  # noqa: E402
from exp15lib.heads import load_heads  # noqa: E402
from exp15lib.run_utils import cheap_stage_is_current, now, upstream_fingerprint, write_status  # noqa: E402

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("exp15_div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8431b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9, "axes.titlesize": 10,
})


def parse_args():
    p = argparse.ArgumentParser(description="Exp 15 stage 07: plots")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def zero_line(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color=MUTED, lw=1, zorder=1)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    agg, sta = out / "05_aggregate", out / "06_statistics"
    stage = out / "plots"
    for p in (agg / "status.json", sta / "status.json"):
        if not p.exists():
            raise RuntimeError(f"missing upstream {p}")
    fp = upstream_fingerprint([agg / "status.json", sta / "status.json"])
    if cheap_stage_is_current(stage, fp, ["fig1_edge_combined_by_head.png"], args.force):
        print("[07] plots up to date; skipping")
        return
    stage.mkdir(parents=True, exist_ok=True)

    labels = [h.label for h in load_heads(cfg)]
    st = pd.read_csv(sta / "head_statistics.csv").set_index("head_label").reindex(labels)
    alpha = float(cfg["statistics"]["fdr_alpha"])
    rej = st[f"reject_FDR_{alpha}"].astype(bool)
    y = np.arange(len(labels))[::-1]
    n_att = int(st.n_attacks.iloc[0])

    # ---- fig1 -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(labels) + 1.4))
    zero_line(ax, vertical=True)
    for yi, lab in zip(y, labels):
        r = st.loc[lab]
        ax.plot([r.e_combined_ci_low, r.e_combined_ci_high], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
        ax.scatter(r.mean_e_combined, yi, s=42, zorder=3, edgecolor=BLUE, lw=1.5,
                   facecolor=BLUE if rej[lab] else SURFACE)
    ax.set_yticks(y, labels)
    ax.set_xlabel("all-query edge recovery  e_combined = min(e_fwd, e_rev)")
    ax.set_title(f"Attack→query edge recovery per head (equal weight over {n_att} attacks, 95% bootstrap CI)\n"
                 f"filled = BH-FDR significant at {alpha} (one-sided sign-flip); hollow = not significant",
                 loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, stage / "fig1_edge_combined_by_head.png")

    # ---- fig2 -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(labels) + 1.4))
    zero_line(ax, vertical=True)
    for metric, col, off, name in (("e_fwd", BLUE, 0.17, "forward (sufficiency)"),
                                   ("e_rev", ORANGE, -0.17, "reverse (necessity)")):
        ax.hlines(y + off, st[f"{metric}_ci_low"], st[f"{metric}_ci_high"], color=col, lw=2)
        ax.scatter(st[f"mean_{metric}"], y + off, color=col, s=34, zorder=3, label=name,
                   edgecolor=SURFACE, lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("all-query edge recovery (equal-weight mean, 95% bootstrap CI)")
    ax.set_title("Forward vs reverse attack→query edge recovery per head", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, stage / "fig2_fwd_vs_rev_by_head.png")

    # ---- fig3 -----------------------------------------------------------------
    g = pd.read_csv(agg / "global_head_summary.csv").set_index("head_label").reindex(labels)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    zero_line(ax)
    zero_line(ax, vertical=True)
    lo = min(g.mean_whole_head_combined.min(), g.mean_e_combined.min(), 0) - 0.01
    hi = max(g.mean_whole_head_combined.max(), g.mean_e_combined.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], ls="--", color=MUTED, lw=1, label="edge = whole head")
    ax.scatter(g.mean_whole_head_combined, g.mean_e_combined, s=40, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3)
    for lab, r in g.iterrows():
        ax.annotate(lab, (r.mean_whole_head_combined, r.mean_e_combined), xytext=(4, 3),
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("whole-head combined recovery (Exp 11 patch, same samples)")
    ax.set_ylabel("all-query edge combined recovery")
    ax.set_title("How much of each head's effect runs through the attack→query edge?\n"
                 "(equal-weight means; ratio is secondary — see edge_vs_whole_head.csv)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    save(fig, stage / "fig3_edge_vs_whole_head.png")

    # ---- fig4 -----------------------------------------------------------------
    pah = pd.read_csv(agg / "per_attack_head_all_query.csv")
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(labels) + 2), 4.2))
    zero_line(ax)
    data = [pah.loc[pah.head_label == l, "e_combined"].to_numpy() for l in labels]
    ax.boxplot(data, positions=range(len(labels)), widths=0.55, showfliers=False,
               medianprops={"color": INK, "lw": 1.5}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED},
               capprops={"color": MUTED})
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=9, color=BLUE, alpha=0.55, lw=0, zorder=3)
    ax.set_xticks(range(len(labels)), labels, rotation=60)
    ax.set_ylabel("attack-level mean e_combined")
    ax.set_title(f"Consistency across the sweep: one dot per attack ({n_att} per head), all-query condition",
                 loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, stage / "fig4_attack_level_distribution.png")

    # ---- fig5 -----------------------------------------------------------------
    order = list(dict.fromkeys(pah.sort_values(["attack_token", "attack_position", "repetitions"]).attack_name))
    mat = pah.pivot(index="head_label", columns="attack_name", values="e_combined").reindex(index=labels, columns=order)
    v = np.nanmax(np.abs(mat.values)) or 1e-6
    fig, ax = plt.subplots(figsize=(max(8, 0.13 * len(order) + 3), 0.3 * len(labels) + 2))
    im = ax.imshow(mat.values, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=5 if len(order) > 30 else 7)
    ax.grid(False)
    ax.set_title("Attack-level mean all-query e_combined (rows: heads; columns: attacks grouped token/position/reps)",
                 loc="left")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="e_combined")
    save(fig, stage / "fig5_attack_head_heatmap.png")

    # ---- fig6 -----------------------------------------------------------------
    sq = pd.read_csv(agg / "raw_single_query_edges.csv.gz",
                     usecols=["head_label", "attack_name", "example_id", "query_token_index", "e_combined"])
    data = [sq.loc[sq.head_label == l, "e_combined"].to_numpy() for l in labels]
    allv = np.concatenate(data)
    lo, hi = np.quantile(allv, [0.005, 0.995])
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    n_out = int(((allv < lo) | (allv > hi)).sum())
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(labels) + 2), 4.2))
    zero_line(ax)
    ax.boxplot(data, positions=range(len(labels)), widths=0.55, whis=(5, 95), showfliers=False,
               medianprops={"color": INK, "lw": 1.5}, boxprops={"color": BLUE}, whiskerprops={"color": BLUE},
               capprops={"color": BLUE})
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(labels)), labels, rotation=60)
    ax.set_ylabel("single-query-token e_combined")
    ax.set_title("Descriptive: per-token edge recovery (box = IQR, whiskers = 5–95th pct.)", loc="left")
    ax.text(0.99, 0.02, f"{n_out} of {len(allv)} token rows outside view (kept in raw CSV)",
            transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    ax.grid(axis="x", visible=False)
    save(fig, stage / "fig6_token_distribution.png")

    # ---- fig7 -----------------------------------------------------------------
    raw = pd.read_csv(agg / "raw_single_query_edges.csv.gz",
                      usecols=["head_label", "attack_name", "example_id", "sample_index", "query_token_index",
                               "query_token_string", "e_combined"])
    wanted = [a for a in cfg["sanity"]["attacks"] if a in set(raw.attack_name)] or list(dict.fromkeys(raw.attack_name))[:3]
    panels = []
    for a in wanted[:3]:
        sub = raw[(raw.attack_name == a) & (raw.sample_index == 0)]
        panels.append((a, sub))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.3 * len(labels) + 2.2), squeeze=False)
    for ax, (a, sub) in zip(axes[0], panels):
        m = sub.pivot(index="head_label", columns="query_token_index", values="e_combined").reindex(labels)
        toks = sub.drop_duplicates("query_token_index").sort_values("query_token_index").query_token_string
        vv = np.nanmax(np.abs(m.values)) or 1e-6
        im = ax.imshow(m.values, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -vv, vv), interpolation="nearest")
        ax.set_xticks(range(m.shape[1]), list(toks), rotation=60, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.grid(False)
        ax.set_title(f"{a}\n{sub.example_id.iloc[0]}", fontsize=8, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Descriptive single-token e_combined for fixed example(s) — token axes are per query, not aligned",
                 x=0.01, y=1.08, ha="left", fontsize=9)
    save(fig, stage / "fig7_token_examples.png")

    write_status(stage, {"status": "success", "finished": now(), "upstream_fingerprint": fp})


if __name__ == "__main__":
    main()
