#!/usr/bin/env python3
"""
scripts/04_make_plot.py
==========================
The one compact deliverable: a scatter plot, one point per attack
configuration (105 points), x = mean_delta_score, y = mean_delta_rank,
point color = success_rate.

Quadrant interpretation (caption, not cluttering the plot):
  top-right    : score AND rank both improve
  bottom-left  : score AND rank both worsen
  top-left     : score decreases but rank improves
  bottom-right : score increases but rank does not improve
"""

from __future__ import annotations

import argparse
import json
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
import pandas as pd  # noqa: E402

from exp9lib.run_utils import load_config, resolve_cfg_path  # noqa: E402

ALWAYS_LABEL = ["relevant_start_5", "information_start_5"]


def pick_quadrant_representative(df: pd.DataFrame, x_sign: int, y_sign: int, exclude: set) -> str:
    """Most extreme (max distance from origin) attack strictly in the given quadrant."""
    cond = (df["mean_delta_score"] * x_sign > 0) & (df["mean_delta_rank"] * y_sign > 0)
    cand = df[cond & ~df["attack_name"].isin(exclude)]
    if cand.empty:
        return None
    dist = cand["mean_delta_score"] ** 2 + cand["mean_delta_rank"] ** 2
    return cand.loc[dist.idxmax(), "attack_name"]


def pick_labels(df: pd.DataFrame) -> list:
    labels = [name for name in ALWAYS_LABEL if name in set(df["attack_name"])]
    excl = set(labels)
    for x_sign, y_sign in [(-1, 1), (1, -1), (-1, -1)]:  # top-left, bottom-right, bottom-left
        pick = pick_quadrant_representative(df, x_sign, y_sign, excl)
        if pick is not None:
            labels.append(pick)
            excl.add(pick)
    return labels


def main() -> None:
    p = argparse.ArgumentParser(description="Score-vs-rank scatter plot.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    agg_dir = outputs_base / "aggregate"

    summary_path = agg_dir / "per_attack_summary.csv"
    if not summary_path.exists():
        sys.exit(f"{summary_path} not found — run scripts/03_aggregate.py first.")
    df = pd.read_csv(summary_path)
    with open(agg_dir / "global_spearman.json", encoding="utf-8") as fh:
        global_stats = json.load(fh)

    labels = pick_labels(df)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        df["mean_delta_score"], df["mean_delta_rank"], c=df["success_rate"],
        cmap="viridis", s=45, edgecolors="white", linewidths=0.5, vmin=0, vmax=1,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("success_rate (fraction of examples with rank improved)")

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1, zorder=0)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1, zorder=0)

    for name in labels:
        row = df[df["attack_name"] == name].iloc[0]
        ax.annotate(
            name, (row["mean_delta_score"], row["mean_delta_rank"]),
            textcoords="offset points", xytext=(6, 5), fontsize=8, color="#222222",
        )

    ax.set_xlabel("mean delta_score (attack_score - control_score)")
    ax.set_ylabel("mean delta_rank (rank_control - rank_attack; positive = improved)")
    rho = global_stats["spearman_global"]
    n = global_stats["n"]
    ax.set_title(
        f"Score change vs. rank change across {len(df)} attack configurations\n"
        f"global Spearman rho = {rho:.3f} (n={n} examples, pooled)\n"
        "top-right: both improve | bottom-left: both worsen | "
        "top-left: score down, rank up | bottom-right: score up, rank flat/down",
        fontsize=10,
    )

    out_path = outputs_base / "plots" / "score_vs_rank_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[04_make_plot] wrote {out_path}")


if __name__ == "__main__":
    main()
