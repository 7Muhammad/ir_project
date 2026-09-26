#!/usr/bin/env python3
"""
scripts/04_make_plots.py
===========================
Experiment 7 figures. Reads the DERIVED summaries written by
scripts/03_aggregate.py; never touches per-example data. Uses the
`canonical` run (single attack, n=100 — deeper, more stable per-layer
statistics) for all three plots, matching Experiments 3/6's convention.

1. bucket_composition_by_layer.png
     Stacked bar of top-k bucket composition per layer, clean vs attack —
     when does the attack-token share start dominating top-k?
2. attack_token_trajectory.png
     Attack-token rank (log scale) and logit per layer, clean/control/attack
     overlaid — the promotion curve.
3. per_head_vs_whole_block.png
     For each Experiment-3-flagged head, its most frequent top-5 tokens
     (attack run) vs the whole-block's at the same layer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp7lib.run_utils import load_config, resolve_cfg_path

BUCKETS = ["attack", "query", "document", "stopword", "other"]
BUCKET_COLORS = {"attack": "#d62728", "query": "#1f77b4", "document": "#2ca02c",
                  "stopword": "#7f7f7f", "other": "#c7c7c7"}
RUN_TYPE_COLORS = {"clean": "#1f77b4", "control": "#7f7f7f", "attack": "#d62728"}


def _load(path: pathlib.Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/02_run_grid.py then scripts/03_aggregate.py first ({what}).")
    return pd.read_csv(path)


def plot_bucket_composition(comp: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, run_type in zip(axes, ["clean", "attack"]):
        sub = comp[comp["run_type"] == run_type].sort_values("layer")
        layers = sub["layer"].to_numpy()
        bottom = np.zeros(len(layers))
        for bucket in BUCKETS:
            ax.bar(layers, sub[bucket], bottom=bottom, color=BUCKET_COLORS[bucket], label=bucket, width=0.8)
            bottom += sub[bucket].to_numpy()
        ax.set_xlabel("decoder layer")
        ax.set_title(f"{run_type} input")
        ax.set_xticks(layers)
    axes[0].set_ylabel("% of top-k (k=10)")
    axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=5)
    fig.suptitle("Top-k bucket composition by layer — clean vs attack\n"
                 "(whole-block cross-attention contribution, canonical attack, n=100)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_attack_token_trajectory(traj: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run_type in ["clean", "control", "attack"]:
        sub = traj[traj["run_type"] == run_type].sort_values("layer")
        axes[0].plot(sub["layer"], sub["attack_token_rank_mean"], marker="o",
                     color=RUN_TYPE_COLORS[run_type], label=run_type)
        axes[1].plot(sub["layer"], sub["attack_token_logit_mean"], marker="o",
                     color=RUN_TYPE_COLORS[run_type], label=run_type)
    axes[0].set_yscale("log")
    axes[0].invert_yaxis()  # rank 1 (best) at the top
    axes[0].set_ylabel("attack-token rank (log scale, 1 = top)")
    axes[1].set_ylabel("attack-token logit")
    for ax in axes:
        ax.set_xlabel("decoder layer")
        ax.legend(fontsize=9)
    fig.suptitle("Attack-token promotion across decoder layers\n"
                 "clean/control = baseline; attack = the promotion curve", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_head_vs_whole_block(freq: pd.DataFrame, out_path: pathlib.Path, run_type: str = "attack", top_n_tokens: int = 3) -> None:
    sub = freq[freq["run_type"] == run_type]
    if sub.empty:
        print(f"  SKIP per_head_vs_whole_block: no '{run_type}' rows in per_head_token_frequency.csv")
        return

    head_rows = sub[sub["head"].notna()].copy()
    head_rows["head"] = head_rows["head"].astype(int)
    row_keys = sorted(head_rows[["layer", "head"]].drop_duplicates().itertuples(index=False), key=lambda r: (r.layer, r.head))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(row_keys) * 2)))
    y = 0
    yticks, yticklabels = [], []
    for layer, head in row_keys:
        for label, mask in [(f"L{layer} whole-block", (sub["layer"] == layer) & sub["head"].isna()),
                             (f"L{layer} H{head}", (sub["layer"] == layer) & (sub["head"] == head))]:
            top = sub[mask].nlargest(top_n_tokens, "frequency_pct")
            for _, r in top.iterrows():
                ax.barh(y, r["frequency_pct"], color="#1f77b4" if "whole-block" in label else "#d62728", alpha=0.75)
                ax.text(r["frequency_pct"] + 1, y, r["token"], va="center", fontsize=8)
            yticks.append(y)
            yticklabels.append(label)
            y -= 1
        y -= 0.5

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel(f"frequency across examples (%) — top {top_n_tokens} tokens shown")
    ax.set_xlim(0, 105)
    fig.suptitle(f"Per-head logit lens vs whole-block, same layer ({run_type} input)\n"
                 "blue = whole-block, red = flagged head", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 7 figures.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[04_make_plots] Plots → {plots_dir}")

    agg_dir = outputs_base / "canonical" / "aggregated"

    comp = _load(agg_dir / "composition_summary.csv", "bucket composition")
    plot_bucket_composition(comp, plots_dir / "bucket_composition_by_layer.png")

    traj = _load(agg_dir / "attack_token_trajectory.csv", "attack-token trajectory")
    plot_attack_token_trajectory(traj, plots_dir / "attack_token_trajectory.png")

    freq = _load(agg_dir / "per_head_token_frequency.csv", "per-head token frequency")
    plot_per_head_vs_whole_block(freq, plots_dir / "per_head_vs_whole_block.png")

    print("[04_make_plots] Done.")


if __name__ == "__main__":
    main()
