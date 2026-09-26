#!/usr/bin/env python3
"""
scripts/04_make_plots.py
==========================
The 5 main compact figures (see experiment prompt's "PLOTS" section).
Reads only outputs/aggregates/*.csv (written by scripts/03_aggregate.py) --
never the raw per-example CSVs. Per-head / per-attack detail figures belong
in outputs/diagnostics/, not here.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from exp12lib.run_utils import load_config, resolve_cfg_path
from exp12lib.structural_masks import STRUCTURAL_GROUP_NAMES

WORD_GROUP_ORDER = ["content_matched", "content_unmatched", "stopword_matched", "stopword_unmatched"]
WORD_GROUP_LABELS = ["content\nmatched", "content\nunmatched", "stopword\nmatched", "stopword\nunmatched"]
STRUCTURAL_LABELS = {
    "Query": "Query", "Query_colon": "Query :", "query_content": "query content",
    "Document": "Document", "Document_colon": "Document :", "attack_tokens": "attack tokens",
    "original_document": "original document", "Relevant": "Relevant", "Relevant_colon": "Relevant :",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 12 -- main plots.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def _heatmap(pivot: pd.DataFrame, title: str, cbar_label: str, out_path: pathlib.Path,
             xtick_labels=None) -> None:
    vmax = max(float(np.nanmax(np.abs(pivot.values))), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * pivot.shape[1]), 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(xtick_labels if xtick_labels is not None else pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("query-word / structural group")
    ax.set_ylabel("encoder layer")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot1_main_causal_heatmap(agg_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    path = agg_dir / "causal_by_layer_wordgroup_global.csv"
    if not path.exists():
        print(f"  [skip] {path} not found"); return
    df = pd.read_csv(path)
    pivot = df.pivot(index="layer", columns="word_group", values="mean_combined")
    pivot = pivot.reindex(columns=[g for g in WORD_GROUP_ORDER if g in pivot.columns]).sort_index()
    _heatmap(pivot, "Causal effect by query-word group (example-balanced, pooled over 105 attacks)\n"
             "value = mean combined_effect = min(forward, reverse)",
             "mean combined_effect", plots_dir / "plot1_main_causal_heatmap.png",
             xtick_labels=[WORD_GROUP_LABELS[WORD_GROUP_ORDER.index(g)] for g in pivot.columns])


def plot2_structural_causal_heatmap(agg_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    path = agg_dir / "causal_by_layer_structural_global.csv"
    if not path.exists():
        print(f"  [skip] {path} not found"); return
    df = pd.read_csv(path)
    pivot = df.pivot(index="layer", columns="intervention_name", values="mean_combined")
    pivot = pivot.reindex(columns=[g for g in STRUCTURAL_GROUP_NAMES if g in pivot.columns]).sort_index()
    _heatmap(pivot, "Causal effect by structural position group (pooled over 105 attacks)\n"
             "value = mean combined_effect = min(forward, reverse)",
             "mean combined_effect", plots_dir / "plot2_structural_causal_heatmap.png",
             xtick_labels=[STRUCTURAL_LABELS[g] for g in pivot.columns])


def plot3_attack_to_query_attention(agg_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    path = agg_dir / "attention_by_layer_wordgroup_global.csv"
    if not path.exists():
        print(f"  [skip] {path} not found"); return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in WORD_GROUP_ORDER:
        sub = df[df["word_group"] == g].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["mean_attack_to_query_delta"], marker="o", label=g)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("delta attention mass (attack -> query word)")
    ax.set_title("Attack-token -> query-word attention, attack minus control\n(layer-averaged over all 12 heads)")
    ax.legend(fontsize=8)
    fig.savefig(plots_dir / "plot3_attack_to_query_attention.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plots_dir / 'plot3_attack_to_query_attention.png'}")


def plot4_query_to_attack_attention(agg_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    path = agg_dir / "attention_by_layer_wordgroup_global.csv"
    if not path.exists():
        print(f"  [skip] {path} not found"); return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in WORD_GROUP_ORDER:
        sub = df[df["word_group"] == g].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["mean_query_to_attack_delta"], marker="o", label=g)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("delta attention mass (query word -> attack)")
    ax.set_title("Query-word -> attack-token attention, attack minus control\n(layer-averaged over all 12 heads)")
    ax.legend(fontsize=8)
    fig.savefig(plots_dir / "plot4_query_to_attack_attention.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plots_dir / 'plot4_query_to_attack_attention.png'}")


def plot5_important_head_attention(agg_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    path = agg_dir / "attention_important_heads_by_layer_head_wordgroup.csv"
    if not path.exists():
        print(f"  [skip] {path} not found"); return
    df = pd.read_csv(path)
    df["head_label"] = "L" + df["layer"].astype(str) + "H" + df["head"].astype(int).astype(str)
    head_order = (df[["layer", "head", "head_label"]].drop_duplicates()
                  .sort_values(["layer", "head"])["head_label"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.5 * len(head_order))), sharey=True)
    for ax, value_col, direction in zip(
        axes, ["mean_attack_to_query_delta", "mean_query_to_attack_delta"],
        ["attack -> query word", "query word -> attack"],
    ):
        pivot = df.pivot(index="head_label", columns="word_group", values=value_col)
        pivot = pivot.reindex(index=head_order, columns=[g for g in WORD_GROUP_ORDER if g in pivot.columns])
        vmax = max(float(np.nanmax(np.abs(pivot.values))), 1e-6)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([WORD_GROUP_LABELS[WORD_GROUP_ORDER.index(g)] for g in pivot.columns],
                            rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(direction, fontsize=10)
        fig.colorbar(im, ax=ax, label="delta attention", shrink=0.8)
    fig.suptitle("Attack<->query attention, previously identified important encoder heads (Exp. 11)",
                 fontsize=11)
    fig.savefig(plots_dir / "plot5_important_head_attention.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plots_dir / 'plot5_important_head_attention.png'}")


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    agg_dir = outputs_base / "aggregates"
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[04_make_plots] Plot 1 -- main causal heatmap")
    plot1_main_causal_heatmap(agg_dir, plots_dir)
    print("[04_make_plots] Plot 2 -- structural causal heatmap")
    plot2_structural_causal_heatmap(agg_dir, plots_dir)
    print("[04_make_plots] Plot 3 -- attack -> query attention")
    plot3_attack_to_query_attention(agg_dir, plots_dir)
    print("[04_make_plots] Plot 4 -- query -> attack attention")
    plot4_query_to_attack_attention(agg_dir, plots_dir)
    print("[04_make_plots] Plot 5 -- important-head attention")
    plot5_important_head_attention(agg_dir, plots_dir)


if __name__ == "__main__":
    main()
