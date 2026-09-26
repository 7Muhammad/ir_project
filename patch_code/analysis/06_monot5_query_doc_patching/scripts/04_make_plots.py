#!/usr/bin/env python3
"""
scripts/04_make_plots.py
=========================
Experiment 6 figures. Reads the DERIVED summaries written by
scripts/03_aggregate.py; never touches per-example data.

Plots (written to outputs/plots/):

1. attention_mass_clean_vs_attack.png
     Normalized q->d and d->q attention mass per encoder layer, clean vs
     attacked input (canonical run — deepest, most stable estimate).

2. effect_by_condition.png
     Combined patching effect per layer, one line per condition
     (doc_only/query_only/both), encoder_self_attn and decoder_cross_attn
     side by side (canonical run).

3. query_only_effect_across_attacks.png
     Distribution (box plot) of query_only combined effect across all 105
     attacks, per layer, encoder and decoder cross-attention separately
     (grid run). The key diagnostic: does query-only patching move the
     score at all, and if so, at which layer(s)?
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
import pandas as pd

from exp6lib.run_utils import load_config, resolve_cfg_path

REGION_LABELS = {"encoder_self_attn": "Encoder Self-Attention", "decoder_cross_attn": "Decoder Cross-Attention"}
CONDITION_COLORS = {"doc_only": "#1f77b4", "query_only": "#d62728", "both": "#2ca02c"}


def _load(path: pathlib.Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/02_run_grid.py then scripts/03_aggregate.py first ({what})."
        )
    return pd.read_csv(path)


def plot_attention_mass(attn_summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    layers = sorted(attn_summary["layer"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, direction in zip(axes, ["q_to_d", "d_to_q"]):
        clean_col = f"attention_{direction}_clean_mean"
        attack_col = f"attention_{direction}_attack_mean"
        ax.plot(layers, attn_summary.set_index("layer").loc[layers, clean_col],
                marker="o", label="clean", color="#1f77b4")
        ax.plot(layers, attn_summary.set_index("layer").loc[layers, attack_col],
                marker="o", label="attack", color="#d62728")
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1, label="uniform baseline")
        ax.set_xlabel("encoder layer")
        ax.set_title(f"{'Query -> Document' if direction=='q_to_d' else 'Document -> Query'}")
    axes[0].set_ylabel("normalized attention mass")
    axes[0].legend(fontsize=9)
    fig.suptitle("Normalized cross-region attention mass — clean vs attacked input", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_effect_by_condition(layer_cond: pd.DataFrame, out_path: pathlib.Path) -> None:
    regions = [r for r in ["encoder_self_attn", "decoder_cross_attn"] if r in layer_cond["region"].unique()]
    fig, axes = plt.subplots(1, len(regions), figsize=(7 * len(regions), 5), sharey=True)
    axes = axes if len(regions) > 1 else [axes]
    for ax, region in zip(axes, regions):
        sub = layer_cond[layer_cond["region"] == region]
        for condition in ["doc_only", "query_only", "both"]:
            cs = sub[sub["condition"] == condition].sort_values("layer")
            ax.plot(cs["layer"], cs["combined_effect_mean"], marker="o",
                     label=condition, color=CONDITION_COLORS[condition])
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel("layer")
        ax.set_title(REGION_LABELS[region])
    axes[0].set_ylabel("combined effect")
    axes[0].legend(fontsize=9)
    fig.suptitle("Positional patching effect by condition, per layer\n"
                 "doc-only expected to trivially reproduce the attack; query-only is the key result",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_query_only_across_attacks(layer_cond_attack: pd.DataFrame, out_path: pathlib.Path) -> None:
    q = layer_cond_attack[layer_cond_attack["condition"] == "query_only"]
    regions = [r for r in ["encoder_self_attn", "decoder_cross_attn"] if r in q["region"].unique()]
    fig, axes = plt.subplots(1, len(regions), figsize=(8 * len(regions), 6), sharey=True)
    axes = axes if len(regions) > 1 else [axes]
    for ax, region in zip(axes, regions):
        sub = q[q["region"] == region]
        layers = sorted(sub["layer"].unique())
        data = [sub.loc[sub["layer"] == l, "combined_effect_mean"].dropna().values for l in layers]
        ax.boxplot(data, tick_labels=layers, patch_artist=True,
                    boxprops=dict(facecolor=CONDITION_COLORS["query_only"], alpha=0.5),
                    medianprops=dict(color="#222222"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel("layer")
        ax.set_title(REGION_LABELS[region])
    axes[0].set_ylabel("query_only combined effect (per-attack mean)")
    fig.suptitle("Query-only patching effect across all attacks, per layer\n"
                 "near-zero everywhere = no document->query cross-contamination in the encoder",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 6 figures.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    args = p.parse_args()

    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[04_make_plots] Plots → {plots_dir}")

    attn_summary = _load(outputs_base / "canonical" / "aggregated" / "attention_summary.csv", "attention mass")
    plot_attention_mass(attn_summary, plots_dir / "attention_mass_clean_vs_attack.png")

    layer_cond = _load(outputs_base / "canonical" / "aggregated" / "layer_condition_summary.csv", "effect by condition")
    plot_effect_by_condition(layer_cond, plots_dir / "effect_by_condition.png")

    layer_cond_attack = _load(outputs_base / "grid" / "aggregated" / "layer_condition_attack_summary.csv",
                               "query-only across attacks")
    plot_query_only_across_attacks(layer_cond_attack, plots_dir / "query_only_effect_across_attacks.png")

    print("[04_make_plots] Done.")


if __name__ == "__main__":
    main()
