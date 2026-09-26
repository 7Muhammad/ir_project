#!/usr/bin/env python3
"""
scripts/09_template_aggregate_and_plot.py
=============================================
Experiment 6 extension -- final aggregation, Step 3 additivity check,
figures, and the written sink/signal/uninvolved verdict per template
position.

Reads (never modifies):
  outputs/template_tokens/causal_patch/results.csv       (Step 1, canonical, n=100)
  outputs/template_tokens/full_sweep/attacks/*/results.csv (Step 4, 105 attacks)
  outputs/template_tokens/attention_mass/classification.csv (Step 2)
  ../outputs/canonical/aggregated/layer_condition_summary.csv (existing exp6
      §4.4 output -- read-only, for the Step 3 additivity comparison)

Writes:
  outputs/template_tokens/aggregated/canonical_layer_position_summary.csv
  outputs/template_tokens/aggregated/sweep_layer_position_summary.csv
  outputs/template_tokens/aggregated/sweep_layer_position_attack_summary.csv
  outputs/template_tokens/aggregated/additivity_check.txt
  outputs/template_tokens/plots/template_position_heatmap.png       (Step 1 heatmap)
  outputs/template_tokens/plots/template_position_peak_bar.png      (Step 1 bar chart)
  outputs/template_tokens/plots/template_position_sweep_boxplots.png (Step 4 box plots)
  outputs/template_tokens/plots/template_position_summary_figure.png (heatmap + classification table)
  outputs/template_tokens/SUMMARY.md                                (written verdict)
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

from exp6lib.run_utils import load_config, resolve_cfg_path
from exp6lib.template_positions import TEMPLATE_POSITION_NAMES

LATE_LAYERS = [9, 10, 11]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 extension -- aggregate, additivity check, plots, summary.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "template_tokens.yaml"))
    return p.parse_args()


def peak_by_position(df: pd.DataFrame, group_keys: list) -> pd.Series:
    """mean per (position, *group_keys, layer) first, then max over LATE_LAYERS."""
    late = df[df["layer"].isin(LATE_LAYERS)]
    group = ["template_position", *group_keys]
    by_layer = late.groupby([*group, "layer"])["combined_effect"].mean()
    return by_layer.groupby(group).max()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    agg_dir = outputs_base / "aggregated"
    plots_dir = outputs_base / "plots"
    agg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    thresholds = cfg["flag_thresholds"]
    primary_t = thresholds[0]

    # ---- Load Step 1 (canonical) ----
    canonical_csv = outputs_base / "causal_patch" / "results.csv"
    if not canonical_csv.exists():
        sys.exit(f"[09] {canonical_csv} not found -- run scripts/06_template_causal_patch.py first.")
    canonical_df = pd.read_csv(canonical_csv)

    canonical_summary = canonical_df.groupby(["template_position", "layer"]).agg(
        n=("qid", "size"),
        combined_effect_mean=("combined_effect", "mean"),
        combined_effect_std=("combined_effect", "std"),
        fwd_effect_mean=("fwd_effect", "mean"),
        rev_effect_mean=("rev_effect", "mean"),
    ).reset_index()
    canonical_summary.to_csv(agg_dir / "canonical_layer_position_summary.csv", index=False)

    peak_canonical = peak_by_position(canonical_df, [])

    # ---- Load Step 4 (full sweep), if present ----
    sweep_dir = outputs_base / "full_sweep" / "attacks"
    sweep_csvs = sorted(sweep_dir.glob("*/results.csv")) if sweep_dir.exists() else []
    have_sweep = bool(sweep_csvs)
    if have_sweep:
        sweep_df = pd.concat([pd.read_csv(p) for p in sweep_csvs], ignore_index=True)
        n_attacks = sweep_df["attack_name"].nunique()
        print(f"[09] loaded {len(sweep_df)} sweep rows from {n_attacks} attack(s)")

        sweep_by_attack = sweep_df.groupby(["template_position", "attack_name", "layer"]).agg(
            n=("qid", "size"), combined_effect_mean=("combined_effect", "mean"),
        ).reset_index()
        sweep_by_attack.to_csv(agg_dir / "sweep_layer_position_attack_summary.csv", index=False)

        sweep_pooled = sweep_df.groupby(["template_position", "layer"]).agg(
            n=("qid", "size"), combined_effect_mean=("combined_effect", "mean"),
            combined_effect_std=("combined_effect", "std"),
        ).reset_index()
        sweep_pooled.to_csv(agg_dir / "sweep_layer_position_summary.csv", index=False)

        peak_per_attack = peak_by_position(sweep_df, ["attack_name"])
        exceed_counts = {}
        for name in TEMPLATE_POSITION_NAMES:
            vals = peak_per_attack.loc[name] if name in peak_per_attack.index.get_level_values(0) else pd.Series(dtype=float)
            exceed_counts[name] = int((vals > primary_t).sum())
    else:
        print("[09] WARNING: no full-sweep results found -- skipping Step 4 outputs (run scripts/08 first).")
        sweep_df = pd.DataFrame()
        n_attacks = 0
        exceed_counts = {name: None for name in TEMPLATE_POSITION_NAMES}

    # ---- Step 3: additivity check ----
    exp6_summary_path = outputs_base.parent / "canonical" / "aggregated" / "layer_condition_summary.csv"
    additivity_lines = []
    if exp6_summary_path.exists():
        exp6_summary = pd.read_csv(exp6_summary_path)
        both_enc = exp6_summary[(exp6_summary["region"] == "encoder_self_attn") & (exp6_summary["condition"] == "both")]
        both_enc_late = both_enc[both_enc["layer"].isin(LATE_LAYERS)]
        whole_span_peak = float(both_enc_late.set_index("layer")["combined_effect_mean"].max())

        sum_individual_peaks = float(sum(peak_canonical.get(name, 0.0) for name in TEMPLATE_POSITION_NAMES))
        ratio = sum_individual_peaks / whole_span_peak if whole_span_peak else float("nan")
        line = (f"Sum of the 7 template positions' individual peak combined_effect (layers 9-11) = "
                f"{sum_individual_peaks:.4f}; exp6's existing whole-span 'both' (query+document) peak "
                f"combined_effect at the same layers = {whole_span_peak:.4f} "
                f"(ratio {ratio:.2f}x) -- "
                f"{'roughly additive (positions act close to independently)' if 0.7 <= ratio <= 1.3 else 'diverges from additive (positions interact, or a modelling non-linearity dominates)'}.")
        additivity_lines.append(line)
    else:
        line = f"[09] Could not run additivity check -- {exp6_summary_path} not found."
        additivity_lines.append(line)
    print("[09] " + additivity_lines[-1])
    (agg_dir / "additivity_check.txt").write_text("\n".join(additivity_lines) + "\n", encoding="utf-8")

    # ---- Load Step 2 classification ----
    classification_csv = outputs_base / "attention_mass" / "classification.csv"
    classification_df = pd.read_csv(classification_csv) if classification_csv.exists() else None

    # ---- Plot 1: heatmap (Step 1) ----
    pivot = canonical_summary.pivot(index="template_position", columns="layer", values="combined_effect_mean")
    pivot = pivot.reindex(TEMPLATE_POSITION_NAMES)
    vmax = float(np.nanmax(np.abs(pivot.values)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("encoder layer")
    ax.set_title("Template-position single-position patch: mean combined_effect\n"
                 "(canonical attack relevant_start_5, n=100)")
    fig.colorbar(im, ax=ax, label="combined_effect")
    fig.tight_layout()
    fig.savefig(plots_dir / "template_position_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] wrote {plots_dir/'template_position_heatmap.png'}")

    # ---- Plot 2: bar chart of peak effect (Step 1) ----
    peak_sorted = peak_canonical.reindex(TEMPLATE_POSITION_NAMES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d62728" if v > primary_t else "#1f77b4" for v in peak_sorted.values]
    ax.bar(peak_sorted.index, peak_sorted.values, color=colors)
    ax.axhline(primary_t, color="#888888", linestyle="--", linewidth=1, label=f"threshold={primary_t}")
    ax.set_ylabel("peak combined_effect (max, layers 9-11)")
    ax.set_title("Template positions, sorted by peak causal effect")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "template_position_peak_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] wrote {plots_dir/'template_position_peak_bar.png'}")

    # ---- Plot 3: box plots across full sweep (Step 4) ----
    if have_sweep:
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
        axes = axes.flatten()
        for ax, name in zip(axes, TEMPLATE_POSITION_NAMES):
            sub = sweep_by_attack[sweep_by_attack["template_position"] == name]
            layers = sorted(sub["layer"].unique())
            data = [sub.loc[sub["layer"] == l, "combined_effect_mean"].dropna().values for l in layers]
            ax.boxplot(data, tick_labels=layers, patch_artist=True,
                        boxprops=dict(facecolor="#9ecae1", alpha=0.6),
                        medianprops=dict(color="#222222"),
                        flierprops=dict(marker=".", markersize=3, alpha=0.5))
            ax.axhline(0, color="#888888", linewidth=0.8)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("layer")
        axes[0].set_ylabel("combined_effect (per-attack mean)")
        for ax in axes[len(TEMPLATE_POSITION_NAMES):]:
            ax.axis("off")
        fig.suptitle(f"Template-position combined_effect across all {n_attacks} attacks, per layer", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(plots_dir / "template_position_sweep_boxplots.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[09] wrote {plots_dir/'template_position_sweep_boxplots.png'}")

    # ---- Plot 4: combined summary figure (heatmap + classification table) ----
    fig, (ax_hm, ax_tbl) = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax_hm.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_hm.set_xticks(range(pivot.shape[1])); ax_hm.set_xticklabels(pivot.columns)
    ax_hm.set_yticks(range(pivot.shape[0])); ax_hm.set_yticklabels(pivot.index)
    ax_hm.set_xlabel("encoder layer")
    ax_hm.set_title("Step 1: mean combined_effect per position/layer")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    ax_tbl.axis("off")
    if classification_df is not None:
        table_data = classification_df[["template_position", "peak_e_comb", "peak_incoming_mass",
                                         "peak_outgoing_mass", "classification"]].round(3)
        tbl = ax_tbl.table(cellText=table_data.values, colLabels=table_data.columns,
                            loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        ax_tbl.set_title("Step 2: sink/signal classification", pad=20)
    else:
        ax_tbl.text(0.5, 0.5, "classification.csv not found\n(run scripts/07 first)",
                     ha="center", va="center")
    fig.suptitle("Experiment 6 extension: template-position sink/signal summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plots_dir / "template_position_summary_figure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] wrote {plots_dir/'template_position_summary_figure.png'}")

    # ---- Written summary ----
    lines = ["# Experiment 6 extension: template-token sink/signal analysis -- summary\n"]
    for name in TEMPLATE_POSITION_NAMES:
        peak = peak_canonical.get(name, 0.0)
        flagged = peak > primary_t
        cls = None
        if classification_df is not None:
            row = classification_df[classification_df["template_position"] == name]
            if not row.empty:
                cls = row.iloc[0]["classification"]
        verdict = "SIGNAL" if flagged else "sink/uninvolved"
        n_exceed = exceed_counts.get(name)
        sweep_note = f", exceeds threshold on {n_exceed}/{n_attacks} attacks in the full sweep" if n_exceed is not None else ""
        lines.append(f"- **{name}** ({verdict}{f', classified {cls}' if cls else ''}): "
                     f"peak combined_effect (layers 9-11, canonical attack) = {peak:+.4f}{sweep_note}.")
    lines.append("")
    lines.append(additivity_lines[-1])
    lines.append("")
    any_signal = any(peak_canonical.get(name, 0.0) > primary_t for name in TEMPLATE_POSITION_NAMES)
    if any_signal:
        signal_names = [n for n in TEMPLATE_POSITION_NAMES if peak_canonical.get(n, 0.0) > primary_t]
        lines.append(
            f"**Conclusion**: {', '.join(signal_names)} carry non-trivial causal signal when patched in "
            "isolation at late encoder layers -- the sink hypothesis does NOT hold for these positions, "
            "so exp6's §4.4 query-only/document-only effect sizes (which freeze all 7 template positions "
            "at their control value) are underestimates for whatever fraction of the pathway runs through "
            "these positions, and the report needs a caveat quantifying this."
        )
    else:
        lines.append(
            "**Conclusion**: no template position exceeds the causal-effect threshold at late layers -- "
            "the sink hypothesis holds for all 7, confirming exp6's §4.4 was safe to freeze them; the "
            "existing Limitations-section caveat can be strengthened into a settled null result."
        )
    summary_text = "\n".join(lines) + "\n"
    (outputs_base / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(f"\n[09] wrote {outputs_base/'SUMMARY.md'}\n")
    print(summary_text)


if __name__ == "__main__":
    main()
