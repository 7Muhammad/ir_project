#!/usr/bin/env python3
"""
scripts/05_aggregate_and_plot.py
===================================
Experiment 11 -- final aggregation, figures (Step 1 heatmap/table/scatter,
Step 2 box plot), the combined summary figure (Step 1 heatmap + Step 4
classification + Step 5 hub verdict), and the written summary.

Reads (never modifies): outputs/canonical/results.csv (Step 1),
outputs/sweep/attacks/*/results.csv (Step 2), outputs/additivity_check.txt
(Step 3), outputs/layer9_crossref/classification.csv (Step 4),
outputs/hub_analysis/{aggregated.csv,verdict.txt} (Step 5).
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
from matplotlib.colors import TwoSlopeNorm

from exp11lib.run_utils import load_config, resolve_cfg_path

LATE_LAYERS = [9, 10, 11]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 11 -- aggregate, plots, combined figure, summary.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def head_label(row) -> str:
    return f"L{int(row['layer'])}H{int(row['head_idx' if 'head_idx' in row else 'head'])}"


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: pathlib.Path) -> None:
    vmax = max(float(np.nanmax(np.abs(pivot.values))), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("head index"); ax.set_ylabel("encoder layer")
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="combined_effect")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    threshold = cfg["flag_threshold"]

    canonical_csv = outputs_base / "canonical" / "results.csv"
    if not canonical_csv.exists():
        sys.exit(f"[05] {canonical_csv} not found -- run scripts/01 first.")
    canonical_df = pd.read_csv(canonical_csv)

    head_summary = canonical_df.groupby(["layer", "head_idx"]).agg(
        n=("qid", "size"),
        fwd_effect_mean=("fwd_effect", "mean"),
        rev_effect_mean=("rev_effect", "mean"),
        combined_effect_mean=("combined_effect", "mean"),
        score_drop_zero_mean=("score_drop_zero", "mean"),
        score_drop_mean_mean=("score_drop_mean", "mean"),
    ).reset_index()
    head_summary.to_csv(outputs_base / "head_summary_canonical.csv", index=False)

    # ---- Plot 1: heatmap (full range) ----
    pivot = head_summary.pivot(index="layer", columns="head_idx", values="combined_effect_mean").sort_index()
    plot_heatmap(pivot, "Per-head encoder self-attn combined effect\n(canonical relevant_start_5, n=100)",
                 plots_dir / "exp11_head_heatmap_canonical.png")

    # ---- Plot 1b: zoomed layers 9-11 heatmap ----
    # Early layers (esp. L0H7: combined_effect ~ -0.9) show the SAME early-layer
    # instability Experiment 6 documented for position-restricted patching (a
    # single-head patch leaves the residual stream internally inconsistent for
    # 11 more layers to process/invert) -- these outliers wash out the full-range
    # color scale, so a separate zoomed view is needed to see the late-layer signal.
    pivot_late = pivot.loc[pivot.index.isin(LATE_LAYERS)]
    plot_heatmap(pivot_late, "Per-head encoder self-attn combined effect -- layers 9-11 only\n"
                 "(canonical relevant_start_5, n=100; early layers show Exp.-6-style instability, see report)",
                 plots_dir / "exp11_head_heatmap_canonical_late_layers.png")

    # ---- Table: top 10 heads ----
    top10 = head_summary.nlargest(10, "combined_effect_mean").copy()
    top10["head"] = top10.apply(head_label, axis=1)
    top10_cols = ["head", "fwd_effect_mean", "rev_effect_mean", "combined_effect_mean",
                  "score_drop_zero_mean", "score_drop_mean_mean"]
    top10[top10_cols].to_csv(outputs_base / "top10_heads.csv", index=False)
    print("\n[05] Top 10 heads by combined_effect_mean:")
    print(top10[top10_cols].round(4).to_string(index=False))

    # ---- Plot 2: zero vs mean ablation scatter ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(head_summary["score_drop_zero_mean"], head_summary["score_drop_mean_mean"],
               s=28, alpha=0.75, color="#1f77b4", edgecolors="white", linewidths=0.4)
    lims = np.array([ax.get_xlim(), ax.get_ylim()])
    lo, hi = lims.min(), lims.max()
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1, zorder=0,
            label="drop(zero) = drop(mean)")
    ax.axhline(0, color="#cccccc", linewidth=0.8); ax.axvline(0, color="#cccccc", linewidth=0.8)
    for _, row in top10.iterrows():
        ax.annotate(row["head"], (row["score_drop_zero_mean"], row["score_drop_mean_mean"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, color="#333333")
    ax.set_xlabel("mean score drop -- zero ablation (attack input)")
    ax.set_ylabel("mean score drop -- mean/control ablation (attack input)")
    ax.set_title("Zero vs mean ablation per encoder head\ndiagonal = attack-specific; below = generally important", fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(plots_dir / "exp11_zero_vs_mean_ablation_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plots_dir/'exp11_zero_vs_mean_ablation_scatter.png'}")

    # ---- Plot 3: box plot across 105 attacks ----
    sweep_csvs = sorted((outputs_base / "sweep" / "attacks").glob("*/results.csv"))
    have_sweep = bool(sweep_csvs)
    if have_sweep:
        sweep_df = pd.concat([pd.read_csv(p) for p in sweep_csvs], ignore_index=True)
        n_attacks = sweep_df["attack_name"].nunique()
        head_attack_summary = sweep_df.groupby(["layer", "head_idx", "attack_name"]).agg(
            combined_effect_mean=("combined_effect", "mean"),
        ).reset_index()
        head_attack_summary.to_csv(outputs_base / "head_attack_summary_sweep.csv", index=False)

        ha = head_attack_summary.copy()
        ha["head"] = ha.apply(head_label, axis=1)
        med = ha.groupby("head")["combined_effect_mean"].median().sort_values(ascending=False)
        top_heads = med.head(20).index.tolist()
        data = [ha.loc[ha["head"] == h, "combined_effect_mean"].dropna().values for h in top_heads]

        fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(top_heads))))
        ax.boxplot(data[::-1], vert=False, patch_artist=True, tick_labels=top_heads[::-1],
                   boxprops=dict(facecolor="#9ecae1", alpha=0.6),
                   medianprops=dict(color="#222222"), flierprops=dict(marker=".", markersize=3, alpha=0.5))
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel("per-attack mean combined_effect")
        ax.set_title(f"Per-head combined effect across {n_attacks} attacks -- top {len(top_heads)} heads", fontsize=11)
        fig.savefig(plots_dir / "exp11_per_head_effect_across_attacks.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {plots_dir/'exp11_per_head_effect_across_attacks.png'}")

        # "flagged on X of 105 attacks" per head
        late = head_attack_summary[head_attack_summary["layer"].isin(LATE_LAYERS)]
        peak_per_attack = late.groupby(["layer", "head_idx", "attack_name"])["combined_effect_mean"].max()
        exceed = (peak_per_attack > threshold).groupby(["layer", "head_idx"]).sum()
        exceed_df = exceed.reset_index(name="n_attacks_exceeding").rename(columns={"head_idx": "head"})
        exceed_df["n_attacks_total"] = n_attacks
        exceed_df.to_csv(outputs_base / "head_exceed_counts.csv", index=False)
    else:
        print("  SKIP box plot / exceed counts -- no sweep results found (run scripts/02 first).")
        exceed_df = pd.DataFrame()
        n_attacks = 0

    # ---- Step 4 / Step 5 read-only ----
    classification_path = outputs_base / "layer9_crossref" / "classification.csv"
    classification_df = pd.read_csv(classification_path) if classification_path.exists() else None
    verdict_path = outputs_base / "hub_analysis" / "verdict.txt"
    hub_verdict = verdict_path.read_text(encoding="utf-8").strip() if verdict_path.exists() else \
        "(Step 5 not yet run -- scripts/04_attack_token_hub_analysis.py)"
    additivity_path = outputs_base / "additivity_check.txt"
    additivity_text = additivity_path.read_text(encoding="utf-8").strip() if additivity_path.exists() else \
        "(Step 3 not yet run -- part of scripts/02_encoder_head_patch_full_sweep.py)"

    # ---- Combined summary figure: heatmap + classification (layer 9) + hub verdict ----
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9])
    ax_hm, ax_tbl, ax_txt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    vmax = max(float(np.nanmax(np.abs(pivot.values))), 1e-6)
    im = ax_hm.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax_hm.set_xticks(range(pivot.shape[1])); ax_hm.set_xticklabels(pivot.columns)
    ax_hm.set_yticks(range(pivot.shape[0])); ax_hm.set_yticklabels(pivot.index)
    ax_hm.set_xlabel("head"); ax_hm.set_ylabel("layer")
    ax_hm.set_title("Step 1: combined_effect (layer x head)")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    ax_tbl.axis("off")
    if classification_df is not None:
        VERDICT_SHORT = {
            "carries both mass and causal effect": "mass + effect",
            "high mass, low causal effect (descriptive-only, not load-bearing)": "mass only",
            "high causal effect, unremarkable mass (mechanism isn't visible in raw attention)": "effect only",
            "uninvolved": "uninvolved",
        }
        l9 = classification_df[classification_df["layer"] == 9].nlargest(10, "combined_effect").copy()
        l9["verdict"] = l9["verdict"].map(VERDICT_SHORT).fillna(l9["verdict"])
        cols = ["head", "attention_d_to_q", "combined_effect", "verdict"]
        cell_text = l9[cols].round(3).astype(str).values
        tbl = ax_tbl.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        tbl.auto_set_column_width(col=list(range(len(cols))))
        tbl.scale(1, 1.8)
        ax_tbl.set_title("Step 4: layer-9 mass x causal-effect verdict\n(full labels in classification.csv)", pad=20)
    else:
        ax_tbl.text(0.5, 0.5, "classification.csv not found\n(run scripts/03 first)", ha="center", va="center")

    ax_txt.axis("off")
    ax_txt.text(0.02, 0.98, "Step 5: attack-token-as-hub verdict\n\n" + hub_verdict,
                ha="left", va="top", fontsize=8, wrap=True, transform=ax_txt.transAxes)

    fig.suptitle("Experiment 11: encoder per-head causal patching -- summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots_dir / "exp11_encoder_head_summary_figure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plots_dir/'exp11_encoder_head_summary_figure.png'}")

    # ---- Written summary ----
    n_flagged = int((head_summary["combined_effect_mean"] > threshold).sum())
    late_summary = head_summary[head_summary["layer"].isin(LATE_LAYERS)].nlargest(5, "combined_effect_mean")
    top_names = ", ".join(head_label(r) for _, r in late_summary.iterrows())

    lines = ["# Experiment 11: encoder per-head causal patching -- summary\n"]
    lines.append(f"Of 144 head-slots, {n_flagged} exceed the combined_effect > {threshold} threshold on the "
                 f"canonical attack (n=100). Top late-layer (9-11) heads: {top_names}.")
    if have_sweep and not exceed_df.empty:
        top_exceed = exceed_df.sort_values("n_attacks_exceeding", ascending=False).head(5)
        rob = "; ".join(f"L{int(r.layer)}H{int(r.head)}: {int(r.n_attacks_exceeding)}/{n_attacks}"
                        for r in top_exceed.itertuples())
        lines.append(f"Robustness across the full {n_attacks}-attack sweep (layers 9-11): {rob}.")
    lines.append("")
    lines.append("**Step 3 additivity:**\n" + additivity_text)
    lines.append("")
    lines.append("**Step 5 attack-token-as-hub:**\n" + hub_verdict)
    lines.append("")
    if classification_df is not None:
        n_both = int((classification_df["verdict"] == "carries both mass and causal effect").sum())
        lines.append(f"**Step 4 overlap:** {n_both} of {len(classification_df)} (layer, head) cells at layers "
                     f"8-11 carry both elevated document->query attention mass and a causal effect above threshold "
                     f"-- see outputs/layer9_crossref/classification.csv for the full per-head breakdown.")
    summary_text = "\n".join(lines) + "\n"
    (outputs_base / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(f"\n[05] wrote {outputs_base/'SUMMARY.md'}\n")
    print(summary_text)


if __name__ == "__main__":
    main()
