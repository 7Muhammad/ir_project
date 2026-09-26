#!/usr/bin/env python3
"""
scripts/09_make_plots.py
===========================
Figure A — Recovery vs clean damage (one point per candidate; encoder
           points use each head's best validation-selected position
           condition).
Figure B — Encoder position-mask comparison for the 18 encoder heads.
Figure C — OOD generalization: IID recovery vs unseen-token / unseen-
           position / unseen-repetition recovery.

Reads scripts/08_aggregate.py's consolidated tables. Does not re-select or
reorder results based on test performance (task spec section 15, explicit
instruction) — Figure A's encoder-condition choice comes from VALIDATION
(encoder_best_condition.csv), never from test.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from exp14lib.run_utils import load_config, resolve_cfg_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make Experiment 14 analysis figures.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def _read_csv_or_empty(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def plot_a_recovery_vs_damage(recovery: pd.DataFrame, clean: pd.DataFrame, best_cond: pd.DataFrame, out_path: pathlib.Path) -> None:
    iid_recov = recovery[(recovery["protocol"] == "iid") & (recovery["scale_role"].astype(str).str.contains("selected"))]
    iid_clean = clean[(clean["protocol"] == "iid") & (clean["scale_role"].astype(str).str.contains("selected"))]

    fig, ax = plt.subplots(figsize=(7, 6))
    for side, color, marker in [("decoder", "#1f77b4", "o"), ("encoder", "#d62728", "^")]:
        if side == "encoder" and len(best_cond):
            keep = set(zip(best_cond["layer"], best_cond["head_idx"], best_cond["best_condition"]))
            rows = iid_recov[iid_recov["side"] == "encoder"]
            rows = rows[[
                (l, h, c) in keep for l, h, c in zip(rows["layer"], rows["head_idx"], rows["condition"])
            ]]
        else:
            rows = iid_recov[iid_recov["side"] == side]
        if not len(rows):
            continue
        merged = rows.merge(
            iid_clean[iid_clean["side"] == side][["layer", "head_idx", "condition", "mean_abs_change"]],
            on=["layer", "head_idx", "condition"], how="inner",
        )
        ax.scatter(merged["mean_abs_change"], merged["mean_recovery"], c=color, marker=marker, label=side, alpha=0.8)

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axhline(1, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Clean input: mean |score change|")
    ax.set_ylabel("Held-out attack recovery (test, selected scale)")
    ax.set_title("Figure A — Recovery vs. clean-input damage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_b_encoder_conditions(recovery: pd.DataFrame, out_path: pathlib.Path) -> None:
    rows = recovery[
        (recovery["protocol"] == "iid") & (recovery["side"] == "encoder")
        & (recovery["scale_role"].astype(str).str.contains("selected"))
    ]
    if not len(rows):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no encoder rows", ha="center")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    pivot = rows.pivot_table(index="condition", columns=["layer", "head_idx"], values="mean_recovery")
    condition_order = ["document", "query", "template", "query_document", "all_valid"]
    pivot = pivot.reindex(condition_order)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * pivot.shape[1]), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"L{l}H{h}" for l, h in pivot.columns], rotation=90, fontsize=7)
    ax.set_title("Figure B — Encoder position-mask comparison (mean recovery)")
    fig.colorbar(im, ax=ax, label="mean recovery")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_c_ood_generalization(recovery: pd.DataFrame, out_path: pathlib.Path) -> None:
    iid = recovery[(recovery["protocol"] == "iid") & (recovery["scale_role"].astype(str).str.contains("selected"))]
    iid_by_cand = iid.groupby(["side", "layer", "head_idx", "condition"])["mean_recovery"].mean().rename("iid_recovery")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, ood_factor in zip(axes, ["ood_token", "ood_position", "ood_repetitions"]):
        ood = recovery[recovery["protocol"] == ood_factor]
        if not len(ood):
            ax.set_title(f"{ood_factor} (no data)")
            continue
        ood_by_cand = ood.groupby(["side", "layer", "head_idx", "condition"])["mean_recovery"].mean().rename("ood_recovery")
        merged = pd.concat([iid_by_cand, ood_by_cand], axis=1, join="inner").reset_index()
        colors = merged["side"].map({"decoder": "#1f77b4", "encoder": "#d62728"})
        ax.scatter(merged["iid_recovery"], merged["ood_recovery"], c=colors, alpha=0.8)
        lims = [-1, 2]
        ax.plot(lims, lims, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("IID held-out recovery")
        ax.set_title(ood_factor)
    axes[0].set_ylabel("OOD recovery")
    fig.suptitle("Figure C — IID vs. attack-OOD generalization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    recovery = _read_csv_or_empty(outputs_base / "full_recovery_summary.csv")
    clean = _read_csv_or_empty(outputs_base / "full_clean_damage_summary.csv")
    best_cond = _read_csv_or_empty(outputs_base / "encoder_best_condition.csv")

    plot_a_recovery_vs_damage(recovery, clean, best_cond, plots_dir / "figure_a_recovery_vs_damage.png")
    plot_b_encoder_conditions(recovery, plots_dir / "figure_b_encoder_conditions.png")
    plot_c_ood_generalization(recovery, plots_dir / "figure_c_ood_generalization.png")

    print(f"[09_make_plots] wrote 3 figures to {plots_dir}")


if __name__ == "__main__":
    main()
