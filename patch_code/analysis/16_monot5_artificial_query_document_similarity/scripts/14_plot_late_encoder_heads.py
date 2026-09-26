"""
scripts/14_plot_late_encoder_heads.py
======================================
All 36 encoder heads of layers 9-11: absolute head-level query/document
similarity for the four populations (no model; reads 13_late_encoder_heads/).

Aggregation = stage 10 (imported): genuine qrel 2/3 and qrel 0 -> mean within
query, equal query weight; attacked successful (example-level delta_score > 0)
and unsuccessful (<= 0) -> mean within attack configuration, equal attack
weight; +-1 SE across queries / attack configurations.

Descriptive per-head differences (no testing):
  success_gap           = successful_attack_mean - unsuccessful_attack_mean
  genuine_gap           = genuine_relevant_mean  - genuine_nonrelevant_mean
  successful_vs_genuine = successful_attack_mean - genuine_relevant_mean
  previously_important_head = head is in the canonical 18 (Exp 13 encoder_senders.json)

Outputs: 10_head_analysis/late_encoder_heads_L9_L11.csv
         plots/fig_heads_encoder_all_L9_L11_four_populations.png
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import all_encoder_heads, encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, stage_argparser  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_10_plot", EXP_DIR / "scripts" / "10_plot_head_similarity.py")
P10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P10)   # aggregate(), draw(), GROUPS, YLABEL; P10.P5 = stage-05 style

LAYERS = [9, 10, 11]
DT = P10.DT


def main():
    args = stage_argparser("Exp 16 stage 14: all encoder heads L9-L11, four populations").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / "13_late_encoder_heads"
    if not is_already_successful(st, ["qrel_late_encoder_heads.csv.gz", "attack_late_encoder_heads.csv.gz"]):
        raise FileNotFoundError(f"run stage 13 first ({st})")
    q = pd.read_csv(st / "qrel_late_encoder_heads.csv.gz", dtype=DT)
    a = pd.read_csv(st / "attack_late_encoder_heads.csv.gz", dtype=DT)
    heads = all_encoder_heads(LAYERS)
    cols = [f"enc_{h.label}" for h in heads]
    if len(heads) != 36 or [c for c in q.columns if c.startswith("enc_")] != cols or \
            [c for c in a.columns if c.startswith("enc_")] != cols:
        raise RuntimeError("expected exactly the 36 encoder heads L9H0..L11H11, in order")
    if not q[q.relevance_group == "relevant"].qrel_grade.isin([2, 3]).all() or \
            not q[q.relevance_group == "nonrelevant"].qrel_grade.isin([0]).all():
        raise RuntimeError("qrel groups are not 2/3 vs 0")
    if not np.allclose(a.delta_score, a.score_attack - a.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")

    t, _, _ = P10.aggregate(q, a, heads, "enc")
    t = t.rename(columns={"n_successful_attacks": "n_successful_attack_configs",
                          "n_unsuccessful_attacks": "n_unsuccessful_attack_configs"})
    t["success_gap"] = t.successful_attack_mean - t.unsuccessful_attack_mean
    t["genuine_gap"] = t.genuine_relevant_mean - t.genuine_nonrelevant_mean
    t["successful_vs_genuine"] = t.successful_attack_mean - t.genuine_relevant_mean
    canon = {h.label for h in encoder_heads()}
    t["previously_important_head"] = t.head_label.isin(canon)
    t["both_gaps_positive"] = (t.genuine_gap > 0) & (t.success_gap > 0)
    first = ["layer", "head", "head_label"]
    t = t[first + [c for c in t.columns if c not in first]]
    csv_path = out / "10_head_analysis" / "late_encoder_heads_L9_L11.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    P10.draw(ax, t, P10.GROUPS)
    for b in (12, 24):                                           # layer 9 | 10 | 11
        ax.axvline(b - 0.5, color=P10.P5.INK2, lw=1.0, ls=":", zorder=1)
    for i, L in enumerate(LAYERS):
        ax.text(12 * i + 5.5, 1.01, f"layer {L}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, color=P10.P5.INK2)
    for lab in ax.get_xticklabels():                              # canonical heads: bold tick label
        if lab.get_text() in canon:
            lab.set_fontweight("bold")
    ax.set_xlabel("encoder head (bold = one of the 18 canonical important heads)")
    ax.set_ylabel("query-document similarity\ncos(mean query head output, mean document head output)")
    n = t.iloc[0]
    ax.set_title("Query-document similarity across all encoder heads in layers 9–11\n"
                 f"{int(n.n_queries)} queries; successful {int(n.n_successful_examples):,} ex. / "
                 f"{int(n.n_successful_attack_configs)} attacks; unsuccessful {int(n.n_unsuccessful_examples):,} ex. / "
                 f"{int(n.n_unsuccessful_attack_configs)} attacks; ±1 SE", loc="left", fontsize=9, pad=18)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    png = out / "plots" / "fig_heads_encoder_all_L9_L11_four_populations.png"
    P10.P5._save(fig, png)
    print(f"[14] {len(t)} heads -> {png}\n  -> {csv_path}")


if __name__ == "__main__":
    main()
