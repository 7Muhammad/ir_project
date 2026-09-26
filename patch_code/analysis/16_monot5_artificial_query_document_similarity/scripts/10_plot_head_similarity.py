"""
scripts/10_plot_head_similarity.py
===================================
Head-level counterpart of scripts/06 and 08 (no model; reads 09_heads/ only).

For every canonical important head (18 encoder, 31 decoder cross-attn,
ordered by layer then head), four populations:
  clean, genuinely relevant (qrel 2/3)  } per-query mean, equal query weight, +-1 SE across queries
  clean, non-relevant (qrel 0)          }
  attacked, successful   (example-level delta_score > 0)   } attacked input; mean within attack,
  attacked, unsuccessful (delta_score <= 0)                } equal attack weight, +-1 SE across attacks

Separation columns (descriptive): succ - unsucc, genuine - succ, genuine - unsucc,
and `successful_closer_to_genuine` = |genuine - succ| < |genuine - unsucc|.

Outputs (10_head_analysis/):
  head_similarity_encoder.csv, head_similarity_decoder.csv
  per_attack_head_similarity.csv, per_query_head_similarity.csv  (reproducibility)
plots/:
  fig_heads_encoder_four_populations.png, fig_heads_decoder_four_populations.png,
  fig_heads_decoder_four_populations_by_layer.png (facets, readability),
  fig_heads_{encoder,decoder}_genuine_vs_successful.png (optional two-line version)
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
from exp16lib.heads import decoder_heads, encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, stage_argparser  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_05_plot", EXP_DIR / "scripts" / "05_plot.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
GROUPS = [  # (name, label, colour, marker, linestyle)
    ("genuine_relevant", "clean, genuinely relevant (qrel 2/3)", P5.BLUE, "o", "-"),
    ("genuine_nonrelevant", "clean, non-relevant (qrel 0)", P5.BLUE, "D", "--"),
    ("successful_attack", "attacked, successful (Δscore > 0)", P5.ORANGE, "s", "-"),
    ("unsuccessful_attack", "attacked, unsuccessful (Δscore ≤ 0)", P5.ORANGE, "^", "--"),
]
YLABEL = "cosine(query representation, document representation)"


def _se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def aggregate(q, a, heads, prefix):
    """Returns (table, per_query long, per_attack long)."""
    cols = [f"{prefix}_{h.label}" for h in heads]
    succ = a.delta_score > 0
    pq = q.groupby(["qid", "relevance_group"])[cols].mean().reset_index()
    pa = pd.concat([a[succ].groupby("attack_name")[cols].mean().assign(outcome="successful", n_examples=a[succ].groupby("attack_name").size()),
                    a[~succ].groupby("attack_name")[cols].mean().assign(outcome="unsuccessful", n_examples=a[~succ].groupby("attack_name").size())]
                   ).reset_index()
    rows = []
    for h, c in zip(heads, cols):
        r = {"head_label": h.label, "layer": h.layer, "head": h.head_idx}
        for name, grp in (("genuine_relevant", "relevant"), ("genuine_nonrelevant", "nonrelevant")):
            v = pq[pq.relevance_group == grp][c]
            r[f"{name}_mean"], r[f"{name}_se"] = float(v.mean()), _se(v)
        for name, oc in (("successful_attack", "successful"), ("unsuccessful_attack", "unsuccessful")):
            v = pa[pa.outcome == oc][c]
            r[f"{name}_mean"], r[f"{name}_se"] = float(v.mean()), _se(v)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["n_queries"] = pq.qid.nunique()
    t["n_successful_attacks"] = int((pa.outcome == "successful").sum())
    t["n_unsuccessful_attacks"] = int((pa.outcome == "unsuccessful").sum())
    t["n_successful_examples"] = int(succ.sum())
    t["n_unsuccessful_examples"] = int((~succ).sum())
    t["successful_minus_unsuccessful"] = t.successful_attack_mean - t.unsuccessful_attack_mean
    t["genuine_minus_successful"] = t.genuine_relevant_mean - t.successful_attack_mean
    t["genuine_minus_unsuccessful"] = t.genuine_relevant_mean - t.unsuccessful_attack_mean
    t["relevant_minus_nonrelevant"] = t.genuine_relevant_mean - t.genuine_nonrelevant_mean
    t["successful_closer_to_genuine"] = t.genuine_minus_successful.abs() < t.genuine_minus_unsuccessful.abs()
    return t, pq, pa


def draw(ax, t, groups, x=None):
    x = np.arange(len(t)) if x is None else x
    for name, label, color, marker, ls in groups:
        y, se = t[f"{name}_mean"].values, t[f"{name}_se"].values
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.15, lw=0)
        ax.plot(x, y, color=color, ls=ls, lw=2, label=label, zorder=3)
        ax.scatter(x, y, s=24, color=color, edgecolor="white", linewidth=1, zorder=4, marker=marker)
    ax.set_xticks(x)
    ax.set_xticklabels(t.head_label, rotation=90, fontsize=7)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)


def main_plot(t, title, png, groups, width):
    fig, ax = plt.subplots(figsize=(width, 4.6))
    draw(ax, t, groups)
    for i in range(1, len(t)):                   # thin separators between layers
        if t.layer.iloc[i] != t.layer.iloc[i - 1]:
            ax.axvline(i - 0.5, color=P5.GRID, lw=1.2, zorder=1)
    ax.set_xlabel("important head (ordered by layer, then head)")
    ax.set_ylabel(YLABEL)
    n = t.iloc[0]
    sub = (f"{int(n.n_queries)} queries; successful {int(n.n_successful_examples):,} ex. / {int(n.n_successful_attacks)} attacks; "
           f"unsuccessful {int(n.n_unsuccessful_examples):,} ex. / {int(n.n_unsuccessful_attacks)} attacks; ±1 SE")
    ax.set_title(f"{title}\n{sub}", loc="left", fontsize=9)
    ax.legend(loc="best", fontsize=8)
    P5._save(fig, png)


def facet_plot(t, title, png):
    layers = sorted(t.layer.unique())
    widths = [max(2, (t.layer == L).sum()) for L in layers]
    fig, axes = plt.subplots(1, len(layers), figsize=(14, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, L in zip(np.atleast_1d(axes), layers):
        draw(ax, t[t.layer == L].reset_index(drop=True), GROUPS)
        ax.set_title(f"decoder L{L}", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel(YLABEL)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    P5._save(fig, png)


def main():
    args = stage_argparser("Exp 16 stage 10: head-level similarity tables + plots").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / "09_heads"
    if not is_already_successful(st, ["qrel_heads.csv.gz", "attack_heads.csv.gz"]):
        raise FileNotFoundError(f"run stage 09 first ({st})")
    q = pd.read_csv(st / "qrel_heads.csv.gz", dtype=DT)
    a = pd.read_csv(st / "attack_heads.csv.gz", dtype=DT)
    if not q[q.relevance_group == "relevant"].qrel_grade.isin([2, 3]).all() or \
            not q[q.relevance_group == "nonrelevant"].qrel_grade.isin([0]).all():
        raise RuntimeError("qrel groups are not 2/3 vs 0")
    if not np.allclose(a.delta_score, a.score_attack - a.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")
    tdir = out / "10_head_analysis"
    tdir.mkdir(parents=True, exist_ok=True)
    pdir = out / "plots"
    for prefix, heads, name in (("enc", encoder_heads(), "encoder"), ("dec", decoder_heads(), "decoder")):
        t, pq, pa = aggregate(q, a, heads, prefix)
        t.to_csv(tdir / f"head_similarity_{name}.csv", index=False)
        pq.to_csv(tdir / f"per_query_head_similarity_{name}.csv", index=False)
        pa.to_csv(tdir / f"per_attack_head_similarity_{name}.csv", index=False)
        width = 9 if name == "encoder" else 13
        main_plot(t, f"Head-level query-document similarity: important {name} heads",
                  pdir / f"fig_heads_{name}_four_populations.png", GROUPS, width)
        main_plot(t, f"Head-level query-document similarity: important {name} heads (genuine relevant vs successful attacks)",
                  pdir / f"fig_heads_{name}_genuine_vs_successful.png", [GROUPS[0], GROUPS[2]], width)
        if name == "decoder":
            facet_plot(t, "Head-level query-document similarity: important decoder cross-attention heads, by layer",
                       pdir / "fig_heads_decoder_four_populations_by_layer.png")
        n = t.iloc[0]
        print(f"[10] {name}: {len(t)} heads; {int(n.n_queries)} queries; successful {int(n.n_successful_examples)} ex / "
              f"{int(n.n_successful_attacks)} attacks; unsuccessful {int(n.n_unsuccessful_examples)} ex / "
              f"{int(n.n_unsuccessful_attacks)} attacks")
    print(f"[10] tables -> {tdir}; plots -> {pdir}")


if __name__ == "__main__":
    main()
