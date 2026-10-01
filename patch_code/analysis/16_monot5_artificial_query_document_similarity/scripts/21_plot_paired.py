"""
scripts/21_plot_paired.py
==========================
Stage 21 — figures, plot data and the compact final summary of the paired extension
(no model; reads stages 17-20). Relevant (qrel 2/3) and non-relevant (qrel 0) base
documents are always separate panels. Successful instances only (delta_score > 0).
Style = stage 05 (exp16lib palette: blue = padded control, orange = attacked).

Level plots (1, 2) show instance-level means with +-1 cluster-robust SE (clusters =
queries, the CV unit); the equal-document-weight means are in the CSVs.

plots/paired/
  fig_paired_layer_similarity.png            1  25 checkpoints: control vs attack (+ paired delta)
  fig_paired_heads_{scope}.png               2  every head of the scope (36 or 144): control vs attack
  fig_paired_abnormal_count_distribution.png 3  abnormal-head-count histogram + ECDF
  fig_paired_abnormal_count_change.png       4  attack - control abnormal count
  fig_paired_head_transitions.png            5  per head normal->abnormal / abnormal->normal / abnormal->abnormal
  fig_paired_per_head_abnormal_rate.png      6  per head control / attack rate and paired difference
  fig_paired_low_high.png                    7  attack-created abnormalities below vs above the reference
  fig_paired_topk_detector.png               8  top-k AUROC / F1 / precision / recall
  fig_paired_scatter_top_heads.png           9  control vs attack similarity, strongest heads, y = x
  fig_paired_document_level.png              10 equal-document-weight summaries
  fig_paired_layer_summary.png               11 per-layer abnormal counts, paired change, transitions
(all-layers scope: plots/paired_all_layers/, 21_paired_report_all_layers/)
21_paired_report/plot_data/*.csv             data behind every figure
21_paired_report/final_summary.json          compact summary of the whole extension
21_paired_report/old_outputs_check.json      pre-existing Exp 16 outputs unchanged (sha256; raises otherwise)
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.checkpoints import CHECKPOINTS  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_status, now, read_jsonl, stage_argparser, write_status  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_10_plot", EXP_DIR / "scripts" / "10_plot_head_similarity.py")
P10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P10)
P5 = P10.P5
K = P.K_ABNORMAL
SIDES = [("control", "padded control", P5.BLUE, "o", "-"), ("attack", "attacked (successful)", P5.ORANGE, "s", "-")]


def mean_cluster_se(X: np.ndarray, qid: np.ndarray):
    """Column means and cluster-robust SEs (clusters = queries)."""
    n = len(X)
    mu = X.mean(0)
    r = pd.DataFrame(X - mu).groupby(qid).sum().values
    G = len(r)
    se = np.sqrt(G / (G - 1) * (r ** 2).sum(0)) / n if G > 1 else np.full(X.shape[1], np.nan)
    return mu, se


def level_table(s, cols_c, cols_a, names, key):
    rows = []
    for g in P.GROUPS:
        d = s[s.relevance_group == g]
        if d.empty:
            continue
        C, A = d[cols_c].values, d[cols_a].values
        q = d.qid.values
        mc, sc = mean_cluster_se(C, q)
        ma, sa = mean_cluster_se(A, q)
        md, sd = mean_cluster_se(A - C, q)
        dm = d.groupby("pair_id")[cols_c + cols_a].mean()          # equal-document weight
        for j, nm in enumerate(names):
            rows.append({"relevance_group": g, key: nm, "control_mean": mc[j], "control_se": sc[j],
                         "attack_mean": ma[j], "attack_se": sa[j], "delta_mean": md[j], "delta_se": sd[j],
                         "control_mean_doc_weighted": dm[cols_c[j]].mean(), "attack_mean_doc_weighted": dm[cols_a[j]].mean(),
                         "n_instances": len(d), "n_docs": d.pair_id.nunique(), "n_queries": d.qid.nunique()})
    return pd.DataFrame(rows)


def head_axis(ax, labels, canon):
    X = np.arange(len(labels))
    ax.set_xticks(X)
    ax.set_xticklabels(labels, rotation=90, fontsize=7 if len(labels) <= 36 else 6)
    for tl in ax.get_xticklabels():
        if tl.get_text() in canon:
            tl.set_fontweight("bold")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    for b in range(12, len(labels), 12):                      # layer separators
        ax.axvline(b - 0.5, color=P5.INK2, lw=1.0, ls=":", zorder=1)
    for i, L in enumerate(P.LAYERS):
        ax.text(12 * i + 5.5, 1.01, f"layer {L}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=P5.INK2)


def old_outputs_check(out: pathlib.Path) -> dict:
    fp = json.loads((out / P.MANIFEST_DIR / "old_outputs_fingerprint.json").read_text())["sha256"]
    now_fp = P.fingerprint_old_outputs(out)
    changed = [k for k, v in fp.items() if now_fp.get(k) != v]
    added = sorted(set(now_fp) - set(fp))
    return {"n_files_fingerprinted": len(fp), "n_changed_or_missing": len(changed), "changed_or_missing": changed[:50],
            "n_new_files_outside_paired_dirs": len(added), "new_files": added[:50],
            "untouched": not changed and not added}


def main():
    args = stage_argparser("Exp 16 stage 21: paired plots + final summary").parse_args()
    cfg = load_config(args.config)
    pc = P.validate_cfg(cfg)
    out = output_dir(cfg)
    st19, st20 = out / P.ANOMALY_DIR, out / P.DETECTOR_DIR
    if not (is_already_successful(st19, ["paired_instances.csv.gz"]) and is_already_successful(st20, ["topk_detector_metrics.csv"])):
        raise FileNotFoundError("run stages 19 and 20 first")
    pdir = out / P.PLOT_DIR
    pdir.mkdir(parents=True, exist_ok=True)
    rep = out / P.REPORT_DIR
    ddir = rep / "plot_data"
    ddir.mkdir(parents=True, exist_ok=True)
    attacks = list(load_status(out / P.MANIFEST_DIR)["sha256"]["attacks"])
    labels = P.head_labels()
    canon = {h.label for h in encoder_heads()}
    X36 = np.arange(len(labels))                              # head x-positions (36 or 144)
    W = max(13.0, 0.19 * len(labels))                        # figure width for per-head plots
    gtitle = P.GROUP_LABEL

    inst = pd.read_csv(st19 / "paired_instances.csv.gz", dtype=P.DT)
    ph = pd.read_csv(st19 / "per_head_paired.csv", dtype=P.DT)
    docs = pd.read_csv(st19 / "document_level.csv.gz", dtype=P.DT)
    rf = pd.read_csv(st19 / "reference_fits.csv")
    topk = pd.read_csv(st20 / "topk_detector_metrics.csv", dtype=P.DT)
    groups = [g for g in P.GROUPS if (inst.relevance_group == g).any()]

    # ---- 1. layer-level similarity -------------------------------------------------------
    cc, ca = P.ckpt_cols("control"), P.ckpt_cols("attack")
    s = P.load_forward_rows(out / P.FORWARD_DIR, attacks, usecols=["attack_name", "pair_id", "qid", "relevance_group",
                                                                    "delta_score", "successful"] + cc + ca +
                            P.head_cols("control") + P.head_cols("attack"))
    s = s[s.successful].reset_index(drop=True)
    assert (s.delta_score > 0).all() and len(s) == len(inst)
    lt = level_table(s, cc, ca, CHECKPOINTS, "checkpoint")
    lt.to_csv(ddir / "paired_layer_similarity.csv", index=False)
    fig, axes = plt.subplots(2, len(groups), figsize=(8.5 * len(groups), 7.4), squeeze=False,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for c, g in enumerate(groups):
        t = lt[lt.relevance_group == g]
        for side, lab, col, mk, ls in SIDES:
            P5._line(axes[0, c], t[f"{side}_mean"].values, col, lab, se=t[f"{side}_se"].values, marker=mk, ls=ls)
        P5._axis(axes[0, c], "cos(query repr., document repr.)", zero=False)
        axes[0, c].legend(loc="lower left", fontsize=8)
        n = t.iloc[0]
        axes[0, c].set_title(f"{gtitle[g]}\n{int(n.n_instances):,} successful instances, {int(n.n_docs):,} docs, "
                             f"{int(n.n_queries)} queries; ±1 SE (query clusters)", loc="left", fontsize=9)
        P5._line(axes[1, c], t.delta_mean.values, P5.ORANGE, "attack − control", se=t.delta_se.values, marker="s")
        P5._axis(axes[1, c], "Δ cos (attack − control)")
        axes[1, c].set_xlabel("encoder checkpoint")
    fig.suptitle("Paired layer-level query-document similarity: same document, padded control → successful attack",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_layer_similarity.png")

    # ---- 2. every head of the scope --------------------------------------------------------
    ht = level_table(s, P.head_cols("control"), P.head_cols("attack"), labels, "head_label")
    ht.to_csv(ddir / f"paired_heads_{P.SCOPE}.csv", index=False)
    fig, axes = plt.subplots(len(groups), 1, figsize=(W + 1, 4.8 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        t = ht[ht.relevance_group == g].reset_index(drop=True)
        ax = axes[r, 0]
        P10.draw(ax, t, SIDES)
        head_axis(ax, labels, canon)
        ax.set_ylabel("cos(mean query head output,\nmean document head output)")
        ax.legend(loc="lower left", fontsize=8)
        n = t.iloc[0]
        ax.set_title(f"{gtitle[g]}: {int(n.n_instances):,} successful instances / {int(n.n_docs):,} docs; ±1 SE (query clusters)",
                     loc="left", fontsize=9, pad=16)
    axes[-1, 0].set_xlabel("encoder head (bold = one of the 18 canonical important heads)")
    fig.suptitle(f"Query-document similarity across all encoder heads, {P.scope_label()}: padded control vs successful attack",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, pdir / f"fig_paired_heads_{P.SCOPE}.png")

    # ---- 3. abnormal-count distributions -----------------------------------------------------
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 4.0 * len(groups)), squeeze=False)
    dist_rows = []
    for r, g in enumerate(groups):
        d = inst[inst.relevance_group == g]
        Cc, Ca = d.control_abnormal_count.values, d.attack_abnormal_count.values
        xmax = int(max(Cc.max(), Ca.max()))
        bins = np.arange(xmax + 2) - 0.5
        for C, side, col, off in ((Cc, "control", P5.BLUE, -0.2), (Ca, "attack", P5.ORANGE, 0.2)):
            h, _ = np.histogram(C, bins=bins)
            lab = f"{side} (mean {C.mean():.2f}, median {np.median(C):.0f})"
            axes[r, 0].bar(np.arange(xmax + 1) + off, h / len(C), width=0.4, color=col, label=lab)
            axes[r, 0].axvline(np.median(C), color=col, ls="--", lw=1.2)
            xs = np.sort(C)
            axes[r, 1].step(np.concatenate([[0], xs]), np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs)]),
                            where="post", color=col, lw=2, label=side)
            dist_rows += [{"relevance_group": g, "side": side, "abnormal_count": v, "fraction": h[v] / len(C)}
                          for v in range(xmax + 1)]
        axes[r, 0].set_ylabel("fraction of successful instances")
        axes[r, 0].set_title(f"{gtitle[g]}  (n = {len(d):,} pairs; dashed = median)", loc="left", fontsize=9)
        axes[r, 0].legend(fontsize=8)
        axes[r, 1].set_ylabel("cumulative fraction")
        axes[r, 1].legend(fontsize=8, loc="lower right")
        axes[r, 1].set_title("ECDF", loc="left", fontsize=9)
    for c in range(2):
        axes[-1, c].set_xlabel(f"abnormal_head_count  (# of {P.N_HEADS} heads with |z| > 2)")
    fig.suptitle(f"{P.scope_label()}: abnormal-head counts of the same documents before (padded control) and after a successful attack\n"
                 "reference = successful qrel 2/3 controls of training queries; 5-fold query CV", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_abnormal_count_distribution.png")
    pd.DataFrame(dist_rows).to_csv(ddir / "paired_abnormal_count_distribution.csv", index=False)

    # ---- 4. paired change ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 3.8), squeeze=False, sharey=True)
    ch_rows = []
    for c, g in enumerate(groups):
        ch = inst[inst.relevance_group == g].abnormal_count_change.values
        vals, cnt = np.unique(ch, return_counts=True)
        ax = axes[0, c]
        ax.bar(vals, cnt / len(ch), width=0.8, color=[P5.ORANGE if v > 0 else P5.BLUE if v < 0 else P5.INK2 for v in vals])
        ax.axvline(ch.mean(), color=P5.INK, lw=1.2, ls="--", label=f"mean {ch.mean():+.2f}")
        ax.axvline(np.median(ch), color=P5.AQUA, lw=1.2, ls=":", label=f"median {np.median(ch):+.0f}")
        ax.set_xlabel("abnormal_count(attack) − abnormal_count(control)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{gtitle[g]}\nincrease {np.mean(ch > 0):.1%}, no change {np.mean(ch == 0):.1%}, "
                     f"decrease {np.mean(ch < 0):.1%}", loc="left", fontsize=9)
        ax.legend(fontsize=8)
        ch_rows += [{"relevance_group": g, "paired_change": int(v), "fraction": n_ / len(ch)} for v, n_ in zip(vals, cnt)]
    axes[0, 0].set_ylabel("fraction of successful instances")
    P5._save(fig, pdir / "fig_paired_abnormal_count_change.png")
    pd.DataFrame(ch_rows).to_csv(ddir / "paired_abnormal_count_change.csv", index=False)

    # ---- 5. per-head transitions -----------------------------------------------------------------
    tcols = ["relevance_group", "head_label", "rate_normal_to_normal", "rate_normal_to_abnormal",
             "rate_abnormal_to_normal", "rate_abnormal_to_abnormal"]
    ph[tcols].to_csv(ddir / "paired_head_transitions.csv", index=False)
    fig, axes = plt.subplots(len(groups), 1, figsize=(W, 4.4 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        t = ph[ph.relevance_group == g].set_index("head_label").loc[labels]
        ax = axes[r, 0]
        ax.bar(X36, t.rate_normal_to_abnormal, color=P5.ORANGE, label="normal → abnormal (attack creates)")
        ax.bar(X36, -t.rate_abnormal_to_normal, color=P5.BLUE, label="abnormal → normal (attack removes)")
        ax.plot(X36, t.rate_abnormal_to_abnormal, color=P5.INK2, lw=0, marker="_", ms=10, mew=2,
                label="abnormal → abnormal (stays)")
        ax.axhline(0, color=P5.INK, lw=1)
        head_axis(ax, labels, canon)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))   # mirrored axis: show |value|
        ax.set_ylabel("fraction of successful instances")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title(gtitle[g], loc="left", fontsize=9, pad=16)
    fig.suptitle(f"Per-head abnormality transitions, padded control → successful attack (|z| > 2), {P.scope_label()}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_head_transitions.png")

    # ---- 6. per-head abnormality rates -----------------------------------------------------------
    rcols = ["relevance_group", "head_label", "control_abnormal_rate", "attack_abnormal_rate", "paired_rate_difference"]
    ph[rcols].to_csv(ddir / "paired_per_head_abnormal_rate.csv", index=False)
    fig, axes = plt.subplots(2 * len(groups), 1, figsize=(W, 3.9 * 2 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        t = ph[ph.relevance_group == g].set_index("head_label").loc[labels]
        ax = axes[2 * r, 0]
        ax.plot(X36, t.control_abnormal_rate, color=P5.BLUE, lw=2, marker="o", ms=4, label="padded control")
        ax.plot(X36, t.attack_abnormal_rate, color=P5.ORANGE, lw=2, marker="s", ms=4, label="successful attack")
        ax.axhline(0.0455, color=P5.INK2, lw=0.8, ls="-.", label="4.6% (Gaussian |z| > 2)")
        head_axis(ax, labels, canon)
        ax.set_ylabel("fraction abnormal (|z| > 2)")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{gtitle[g]}: per-head abnormality rate", loc="left", fontsize=9, pad=16)
        ax = axes[2 * r + 1, 0]
        d = t.paired_rate_difference.values
        ax.bar(X36, d, color=[P5.ORANGE if v > 0 else P5.BLUE for v in d])
        ax.axhline(0, color=P5.INK, lw=1)
        head_axis(ax, labels, canon)
        ax.set_ylabel("attack − control rate")
        ax.set_title(f"{gtitle[g]}: paired difference (positive = more often abnormal after the attack)",
                     loc="left", fontsize=9, pad=16)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_per_head_abnormal_rate.png")

    # ---- 7. low vs high ------------------------------------------------------------------------------
    lh = pd.read_csv(st19 / "low_high_summary.csv", dtype=P.DT)
    lcols = ["relevance_group", "head_label", "rate_normal_to_abnormal_low", "rate_normal_to_abnormal_high",
             "control_low_rate", "control_high_rate", "attack_low_rate", "attack_high_rate"]
    ph[lcols].to_csv(ddir / "paired_low_high_per_head.csv", index=False)
    fig, axes = plt.subplots(len(groups), 1, figsize=(W, 4.4 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        t = ph[ph.relevance_group == g].set_index("head_label").loc[labels]
        s_ = lh[lh.relevance_group == g].iloc[0]
        ax = axes[r, 0]
        ax.bar(X36, t.rate_normal_to_abnormal_high, color=P5.ORANGE, label="normal → HIGH (z > +2)")
        ax.bar(X36, -t.rate_normal_to_abnormal_low, color=P5.BLUE, label="normal → LOW (z < −2)")
        ax.axhline(0, color=P5.INK, lw=1)
        head_axis(ax, labels, canon)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))   # mirrored axis: show |value|
        ax.set_ylabel("fraction of successful instances\n(down = below, up = above reference)")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title(f"{gtitle[g]}: attack-created abnormalities — {s_.normal_to_abnormal_share_low:.0%} below, "
                     f"{s_.normal_to_abnormal_share_high:.0%} above the genuine-relevance reference "
                     f"({int(s_.normal_to_abnormal_events):,} head events)", loc="left", fontsize=9, pad=16)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_low_high.png")

    # ---- 8. top-k detector ---------------------------------------------------------------------------
    topk.to_csv(ddir / "paired_topk_detector.csv", index=False)
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 4.0 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        t = topk[topk.relevance_group == g]
        ax = axes[r, 0]
        ax.fill_between(t.k_heads, t.auroc_fold_min, t.auroc_fold_max, color=P5.ORANGE, alpha=0.15, lw=0, label="fold min–max")
        ax.plot(t.k_heads, t.auroc_pooled, color=P5.ORANGE, lw=2, marker="s", ms=3, label="pooled held-out AUROC")
        ax.axhline(0.5, color=P5.INK2, lw=0.8, ls="-.")
        for k in sorted({1, 5, 10, 36, P.N_HEADS}):
            v = float(t.loc[t.k_heads == k, "auroc_pooled"].iloc[0])
            ax.annotate(f"{v:.3f}", (k, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylabel("AUROC (attack vs own control)")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{gtitle[g]}: AUROC", loc="left", fontsize=9)
        ax = axes[r, 1]
        for col, c_, mk, lab in (("f1", P5.INK, "o", "F1"), ("precision", P5.BLUE, "^", "precision"),
                                 ("recall", P5.ORANGE, "s", "recall")):
            ax.plot(t.k_heads, t[col], color=c_, lw=2, marker=mk, ms=3, label=lab)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("at training-chosen threshold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{gtitle[g]}: F1 / precision / recall", loc="left", fontsize=9)
    for c in range(2):
        axes[-1, c].set_xlabel("number of heads (top-k, ranked on training folds only)")
    fig.suptitle("Paired top-k head detector: positive = successful attack, negative = the same document's padded control",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_topk_detector.png")

    # ---- 9. scatter, strongest heads ---------------------------------------------------------------------
    nh, nmax = int(pc["scatter_heads"]), int(pc["scatter_max_points"])
    ref_mu = rf.groupby("head_label").mu.mean()
    ref_sd = rf.groupby("head_label").sigma.mean()
    fig, axes = plt.subplots(len(groups), nh, figsize=(3.0 * nh, 3.2 * len(groups)), squeeze=False)
    sc_rows = []
    rng = np.random.default_rng(42)
    for r, g in enumerate(groups):
        t = ph[ph.relevance_group == g].copy()
        top = t.reindex(t.paired_rate_difference.abs().sort_values(ascending=False).index).head(nh).head_label.tolist()
        d = s[s.relevance_group == g]
        idx = rng.choice(len(d), size=min(nmax, len(d)), replace=False)
        for c, h in enumerate(top):
            x, y = d[f"control_head_{h}"].values, d[f"attack_head_{h}"].values
            ax = axes[r, c]
            ax.scatter(x[idx], y[idx], s=4, alpha=0.3, color=P5.ORANGE, lw=0)
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], color=P5.INK, lw=1, label="y = x")
            m_, s_ = ref_mu[h], ref_sd[h]
            for v in (m_ - 2 * s_, m_ + 2 * s_):
                ax.axvline(v, color=P5.BLUE, lw=0.8, ls="--")
                ax.axhline(v, color=P5.BLUE, lw=0.8, ls="--")
            dr = float(t.set_index("head_label").loc[h, "paired_rate_difference"])
            ax.set_title(f"{h}{' (canonical)' if h in canon else ''}\nΔ abnormal rate {dr:+.3f}", fontsize=8, loc="left")
            ax.set_xlabel("control similarity", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{'relevant' if g == P.RELEVANT else 'non-relevant'}\nattack similarity", fontsize=8)
            sc_rows.append({"relevance_group": g, "head_label": h, "rank": c + 1, "paired_rate_difference": dr,
                            "reference_mu_fold_mean": m_, "reference_sigma_fold_mean": s_,
                            "frac_attack_above_control": float((y > x).mean()),
                            "mean_control_similarity": float(x.mean()), "mean_attack_similarity": float(y.mean())})
    fig.suptitle("Control vs attack similarity for the heads with the largest paired abnormality change "
                 f"(≤{nmax:,} random points per panel; dashed = fold-mean reference μ ± 2σ; heads chosen descriptively)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_scatter_top_heads.png")
    pd.DataFrame(sc_rows).to_csv(ddir / "paired_scatter_top_heads.csv", index=False)

    # ---- 10. document level --------------------------------------------------------------------------------
    dsum = pd.read_csv(st19 / "document_level_summary.csv", dtype=P.DT)
    docs.to_csv(ddir / "paired_document_level.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = {P.RELEVANT: P5.BLUE, P.NONRELEVANT: P5.ORANGE}
    allv = docs.mean_paired_change.values
    bins = np.linspace(allv.min() - 0.5, allv.max() + 0.5, 40)
    for g in groups:
        d = docs[docs.relevance_group == g]
        axes[0].hist(d.mean_paired_change, bins=bins, histtype="step", lw=2, color=colors[g], density=True,
                     label=f"{'relevant' if g == P.RELEVANT else 'non-relevant'} ({len(d):,} docs; mean {d.mean_paired_change.mean():+.2f})")
        axes[1].scatter(d.mean_control_abnormal_count, d.mean_attack_abnormal_count, s=6, alpha=0.4, color=colors[g], lw=0,
                        label="relevant" if g == P.RELEVANT else "non-relevant")
    axes[0].axvline(0, color=P5.INK, lw=1)
    axes[0].set_xlabel("per-document mean paired change (attack − control)")
    axes[0].set_ylabel("density over documents")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Paired change, equal document weight", loc="left", fontsize=9)
    lim = [0, max(docs.mean_control_abnormal_count.max(), docs.mean_attack_abnormal_count.max()) + 0.5]
    axes[1].plot(lim, lim, color=P5.INK, lw=1, label="y = x")
    axes[1].set_xlabel("mean control abnormal count (per document)")
    axes[1].set_ylabel("mean attack abnormal count (per document)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Each document vs its own controls", loc="left", fontsize=9)
    xs = np.arange(len(groups))
    for off, side, col in ((-0.2, "control", P5.BLUE), (0.2, "attack", P5.ORANGE)):
        vals = [float(dsum[dsum.relevance_group == g][f"mean_{side}_abnormal_count"].iloc[0]) for g in groups]
        axes[2].bar(xs + off, vals, width=0.4, color=col, label=side)
        for x_, v in zip(xs + off, vals):
            axes[2].annotate(f"{v:.2f}", (x_, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(["relevant (qrel 2/3)" if g == P.RELEVANT else "non-relevant (qrel 0)" for g in groups])
    axes[2].set_ylabel("mean abnormal count (doc-weighted)")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Equal-document-weight means", loc="left", fontsize=9)
    P5._save(fig, pdir / "fig_paired_document_level.png")
    dsum.to_csv(ddir / "paired_document_level_summary.csv", index=False)

    # ---- 11. per-layer summary: where is any paired effect strongest? ---------------------------------------
    lsum = pd.read_csv(st19 / "layer_summary.csv", dtype=P.DT)
    lsum.to_csv(ddir / "paired_layer_summary.csv", index=False)
    fig, axes = plt.subplots(len(groups), 3, figsize=(17, 3.9 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        t = lsum[lsum.relevance_group == g].sort_values("layer")
        x = t.layer.values
        ax = axes[r, 0]
        ax.plot(x, t.mean_control_abnormal_count, color=P5.BLUE, marker="o", ms=4, label="padded control")
        ax.plot(x, t.mean_attack_abnormal_count, color=P5.ORANGE, marker="s", ms=4, label="successful attack")
        ax.set_ylabel("mean # abnormal heads (of 12)")
        ax.set_title(f"{gtitle[g]}\nabnormal heads per layer", loc="left", fontsize=9)
        ax.legend(fontsize=8)
        ax = axes[r, 1]
        ax.bar(x, t.mean_paired_change, color=[P5.ORANGE if v > 0 else P5.BLUE for v in t.mean_paired_change])
        ax.errorbar(x, t.mean_paired_change, yerr=t.se_paired_change_query_cluster, fmt="none", ecolor=P5.INK2, capsize=2)
        ax.axhline(0, color=P5.INK, lw=1)
        ax.set_ylabel("mean paired change (attack − control)")
        ax.set_title("paired change per layer (±1 SE, query clusters)", loc="left", fontsize=9)
        ax = axes[r, 2]
        ax.bar(x, t.rate_normal_to_abnormal, color=P5.ORANGE, label="normal → abnormal")
        ax.bar(x, -t.rate_abnormal_to_normal, color=P5.BLUE, label="abnormal → normal")
        ax.axhline(0, color=P5.INK, lw=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))
        ax.set_ylabel("rate per head-instance")
        ax.set_title("transitions per layer", loc="left", fontsize=9)
        ax.legend(fontsize=8)
        for ax in axes[r]:
            ax.set_xticks(x)
    for ax in axes[-1]:
        ax.set_xlabel("encoder layer")
    fig.suptitle(f"Where is the paired effect strongest? Per-layer summary, {P.scope_label()}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_paired_layer_summary.png")

    # ---- final summary -----------------------------------------------------------------------------------------
    prov = json.loads((out / P.MANIFEST_DIR / "provenance.json").read_text())
    s19 = json.loads((st19 / "summary.json").read_text())
    s20 = json.loads((st20 / "detector_summary.json").read_text())
    xc = pd.read_csv(out / P.FORWARD_DIR / "per_attack_status.csv")
    xcs = [json.loads(v) for v in xc.crosscheck]
    agg = {}
    for name in ("exp01_scores", "stage03_ckpt_sims", "stage13_head_sims"):
        v = [c[name] for c in xcs if name in c]
        agg[name] = {"n_attacks_checked": len(v), "n_overlap_instances": int(sum(x["n_overlap"] for x in v)),
                     "max_abs_diff": float(max((x["max_abs_diff"] for x in v), default=np.nan))}
    oc = old_outputs_check(out)
    (rep / "old_outputs_check.json").write_text(json.dumps(oc, indent=2))
    cs = pd.read_csv(st19 / "paired_count_summary.csv", dtype=P.DT).set_index("relevance_group")
    final = {
        "created": now(), "config": cfg["_config_path"],
        "design": {"pair": "padded control -> attacked input (same length / positions)",
                   "success": "delta_score = score_attack - score_control > 0",
                   "features": f"{P.N_HEADS} encoder heads of layers {P.LAYERS}, pre-o_proj, cos(mean query, mean document)",
                   "scope": P.SCOPE,
                   "reference": "successful qrel 2/3 controls of training queries, all attack configs pooled",
                   "abnormal": "|z| > 2", "cv": "5-fold by query, anomaly.query_folds, seed 42"},
        "dataset": {k: prov[k] for k in ("base_pairs_in_tsv", "base_pair_grade_counts", "n_base_relevant",
                                         "n_base_nonrelevant", "n_queries", "fold_sizes_queries", "n_attacks",
                                         "n_instances_before_alignment", "n_instances_before_alignment_relevant",
                                         "n_instances_before_alignment_nonrelevant", "n_instances_aligned",
                                         "n_boundary_shift", "smoke_trimmed")},
        "alignment_failures": {k: prov[k] for k in ("n_alignment_failures", "n_alignment_failures_relevant",
                                                    "n_alignment_failures_nonrelevant", "n_attacks_with_failures")},
        "success": s19["success_by_group"],
        "reference": s19["reference"],
        "paired_abnormal_count": {g: {k: s19["groups"][g]["instance_level"][k] for k in (
            "n_successful_instances", "n_docs", "mean_control_abnormal_count", "mean_attack_abnormal_count",
            "mean_paired_change", "median_paired_change", "se_paired_change_query_cluster", "frac_change_positive",
            "frac_change_zero", "frac_change_negative", "query_level_signflip_p_one_sided_gt0")} for g in s19["groups"]},
        "low_high": {g: s19["groups"][g]["low_high"] for g in s19["groups"]},
        "detector_all_heads": {g: {k: v.get("all_heads", v.get("all_36_heads"))[k] for k in ("auroc_pooled", "auroc_fold_mean", "f1", "precision",
                                                                  "recall", "fpr", "paired_win_rate", "T_per_fold")}
                            for g, v in s20["groups"].items()},
        "detector_cv_selected_k": {g: v["cv_selected_k"] for g, v in s20["groups"].items()},
        "detector_topk": {g: v["topk"] for g, v in s20["groups"].items()},
        "document_level": {g: s19["groups"][g]["document_level"] for g in s19["groups"]},
        "layer_summary": {g: lsum[lsum.relevance_group == g][["layer", "mean_control_abnormal_count",
                              "mean_attack_abnormal_count", "mean_paired_change", "se_paired_change_query_cluster"]]
                          .to_dict("records") for g in groups},
        "crosscheck_vs_original_caches": agg,
        "old_outputs_untouched": oc["untouched"],
    }
    (rep / "final_summary.json").write_text(json.dumps(final, indent=2, default=float))
    write_status(rep, {"status": "success" if oc["untouched"] else "failed", "finished": now()})
    print(cs[["n_successful_instances", "mean_control_abnormal_count", "mean_attack_abnormal_count",
              "mean_paired_change", "median_paired_change"]].to_string())
    print(f"[21] plots -> {pdir}\n[21] summary -> {rep / 'final_summary.json'}")
    if not oc["untouched"]:
        raise RuntimeError(f"pre-existing Exp 16 outputs changed: {oc}")


if __name__ == "__main__":
    main()
