"""
scripts/16_head_anomaly_attribution.py
=======================================
Stage 16 — WHICH of the 36 layer-9-11 encoder heads make successful attacks
more abnormal than genuine relevant inputs? (no model; reuses stage 15's
held-out predictions 15_head_anomaly/anomaly_predictions.csv.gz: raw sims,
z-scores vs the fold's genuine-training reference, fold ids; |z| > 2 fixed.)

Per head (pooled held-out rows, and per fold):
  genuine/attack abnormal rate, low (z < -2) / high (z > +2) rates,
  excess = attack rate - genuine rate, mean z, mean |z|,
  fold mean/std/min/max of the excess, #folds with excess > 0.
Canonical flag: head in Exp 13's 18 important encoder heads.

Top-k detector ablation (no leakage): for held-out fold f, heads are ranked by
exp16lib.anomaly.inner_head_ranking on the TRAINING folds only (inner references
refitted from the raw similarities without fold f); fold f is then scored with
its own saved z-scores restricted to the top-k heads. AUROC of the restricted
abnormal-head count is pooled over the held-out folds, for k = 1..36.

Outputs: 15_head_anomaly/per_head_attack_anomaly_analysis.csv,
         15_head_anomaly/topk_head_detector_auroc.csv,
         15_head_anomaly/attribution_summary.json,
         plots/fig_anomaly_attr_*.png
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.anomaly import auroc, inner_head_ranking  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import all_encoder_heads, encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, now, stage_argparser, write_status  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_05_plot", EXP_DIR / "scripts" / "05_plot.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

K = 2.0
N_FOLDS = 5
GEN, ATK = "genuine_relevant", "successful_attack"
TOPK_REPORT = [3, 5, 10, 36]


def main():
    args = stage_argparser("Exp 16 stage 16: which heads drive attack abnormality").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / "15_head_anomaly"
    if not is_already_successful(st, ["anomaly_predictions.csv.gz"]):
        raise FileNotFoundError(f"run stage 15 first ({st})")
    pred_path = st / "anomaly_predictions.csv.gz"
    p = pd.read_csv(pred_path, dtype={"qid": str, "docid": str, "pair_id": str, "attack_name": str})
    heads = all_encoder_heads([9, 10, 11])
    labels = [h.label for h in heads]
    zc, sc = [f"z_{l}" for l in labels], [f"sim_{l}" for l in labels]
    if len(labels) != 36 or not set(zc + sc) <= set(p.columns):
        raise RuntimeError("stage-15 predictions must contain sim_/z_ columns for the 36 heads")
    if not (p[p.group == ATK].delta_score > 0).all():
        raise RuntimeError("attack rows must all have delta_score > 0")
    for f in range(N_FOLDS):                                     # query-disjoint folds
        assert not (set(p.qid[p.fold == f]) & set(p.qid[p.fold != f]))
    canon = {h.label for h in encoder_heads()}

    g, a = p[p.group == GEN], p[p.group == ATK]
    Zg, Za = np.nan_to_num(g[zc].values), np.nan_to_num(a[zc].values)
    fg, fa = g.fold.values, a.fold.values

    def rates(Z):
        return (np.abs(Z) > K).mean(0), (Z < -K).mean(0), (Z > K).mean(0)

    gr, gl, gh = rates(Zg)
    ar, al, ah = rates(Za)
    fold_g = np.stack([rates(Zg[fg == f])[0] for f in range(N_FOLDS)])
    fold_a = np.stack([rates(Za[fa == f])[0] for f in range(N_FOLDS)])
    fold_ex = fold_a - fold_g
    t = pd.DataFrame({
        "layer": [h.layer for h in heads], "head": [h.head_idx for h in heads], "head_label": labels,
        "canonical_important_head": [l in canon for l in labels],
        "genuine_abnormal_rate": gr, "attack_abnormal_rate": ar, "excess_attack_abnormality": ar - gr,
        "genuine_low_rate": gl, "genuine_high_rate": gh, "attack_low_rate": al, "attack_high_rate": ah,
        "mean_z_genuine": Zg.mean(0), "mean_z_attack": Za.mean(0),
        "mean_abs_z_genuine": np.abs(Zg).mean(0), "mean_abs_z_attack": np.abs(Za).mean(0),
        "fold_mean_genuine_rate": fold_g.mean(0), "fold_se_genuine_rate": fold_g.std(0, ddof=1) / np.sqrt(N_FOLDS),
        "fold_mean_attack_rate": fold_a.mean(0), "fold_se_attack_rate": fold_a.std(0, ddof=1) / np.sqrt(N_FOLDS),
        "fold_mean_excess": fold_ex.mean(0), "fold_std_excess": fold_ex.std(0, ddof=1),
        "fold_min_excess": fold_ex.min(0), "fold_max_excess": fold_ex.max(0),
        "folds_attack_greater_than_genuine": (fold_ex > 0).sum(0),
    })
    t["attack_abnormality_direction"] = np.where(t.attack_low_rate >= t.attack_high_rate, "low", "high")
    t["rank_by_excess"] = t.excess_attack_abnormality.rank(ascending=False, method="first").astype(int)
    t.to_csv(st / "per_head_attack_anomaly_analysis.csv", index=False)

    # --- top-k detector ablation, heads chosen on training folds only -----------------
    Xg, Xa = g[sc].values.astype(float), a[sc].values.astype(float)
    orders = {f: inner_head_ranking(Xg, fg, Xa, fa, [x for x in range(N_FOLDS) if x != f], K) for f in range(N_FOLDS)}
    rows = []
    for k in range(1, 37):
        cg, ca = np.zeros(len(g)), np.zeros(len(a))
        for f in range(N_FOLDS):
            sel = orders[f][:k]
            cg[fg == f] = (np.abs(Zg[fg == f][:, sel]) > K).sum(1)
            ca[fa == f] = (np.abs(Za[fa == f][:, sel]) > K).sum(1)
        per_fold = [auroc(ca[fa == f], cg[fg == f]) for f in range(N_FOLDS)]
        rows.append({"k_heads": k, "auroc_pooled": auroc(ca, cg), "auroc_fold_mean": float(np.mean(per_fold)),
                     "auroc_fold_min": float(np.min(per_fold)), "auroc_fold_max": float(np.max(per_fold))})
    topk = pd.DataFrame(rows)
    topk.to_csv(st / "topk_head_detector_auroc.csv", index=False)
    sel_heads = {f: [labels[i] for i in orders[f][:10]] for f in range(N_FOLDS)}

    top10 = t.sort_values("excess_attack_abnormality", ascending=False).head(10)
    summary = {
        "created": now(), "predictions_reused": str(pred_path), "k": K,
        "n_genuine": int(len(g)), "n_attack": int(len(a)),
        "top10_by_excess": top10[["head_label", "genuine_abnormal_rate", "attack_abnormal_rate",
                                  "excess_attack_abnormality", "attack_abnormality_direction",
                                  "canonical_important_head", "folds_attack_greater_than_genuine"]].round(4).to_dict("records"),
        "n_top10_canonical": int(top10.canonical_important_head.sum()),
        "n_canonical_among_36": int(t.canonical_important_head.sum()),
        "heads_positive_in_all_folds": t[t.folds_attack_greater_than_genuine == N_FOLDS].head_label.tolist(),
        "share_positive_excess_in_canonical": float(t.excess_attack_abnormality.clip(lower=0)[t.canonical_important_head].sum()
                                                    / t.excess_attack_abnormality.clip(lower=0).sum()),
        "share_positive_excess_top_k": {k: float(t.excess_attack_abnormality.clip(lower=0)
                                                 .sort_values(ascending=False).head(k).sum()
                                                 / t.excess_attack_abnormality.clip(lower=0).sum()) for k in (3, 5, 10)},
        "topk_auroc": topk[topk.k_heads.isin(TOPK_REPORT)].round(4).to_dict("records"),
        "top10_heads_selected_per_fold_on_training": sel_heads,
    }
    (st / "attribution_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # --- plots ---------------------------------------------------------------------------
    pdir = out / "plots"
    X = np.arange(36)

    def head_axis(ax, lab=labels):
        ax.set_xticks(X)
        ax.set_xticklabels(lab, rotation=90, fontsize=7)
        for tl in ax.get_xticklabels():
            if tl.get_text() in canon:
                tl.set_fontweight("bold")
        ax.set_xlim(-0.6, 35.6)

    def layer_seps(ax):
        for b in (12, 24):
            ax.axvline(b - 0.5, color=P5.INK2, lw=1.0, ls=":", zorder=1)
        for i, L in enumerate((9, 10, 11)):
            ax.text(12 * i + 5.5, 1.01, f"layer {L}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=P5.INK2)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    for y, se, c, m, lab in ((t.fold_mean_genuine_rate, t.fold_se_genuine_rate, P5.BLUE, "o", "genuine relevant (held-out)"),
                             (t.fold_mean_attack_rate, t.fold_se_attack_rate, P5.ORANGE, "s", "successful attacks (held-out)")):
        ax.fill_between(X, y - se, y + se, color=c, alpha=0.2, lw=0)
        ax.plot(X, y, color=c, lw=2, marker=m, ms=4, label=lab)
    head_axis(ax); layer_seps(ax)
    ax.set_xlabel("encoder head (bold = one of the 18 canonical causally important heads)")
    ax.set_ylabel("fraction abnormal (|z| > 2)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Which late-encoder heads are abnormal under successful attacks?\n"
                 "mean over the 5 held-out query folds, ±1 SE across folds", loc="left", fontsize=9, pad=16)
    P5._save(fig, pdir / "fig_anomaly_attr_rates_by_head.png")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4))
    ex = t.excess_attack_abnormality.values
    err = np.vstack([ex - t.fold_min_excess.values, t.fold_max_excess.values - ex])
    for ax, order, title in ((axes[0], np.arange(36), "ordered by layer / head"),
                             (axes[1], np.argsort(-ex, kind="stable"), "ranked by excess")):
        colors = [P5.ORANGE if ex[i] > 0 else P5.BLUE for i in order]
        hatch = ["////" if labels[i] in canon else "" for i in order]
        bars = ax.bar(X, ex[order], color=colors, edgecolor="white", linewidth=0.5)
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        ax.errorbar(X, ex[order], yerr=err[:, order], fmt="none", ecolor=P5.INK2, lw=0.8, capsize=2)
        ax.axhline(0, color=P5.INK, lw=1)
        head_axis(ax, [labels[i] for i in order])
        if title.startswith("ordered"):
            layer_seps(ax)
        ax.set_ylabel("attack − genuine abnormal rate")
        ax.set_title(title + "  (whiskers = min/max over the 5 folds; hatched = canonical important head)",
                     loc="left", fontsize=9, pad=14)
    fig.suptitle("Excess abnormality under successful attacks, per head (positive = more often abnormal under attack)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, pdir / "fig_anomaly_attr_excess_bars.png")

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(X, t.attack_high_rate, color=P5.ORANGE, label="successful attacks: z > +2 (too HIGH)")
    ax.bar(X, -t.attack_low_rate, color=P5.BLUE, label="successful attacks: z < −2 (too LOW)")
    ax.plot(X, t.genuine_high_rate, color=P5.INK2, lw=0, marker="_", ms=10, label="genuine reference rates (±)")
    ax.plot(X, -t.genuine_low_rate, color=P5.INK2, lw=0, marker="_", ms=10)
    ax.axhline(0, color=P5.INK, lw=1)
    head_axis(ax); layer_seps(ax)
    yt = ax.get_yticks()
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{abs(v):.1f}" for v in yt])
    ax.set_xlabel("encoder head (bold = canonical important head)")
    ax.set_ylabel("fraction of successful attacks\n(down = below, up = above genuine)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Direction of abnormality under successful attacks: too low or too high vs genuine relevance",
                 loc="left", fontsize=9, pad=16)
    P5._save(fig, pdir / "fig_anomaly_attr_direction.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(topk.k_heads, topk.auroc_fold_min, topk.auroc_fold_max, color=P5.ORANGE, alpha=0.15, lw=0,
                    label="fold min–max")
    ax.plot(topk.k_heads, topk.auroc_pooled, color=P5.ORANGE, lw=2, marker="s", ms=3, label="pooled held-out AUROC")
    for k in TOPK_REPORT:
        v = float(topk.loc[topk.k_heads == k, "auroc_pooled"].iloc[0])
        ax.annotate(f"{v:.3f}", (k, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("number of heads used (top-k, ranked on training folds only)")
    ax.set_ylabel("AUROC (abnormal-head count)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("How few heads does the detector need?", loc="left", fontsize=9)
    P5._save(fig, pdir / "fig_anomaly_attr_topk_auroc.png")

    write_status(st / "attribution", {"status": "success", "finished": now()})
    print(json.dumps({k: summary[k] for k in ("top10_by_excess", "n_top10_canonical", "heads_positive_in_all_folds",
                                              "share_positive_excess_in_canonical", "share_positive_excess_top_k",
                                              "topk_auroc", "top10_heads_selected_per_fold_on_training")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
