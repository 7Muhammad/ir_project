"""
scripts/15_head_anomaly_detection.py
=====================================
Stage 15 — interpretable anomaly detection on the 36 layer-9-11 encoder-head
query/document similarity features (no model; reads 13_late_encoder_heads/).
Method, folds and threshold selection: exp16lib/anomaly.py.

  normal reference : clean qrel 2/3 documents of TRAINING queries only
  test groups      : held-out genuine relevant (qrel 2/3) vs held-out
                     successful attacks (example-level delta_score > 0, attacked input)
  queries          : only qids present in BOTH groups (qid-level 5-fold CV, seed 42)
  primary          : k = 2 (|z| > 2), score = abnormal_head_count (0..36)
  robustness       : k in {1.0, 1.5, 2.5, 3.0}; secondary score mean_abs_z

Metrics are example-level over the pooled held-out folds (attacks outnumber
genuine ~8:1, so precision/accuracy reflect that ratio; balanced accuracy,
TPR/FPR and AUROC do not). An equal-attack-configuration recall is added.

Outputs (15_head_anomaly/): anomaly_predictions.csv.gz, per_head_abnormality.csv,
detector_threshold_curve.csv, fold_summary.csv, overall_summary.json;
plots/fig_anomaly_*.png
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

from exp16lib.anomaly import (abnormal_counts, auroc, fit_reference, inner_threshold, query_folds,  # noqa: E402
                              threshold_metrics, zscores)
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import all_encoder_heads, encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, now, stage_argparser, write_status  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_05_plot", EXP_DIR / "scripts" / "05_plot.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

LAYERS, N_FOLDS, SEED, K_PRIMARY = [9, 10, 11], 5, 42, 2.0
K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
GEN, ATK = "genuine_relevant", "successful_attack"


def run_k(Xg, fg, Xa, fa, k):
    """Held-out z / counts for every example at threshold k, plus per-fold records."""
    H = Xg.shape[1]
    zg, za = np.full(Xg.shape, np.nan), np.full(Xa.shape, np.nan)
    folds, fits = [], {}
    for f in range(N_FOLDS):
        mu, sd, ok = fit_reference(Xg[fg != f])                      # genuine TRAINING queries only
        fits[f] = (mu, sd, ok)
        zg[fg == f] = zscores(Xg[fg == f], mu, sd, ok)
        za[fa == f] = zscores(Xa[fa == f], mu, sd, ok)
        T = inner_threshold(Xg, fg, Xa, fa, [g for g in range(N_FOLDS) if g != f], k)
        cg = abnormal_counts(zg[fg == f], k)
        ca = abnormal_counts(za[fa == f], k)
        m = threshold_metrics(ca["abnormal_head_count"], cg["abnormal_head_count"], T)
        folds.append({"k": k, "fold": f, "n_genuine_test": int((fg == f).sum()), "n_attack_test": int((fa == f).sum()),
                      "n_heads_sigma_below_min": int((~ok).sum()), "T_chosen_inner_cv": T, **m,
                      "auroc_abnormal_head_count": auroc(ca["abnormal_head_count"], cg["abnormal_head_count"]),
                      "auroc_mean_abs_z": auroc(ca["mean_abs_z"], cg["mean_abs_z"])})
    return zg, za, pd.DataFrame(folds), fits


def dist(x):
    q = np.percentile(x, [25, 50, 75])
    return {"mean": float(np.mean(x)), "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]),
            "min": int(np.min(x)), "max": int(np.max(x)), "frac_zero": float(np.mean(x == 0))}


def main():
    args = stage_argparser("Exp 16 stage 15: head-count anomaly detection").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / "13_late_encoder_heads"
    if not is_already_successful(st, ["qrel_late_encoder_heads.csv.gz", "attack_late_encoder_heads.csv.gz"]):
        raise FileNotFoundError(f"run stage 13 first ({st})")
    heads = all_encoder_heads(LAYERS)
    labels = [h.label for h in heads]
    cols = [f"enc_{l}" for l in labels]
    q = pd.read_csv(st / "qrel_late_encoder_heads.csv.gz", dtype=DT)
    a = pd.read_csv(st / "attack_late_encoder_heads.csv.gz", dtype=DT)
    for df in (q, a):
        if [c for c in df.columns if c.startswith("enc_")] != cols:
            raise RuntimeError("features must be exactly the 36 heads L9H0..L11H11 (stage 13)")
    gen = q[q.relevance_group == "relevant"]
    if not gen.qrel_grade.isin([2, 3]).all():
        raise RuntimeError("genuine reference must be qrel 2/3 only")
    if not np.allclose(a.delta_score, a.score_attack - a.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")
    atk = a[a.delta_score > 0]
    common = sorted(set(gen.qid) & set(atk.qid))
    excluded = {"genuine_only_qids": sorted(set(gen.qid) - set(common)), "attack_only_qids": sorted(set(atk.qid) - set(common))}
    gen = gen[gen.qid.isin(common)].reset_index(drop=True)
    atk = atk[atk.qid.isin(common)].reset_index(drop=True)
    fold_of = query_folds(common, N_FOLDS, SEED)
    fg, fa = gen.qid.map(fold_of).values, atk.qid.map(fold_of).values
    for f in range(N_FOLDS):                                          # qid-disjointness check
        assert not (set(gen.qid[fg == f]) & set(gen.qid[fg != f]))
        assert set(atk.qid[fa == f]) <= set(gen.qid[fg == f])
    Xg, Xa = gen[cols].values.astype(float), atk[cols].values.astype(float)

    stage = out / "15_head_anomaly"
    stage.mkdir(parents=True, exist_ok=True)
    per_k, folds_all, curves = {}, [], []
    for k in K_GRID:
        zg, za, fdf, fits = run_k(Xg, fg, Xa, fa, k)
        cg, ca = abnormal_counts(zg, k), abnormal_counts(za, k)
        per_k[k] = (zg, za, cg, ca, fdf, fits)
        folds_all.append(fdf)
        for T in range(len(labels) + 1):
            curves.append({"k": k, **threshold_metrics(ca["abnormal_head_count"], cg["abnormal_head_count"], T)})
    zg, za, cg, ca, fdf, fits = per_k[K_PRIMARY]

    # --- predictions (primary k) -------------------------------------------------------
    def pred_frame(df, z, c, group, folds):
        meta = df[["qid", "docid", "pair_id"]].copy()
        meta["attack_name"] = df["attack_name"] if "attack_name" in df else ""
        meta["delta_score"] = df["delta_score"] if "delta_score" in df else np.nan
        meta.insert(0, "group", group)
        meta["fold"] = folds
        raw = pd.DataFrame(df[cols].values, columns=[f"sim_{l}" for l in labels])
        zz = pd.DataFrame(z, columns=[f"z_{l}" for l in labels])
        cc = pd.DataFrame({key: v for key, v in c.items()})
        return pd.concat([meta, raw, zz, cc], axis=1)
    pred = pd.concat([pred_frame(gen, zg, cg, GEN, fg), pred_frame(atk, za, ca, ATK, fa)], ignore_index=True)
    pred.to_csv(stage / "anomaly_predictions.csv.gz", index=False)

    # --- per head (primary k) ----------------------------------------------------------
    mus = np.stack([fits[f][0] for f in range(N_FOLDS)]); sds = np.stack([fits[f][1] for f in range(N_FOLDS)])
    canon = {h.label for h in encoder_heads()}
    ph = pd.DataFrame({
        "head_label": labels, "layer": [h.layer for h in heads], "head": [h.head_idx for h in heads],
        "mu_train_mean_over_folds": mus.mean(0), "sigma_train_mean_over_folds": sds.mean(0),
        "sigma_train_min_over_folds": sds.min(0),
        "genuine_abnormal_rate": np.mean(np.abs(np.nan_to_num(zg)) > K_PRIMARY, 0),
        "attack_abnormal_rate": np.mean(np.abs(np.nan_to_num(za)) > K_PRIMARY, 0),
        "attack_low_rate": np.mean(np.nan_to_num(za) < -K_PRIMARY, 0),
        "attack_high_rate": np.mean(np.nan_to_num(za) > K_PRIMARY, 0),
        "genuine_low_rate": np.mean(np.nan_to_num(zg) < -K_PRIMARY, 0),
        "genuine_high_rate": np.mean(np.nan_to_num(zg) > K_PRIMARY, 0),
        "mean_signed_z_genuine": np.nanmean(zg, 0), "mean_signed_z_attack": np.nanmean(za, 0),
        "previously_important_head": [l in canon for l in labels],
    })
    ph["attack_minus_genuine_rate"] = ph.attack_abnormal_rate - ph.genuine_abnormal_rate
    ph["rank_by_rate_difference"] = ph.attack_minus_genuine_rate.rank(ascending=False, method="first").astype(int)
    ph.to_csv(stage / "per_head_abnormality.csv", index=False)
    pd.DataFrame(curves).to_csv(stage / "detector_threshold_curve.csv", index=False)
    fold_summary = pd.concat(folds_all, ignore_index=True)
    fold_summary.to_csv(stage / "fold_summary.csv", index=False)

    # --- summary -----------------------------------------------------------------------
    def cv_detector(fdf):
        tot = fdf[["tp", "fp", "tn", "fn"]].sum()
        tpr, fpr = tot.tp / (tot.tp + tot.fn), tot.fp / (tot.fp + tot.tn)
        return {"T_chosen_per_fold": fdf.T_chosen_inner_cv.tolist(), "tp": int(tot.tp), "fp": int(tot.fp),
                "tn": int(tot.tn), "fn": int(tot.fn), "tpr_recall": tpr, "fpr": fpr, "specificity": 1 - fpr,
                "precision": tot.tp / (tot.tp + tot.fp) if tot.tp + tot.fp else None,
                "balanced_accuracy": (tpr + 1 - fpr) / 2,
                "accuracy": (tot.tp + tot.tn) / tot.sum()}
    # equal-attack-configuration recall at each fold's chosen T
    Tmap = dict(zip(fdf.fold, fdf.T_chosen_inner_cv))
    flagged = ca["abnormal_head_count"] >= np.array([Tmap[f] for f in fa])
    eq_recall = float(pd.Series(flagged).groupby(atk.attack_name.values).mean().mean())
    at = lambda T: threshold_metrics(ca["abnormal_head_count"], cg["abnormal_head_count"], T)  # noqa: E731
    low_share = float(ca["n_low_abnormal"].sum() / max(1, ca["abnormal_head_count"].sum()))
    summary = {
        "created": now(), "feature_files": [str(st / "qrel_late_encoder_heads.csv.gz"),
                                            str(st / "attack_late_encoder_heads.csv.gz")],
        "heads": labels, "n_heads": len(labels), "k_primary": K_PRIMARY, "k_grid": K_GRID,
        "n_folds": N_FOLDS, "seed": SEED, "n_common_queries": len(common), "excluded": excluded,
        "fold_sizes_queries": np.bincount(list(fold_of.values())).tolist(),
        "n_genuine_test": int(len(gen)), "n_successful_attack_test": int(len(atk)),
        "n_attack_configs": int(atk.attack_name.nunique()),
        "heads_sigma_below_min_any_fold": int(fdf.n_heads_sigma_below_min.max()),
        "abnormal_head_count_distribution": {GEN: dist(cg["abnormal_head_count"]), ATK: dist(ca["abnormal_head_count"])},
        "auroc_abnormal_head_count_pooled": auroc(ca["abnormal_head_count"], cg["abnormal_head_count"]),
        "auroc_abnormal_head_count_fold_mean": float(fdf.auroc_abnormal_head_count.mean()),
        "auroc_mean_abs_z_pooled": auroc(ca["mean_abs_z"], cg["mean_abs_z"]),
        "detector_inner_cv_T": cv_detector(fdf) | {"recall_equal_attack_config_weight": eq_recall},
        "threshold_examples": {f"T={T}": {m: at(T)[m] for m in ("tpr_recall", "fpr", "precision", "balanced_accuracy")}
                               for T in (1, 3, 5, 10, 15)},
        "abnormality_direction_attacks": {"share_of_abnormal_head_events_below": low_share,
                                          "share_above": 1 - low_share},
        "top_heads_by_rate_difference": ph.sort_values("attack_minus_genuine_rate", ascending=False)
            .head(8)[["head_label", "genuine_abnormal_rate", "attack_abnormal_rate", "attack_low_rate", "attack_high_rate",
                      "mean_signed_z_attack"]].round(4).to_dict("records"),
        "sensitivity_k": [{"k": k, "mean_count_genuine": float(per_k[k][2]["abnormal_head_count"].mean()),
                           "mean_count_attack": float(per_k[k][3]["abnormal_head_count"].mean()),
                           "auroc_count": auroc(per_k[k][3]["abnormal_head_count"], per_k[k][2]["abnormal_head_count"]),
                           **{f"cv_{m}": v for m, v in cv_detector(per_k[k][4]).items()
                              if m in ("T_chosen_per_fold", "tpr_recall", "fpr", "balanced_accuracy")}}
                          for k in K_GRID],
    }
    (stage / "overall_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # --- plots ---------------------------------------------------------------------------
    pdir = out / "plots"
    Cg, Ca = cg["abnormal_head_count"], ca["abnormal_head_count"]
    xmax = int(max(Cg.max(), Ca.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.arange(xmax + 2) - 0.5
    for C, color, lab, off in ((Cg, P5.BLUE, f"genuine relevant, held-out (n={len(Cg):,})", -0.2),
                               (Ca, P5.ORANGE, f"successful attacks, held-out (n={len(Ca):,})", 0.2)):
        h, _ = np.histogram(C, bins=bins)
        axes[0].bar(np.arange(xmax + 1) + off, h / len(C), width=0.4, color=color, label=lab)
        xs = np.sort(C)
        axes[1].step(np.concatenate([[0], xs]), np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs)]),
                     where="post", color=color, lw=2, label=lab)
        axes[0].axvline(np.median(C), color=color, ls="--", lw=1.2)
    axes[0].set_xlabel("abnormal_head_count  (# of 36 heads with |z| > 2)")
    axes[0].set_ylabel("fraction of examples")
    axes[0].set_title("Distribution (dashed = median)", loc="left", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("abnormal_head_count")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title(f"ECDF; AUROC = {summary['auroc_abnormal_head_count_pooled']:.3f}", loc="left", fontsize=9)
    fig.suptitle("Do successful attacks activate more abnormal L9–L11 heads than genuine relevant inputs?  "
                 "(reference = genuine qrel 2/3 of training queries; 5-fold query CV)", x=0.01, ha="left", fontsize=9)
    P5._save(fig, pdir / "fig_anomaly_head_count_distribution.png")

    X = np.arange(len(labels))

    def head_axis(ax):
        ax.set_xticks(X)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        for lab in ax.get_xticklabels():
            if lab.get_text() in canon:
                lab.set_fontweight("bold")
        for b in (12, 24):
            ax.axvline(b - 0.5, color=P5.INK2, lw=1.0, ls=":", zorder=1)
        ax.set_xlim(-0.6, len(X) - 0.4)
        ax.set_xlabel("encoder head (bold = canonical important head)")

    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.plot(X, ph.genuine_abnormal_rate, color=P5.BLUE, lw=2, marker="o", ms=4, label="genuine relevant (held-out): |z| > 2")
    ax.plot(X, ph.attack_abnormal_rate, color=P5.ORANGE, lw=2, marker="s", ms=4, label="successful attacks: |z| > 2")
    ax.plot(X, ph.attack_low_rate, color=P5.ORANGE, lw=1.2, ls="--", label="successful attacks: z < −2 (below)")
    ax.plot(X, ph.attack_high_rate, color=P5.ORANGE, lw=1.2, ls=":", label="successful attacks: z > +2 (above)")
    ax.axhline(0.0455, color=P5.INK2, lw=0.8, ls="-.", label="4.6% (Gaussian |z| > 2)")
    head_axis(ax)
    ax.set_ylabel("fraction of examples abnormal at head")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Which heads flag attacks? Per-head abnormality rate (|z| > 2 vs the genuine-relevant reference)",
                 loc="left", fontsize=9)
    P5._save(fig, pdir / "fig_anomaly_per_head_abnormal_rate.png")

    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axhspan(-K_PRIMARY, K_PRIMARY, color=P5.GRID, alpha=0.5, zorder=0, label="|z| ≤ 2")
    ax.axhline(0, color=P5.INK2, lw=0.8)
    ax.plot(X, ph.mean_signed_z_genuine, color=P5.BLUE, lw=2, marker="o", ms=4, label="genuine relevant (held-out)")
    ax.plot(X, ph.mean_signed_z_attack, color=P5.ORANGE, lw=2, marker="s", ms=4, label="successful attacks")
    head_axis(ax)
    ax.set_ylabel("mean signed z  (vs genuine-relevant reference)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Are attacks below or above the genuine-relevance distribution? Mean signed z per head",
                 loc="left", fontsize=9)
    P5._save(fig, pdir / "fig_anomaly_mean_signed_z.png")

    cur = pd.DataFrame([c for c in curves if c["k"] == K_PRIMARY])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(cur["T"], cur.tpr_recall, color=P5.ORANGE, lw=2, marker="s", ms=3, label="TPR (attacks flagged)")
    axes[0].plot(cur["T"], cur.fpr, color=P5.BLUE, lw=2, marker="o", ms=3, label="FPR (genuine flagged)")
    axes[0].plot(cur["T"], cur.balanced_accuracy, color=P5.INK2, lw=1.5, ls="--", label="balanced accuracy")
    for T in sorted(set(fdf.T_chosen_inner_cv)):
        axes[0].axvline(T, color=P5.AQUA, lw=1, ls=":")
    axes[0].set_xlabel("threshold T  (flag if abnormal_head_count ≥ T)")
    axes[0].set_ylabel("rate")
    axes[0].set_title("Threshold curve, k = 2 (dotted = T chosen on training queries)", loc="left", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].plot(cur.fpr, cur.tpr_recall, color=P5.ORANGE, lw=2, marker="s", ms=3,
                 label=f"abnormal_head_count (AUROC {summary['auroc_abnormal_head_count_pooled']:.3f})")
    s_g, s_a = cg["mean_abs_z"], ca["mean_abs_z"]
    thr = np.unique(np.concatenate([s_g, s_a]))[::-1]
    axes[1].plot([np.mean(s_g >= t) for t in thr], [np.mean(s_a >= t) for t in thr], color=P5.INK2, lw=1.5, ls="--",
                 label=f"mean |z| (secondary, AUROC {summary['auroc_mean_abs_z_pooled']:.3f})")
    axes[1].plot([0, 1], [0, 1], color=P5.GRID, lw=1)
    axes[1].set_xlabel("false positive rate (genuine relevant)")
    axes[1].set_ylabel("true positive rate (successful attacks)")
    axes[1].set_title("ROC", loc="left", fontsize=9)
    axes[1].legend(fontsize=8, loc="lower right")
    P5._save(fig, pdir / "fig_anomaly_threshold_curve.png")

    write_status(stage, {"status": "success", "finished": now()})
    print(json.dumps({k: summary[k] for k in ("n_common_queries", "n_genuine_test", "n_successful_attack_test",
                                              "abnormal_head_count_distribution", "auroc_abnormal_head_count_pooled",
                                              "auroc_mean_abs_z_pooled")}, indent=1, default=float))
    print(f"[15] -> {stage}; plots fig_anomaly_*.png")


if __name__ == "__main__":
    main()
