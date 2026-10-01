"""
scripts/20_paired_detector.py
==============================
Stage 20 — paired top-k head detector, evaluated SEPARATELY for relevant (qrel 2/3) and
non-relevant (qrel 0) base documents (no model; reads 18_paired_forward/ and checks 19).

  positive = attacked input of a successful instance
  negative = THAT instance's own padded control (never another document)
  score    = # abnormal heads (|z| > 2) among the top-k heads, k = 1..N_HEADS (36 for L9-L11, 144 for all layers)

Outer query fold f (stage-17 fold map). Everything except the final scoring of fold f uses
ONLY the training folds (inner CV, exp16lib.anomaly.inner_heldout_z): for every inner fold
g != f the reference (successful qrel 2/3 controls, all attacks pooled) is refitted on the
other training folds and fold-g positives/negatives are z-scored. From these inner held-out
z's (exp16lib.paired.topk_training_selection):
  head ranking  excess abnormality rate (pos - neg)       (= anomaly.inner_head_ranking rule)
  T_k           argmax balanced accuracy of count >= T     (= anomaly.choose_threshold rule)
  k*            argmax inner AUROC (ties -> smallest k)    (single-k report only)
Fold f is scored against the reference fitted on all training folds (= stage 19 z).

Metrics pooled over the 5 held-out folds: AUROC (threshold-free; also per-fold mean/min/max),
precision / recall / F1 / FPR / balanced accuracy at the training-chosen T_k, and the paired
win rate P(count_attack > count_control) + 0.5 P(tie) within instance (descriptive).

Outputs (20_paired_detector/): topk_detector_metrics.csv, detector_fold_details.csv,
selected_heads.csv, detector_summary.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.anomaly import auroc, inner_heldout_z, threshold_metrics  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_status, now, stage_argparser, write_status  # noqa: E402

K = P.K_ABNORMAL


def evaluate_group(Xc, Xa, Zc, Za, fold, ref, m, labels):
    """Returns (per-k pooled rows, per-fold rows, selected-head rows, k* result)."""
    KS = list(range(1, P.N_HEADS + 1))                      # scope set by validate_cfg
    Xref, fref = Xc[ref], fold[ref]
    Xp, Xn, fg = Xa[m], Xc[m], fold[m]
    Zp, Zn = Za[m], Zc[m]
    cnt_p = {k: np.full(m.sum(), -1) for k in KS}
    cnt_n = {k: np.full(m.sum(), -1) for k in KS}
    Tf, fold_rows, sel_rows, kstar = {}, [], [], {}
    for f in range(P.N_FOLDS):
        train = [x for x in range(P.N_FOLDS) if x != f]
        assert f not in train
        zp_in, zn_in = inner_heldout_z(Xref, fref, [Xp, Xn], [fg, fg], train)     # training folds only
        order, T, inner_auc = P.topk_training_selection(zp_in, zn_in, K, KS)
        kstar[f] = max(KS, key=lambda k: (round(inner_auc[k], 12), -k))
        Tf[f] = T
        for r, j in enumerate(order):
            sel_rows.append({"fold": f, "rank": r + 1, "head_label": labels[j]})
        te = fg == f
        for k in KS:
            sel = order[:k]
            cp = (np.abs(np.nan_to_num(Zp[te][:, sel])) > K).sum(1)
            cn = (np.abs(np.nan_to_num(Zn[te][:, sel])) > K).sum(1)
            cnt_p[k][te], cnt_n[k][te] = cp, cn
            tm = threshold_metrics(cp, cn, T[k]) if te.any() else {}
            fold_rows.append({"fold": f, "k_heads": k, "T_train": T[k], "inner_auroc_train": inner_auc[k],
                              "is_k_star": k == kstar[f], "n_test_pairs": int(te.sum()),
                              "auroc_test": auroc(cp, cn) if te.any() else np.nan,
                              **{kk: tm.get(kk) for kk in ("tp", "fp", "tn", "fn")},
                              **P.f1_from(tm.get("tp", 0), tm.get("fp", 0), tm.get("fn", 0))})
    fr = pd.DataFrame(fold_rows)
    rows = []
    for k in KS:
        assert (cnt_p[k] >= 0).all() and (cnt_n[k] >= 0).all()
        d = fr[fr.k_heads == k]
        tp, fp, tn, fn = (int(d[c].sum()) for c in ("tp", "fp", "tn", "fn"))
        diff = cnt_p[k] - cnt_n[k]
        rows.append({"k_heads": k, "auroc_pooled": auroc(cnt_p[k], cnt_n[k]),
                     "auroc_fold_mean": d.auroc_test.mean(), "auroc_fold_min": d.auroc_test.min(),
                     "auroc_fold_max": d.auroc_test.max(), **P.f1_from(tp, fp, fn),
                     "fpr": fp / (fp + tn) if fp + tn else np.nan,
                     "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else np.nan,
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn, "T_per_fold": json.dumps([Tf[f][k] for f in range(P.N_FOLDS)]),
                     "paired_win_rate": float((diff > 0).mean() + 0.5 * (diff == 0).mean()),
                     "n_pairs": int(m.sum())})
    # single-k report: each fold uses its own training-chosen k*
    ps, ns = [], []
    tp = fp = tn = fn = 0
    for f in range(P.N_FOLDS):
        te = fg == f
        k = kstar[f]
        ps.append(cnt_p[k][te]); ns.append(cnt_n[k][te])
        d = fr[(fr.fold == f) & (fr.k_heads == k)].iloc[0]
        tp += int(d.tp); fp += int(d.fp); tn += int(d.tn); fn += int(d.fn)
    ks_res = {"k_star_per_fold": [kstar[f] for f in range(P.N_FOLDS)],
              "auroc_pooled": auroc(np.concatenate(ps), np.concatenate(ns)), **P.f1_from(tp, fp, fn),
              "fpr": fp / (fp + tn) if fp + tn else np.nan, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return pd.DataFrame(rows), fr, pd.DataFrame(sel_rows), ks_res


def main():
    args = stage_argparser("Exp 16 stage 20: paired top-k detector").parse_args()
    cfg = load_config(args.config)
    P.validate_cfg(cfg)
    out = output_dir(cfg)
    fwd, st19 = out / P.FORWARD_DIR, out / P.ANOMALY_DIR
    if not is_already_successful(st19, ["paired_instances.csv.gz"]):
        raise FileNotFoundError(f"run stage 19 first ({st19})")
    attacks = list(load_status(out / P.MANIFEST_DIR)["sha256"]["attacks"])
    labels = P.head_labels()
    hc_c, hc_a = P.head_cols("control"), P.head_cols("attack")
    df = P.load_forward_rows(fwd, attacks, usecols=P.INSTANCE_META + ["score_control", "score_attack", "delta_score",
                                                                        "successful"] + hc_c + hc_a)
    P.check_forward_frame(df)
    s = df[df.successful].reset_index(drop=True)
    inst = pd.read_csv(st19 / "paired_instances.csv.gz", dtype=P.DT)
    if not (inst.attack_name.values == s.attack_name.values).all() or not (inst.pair_id.values == s.pair_id.values).all():
        raise RuntimeError("stage 19 rows are not aligned with the successful forward rows")
    fold = inst.fold.values.astype(int)
    Xc, Xa = s[hc_c].values.astype(float), s[hc_a].values.astype(float)
    ref = P.reference_mask(s)
    Zc, _ = P.outer_zscores(Xc, fold, Xc[ref], fold[ref])
    Za, _ = P.outer_zscores(Xa, fold, Xc[ref], fold[ref])
    for Z, side in ((Zc, "control"), (Za, "attack")):          # identical to stage 19 (up to its 6-digit CSV)
        saved = inst[[f"z_{side}_{l}" for l in labels]].values
        if not np.allclose(np.nan_to_num(Z), np.nan_to_num(saved), rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"recomputed {side} z differs from stage 19")

    stage = out / P.DETECTOR_DIR
    stage.mkdir(parents=True, exist_ok=True)
    allk, allf, alls, summ = [], [], [], {}
    report_k = sorted({1, 3, 5, 10, 36, P.N_HEADS} & set(range(1, P.N_HEADS + 1)))
    for g in P.GROUPS:
        m = (s.relevance_group == g).values
        if not m.any():
            continue
        rows, fr, sel, ks_res = evaluate_group(Xc, Xa, Zc, Za, fold, ref, m, labels)
        allk.append(rows.assign(relevance_group=g)); allf.append(fr.assign(relevance_group=g))
        alls.append(sel.assign(relevance_group=g))
        r_all = rows[rows.k_heads == P.N_HEADS].iloc[0]
        summ[g] = {"n_pairs": int(m.sum()), "n_queries": int(s.qid[m].nunique()),
                   "all_heads": r_all.drop(["k_heads"]).to_dict(),
                   **({"all_36_heads": r_all.drop(["k_heads"]).to_dict()} if P.N_HEADS == 36 else {}),
                   "cv_selected_k": ks_res,
                   "topk": rows[rows.k_heads.isin(report_k)][["k_heads", "auroc_pooled", "auroc_fold_mean", "precision",
                                                              "recall", "f1", "fpr", "paired_win_rate"]].to_dict("records"),
                   "best_k_by_heldout_auroc_DESCRIPTIVE_ONLY": int(rows.loc[rows.auroc_pooled.idxmax(), "k_heads"]),
                   "top10_heads_per_fold": {int(f): sel[(sel.fold == f) & (sel["rank"] <= 10)].head_label.tolist()
                                            for f in range(P.N_FOLDS)}}
    first = lambda d: d[["relevance_group"] + [c for c in d.columns if c != "relevance_group"]]  # noqa: E731
    first(pd.concat(allk, ignore_index=True)).to_csv(stage / "topk_detector_metrics.csv", index=False)
    first(pd.concat(allf, ignore_index=True)).to_csv(stage / "detector_fold_details.csv", index=False)
    first(pd.concat(alls, ignore_index=True)).to_csv(stage / "selected_heads.csv", index=False)
    summary = {"created": now(), "scope": P.SCOPE, "layers": P.LAYERS, "n_heads": P.N_HEADS, "k_abnormal": K, "threshold_rule": "argmax balanced accuracy on inner held-out "
               "training folds (anomaly.choose_threshold); flag if top-k abnormal count >= T",
               "positive": "attacked input", "negative": "the same instance's padded control", "groups": summ}
    (stage / "detector_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_status(stage, {"status": "success", "finished": now()})
    for g, v in summ.items():
        print(f"[20] {g}: n_pairs={v['n_pairs']} all-{P.N_HEADS} AUROC={v['all_heads']['auroc_pooled']:.3f} "
              f"F1={v['all_heads']['f1']:.3f}; k*={v['cv_selected_k']['k_star_per_fold']} "
              f"AUROC={v['cv_selected_k']['auroc_pooled']:.3f} F1={v['cv_selected_k']['f1']:.3f}")
    print(f"[20] -> {stage}")


if __name__ == "__main__":
    main()
