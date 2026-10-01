"""
scripts/19_paired_anomaly.py
=============================
Stage 19 — paired abnormality of SUCCESSFUL attacks (no model; reads 18_paired_forward[_all_layers]/).
Head scope = paired.scope (36 heads L9-L11, or 144 heads L0-L11); abnormal_count is out of N_HEADS.

For outer query fold f (fold map from stage 17: query_folds over all judged attackable qids,
5 folds, seed 42):
  reference_f = CONTROL head sims of successful qrel 2/3 instances whose qid is NOT in fold f,
                pooled over ALL attack configurations (no per-token / position / repetition fit)
  mu_h, sigma_h (ddof=1) per head = exp16lib.anomaly.fit_reference(reference_f)
  every successful instance of fold f (relevant AND non-relevant): z_h for its control and its attack
  abnormal_h = |z_h| > 2;  abnormal_count = # of the 36 heads abnormal

Per instance: control/attack abnormal count, paired change (attack - control), low/high counts,
and the four per-head transitions normal/abnormal -> normal/abnormal (+ low/high split).
Relevant and non-relevant base documents are summarised SEPARATELY, at instance level and at
equal-document weight (per (qid, docid) aggregate over its successful attacks, then across docs).
Secondary (descriptive, flagged): one-sided query-level sign-flip test of the mean paired change.

Outputs (19_paired_anomaly/):
  paired_instances.csv.gz       one row per successful instance: meta, fold, z_control_*, z_attack_*, counts
  reference_fits.csv            fold x head: mu, sigma, valid, n reference rows / docs / qids / attack configs
  layer_summary.csv             group x layer: per-layer abnormal counts, paired change (+ query-cluster SE), rates
  per_head_paired.csv           group x head: control/attack abnormal (low/high) rates, transitions, mean z
  paired_count_summary.csv      group: instance-level paired abnormal-count statistics
  count_distribution.csv        group x count: fraction of instances (control, attack); change distribution
  document_level.csv.gz         one row per (qid, docid) with >= 1 successful attack
  document_level_summary.csv    group: equal-document-weight summary
  low_high_summary.csv          group: low vs high shares of abnormal events and of attack-created abnormalities
  per_attack_paired.csv         group x attack configuration: descriptive breakdown
  success_counts.csv            group x attack: instances, successes, success rate
  summary.json
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
from exp16lib.anomaly import abnormal_counts, query_folds  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_status, now, stage_argparser, write_status  # noqa: E402
from exp16lib.stats import sign_flip_test  # noqa: E402

K = P.K_ABNORMAL
META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid", "docid",
        "qrel_grade", "relevance_group", "score_control", "score_attack", "delta_score", "successful"]


def load_folds(out):
    fm = json.loads((out / P.MANIFEST_DIR / "fold_map.json").read_text())
    if query_folds(fm["qids"], P.N_FOLDS, P.FOLD_SEED) != fm["fold_of"]:
        raise RuntimeError("fold map is not query_folds(qids, 5, 42)")
    return fm["fold_of"]


def main():
    args = stage_argparser("Exp 16 stage 19: paired abnormality").parse_args()
    cfg = load_config(args.config)
    pc = P.validate_cfg(cfg)
    out = output_dir(cfg)
    fwd = out / P.FORWARD_DIR
    if not is_already_successful(fwd, ["per_attack_status.csv"]):
        raise FileNotFoundError(f"run stage 18 first ({fwd})")
    attacks = list(load_status(out / P.MANIFEST_DIR)["sha256"]["attacks"])
    labels = P.head_labels()
    hc_c, hc_a = P.head_cols("control"), P.head_cols("attack")
    df = P.load_forward_rows(fwd, attacks, usecols=P.INSTANCE_META + ["score_control", "score_attack", "delta_score",
                                                                        "successful"] + hc_c + hc_a)
    P.check_forward_frame(df)
    fold_of = load_folds(out)
    df["fold"] = df.qid.map(fold_of)
    if df.fold.isna().any():
        raise RuntimeError("instance qid missing from the fold map")
    df["fold"] = df.fold.astype(int)

    # ---- success counts (all aligned instances) ---------------------------------------------
    sc = (df.groupby(["relevance_group", "attack_name"]).agg(n_instances=("successful", "size"),
                                                             n_successful=("successful", "sum")).reset_index())
    sc["success_rate"] = sc.n_successful / sc.n_instances
    stage = out / P.ANOMALY_DIR
    stage.mkdir(parents=True, exist_ok=True)
    sc.to_csv(stage / "success_counts.csv", index=False)

    s = df[df.successful].reset_index(drop=True)
    fold = s.fold.values
    Xc, Xa = s[hc_c].values.astype(float), s[hc_a].values.astype(float)

    # ---- reference: successful qrel 2/3 CONTROLS of training queries, all attacks pooled ---
    ref = P.reference_mask(s)
    assert s.relevance_group[ref].eq(P.RELEVANT).all() and s.qrel_grade[ref].isin([2, 3]).all()
    assert (s.delta_score[ref] > 0).all()
    Zc, fits = P.outer_zscores(Xc, fold, Xc[ref], fold[ref])
    Za, _ = P.outer_zscores(Xa, fold, Xc[ref], fold[ref])
    rf = []
    for f in range(P.N_FOLDS):
        r_rows = ref & (fold != f)
        assert not set(s.qid[r_rows]) & set(s.qid[fold == f])        # held-out queries never in mu/sigma
        mu, sd, ok = fits[f]
        for j, l in enumerate(labels):
            rf.append({"fold": f, "head_label": l, "mu": mu[j], "sigma": sd[j], "valid": bool(ok[j]),
                       "n_reference_rows": int(r_rows.sum()), "n_reference_docs": int(s.pair_id[r_rows].nunique()),
                       "n_reference_qids": int(s.qid[r_rows].nunique()),
                       "n_reference_attack_configs": int(s.attack_name[r_rows].nunique())})
    rf = pd.DataFrame(rf)
    rf.to_csv(stage / "reference_fits.csv", index=False)

    # ---- per-instance counts / transitions --------------------------------------------------
    cc, ca = abnormal_counts(Zc, K), abnormal_counts(Za, K)
    Sc, Sa = P.states(Zc), P.states(Za)
    assert (np.abs(Sc).sum(1) == cc["abnormal_head_count"]).all()   # |z| > 2 rule, shared with anomaly.py
    tr = P.transition_counts(Sc, Sa)
    inst = s[META + ["fold"]].copy()
    inst["control_abnormal_count"] = cc["abnormal_head_count"]
    inst["attack_abnormal_count"] = ca["abnormal_head_count"]
    inst["abnormal_count_change"] = inst.attack_abnormal_count - inst.control_abnormal_count
    inst["control_n_low"], inst["control_n_high"] = cc["n_low_abnormal"], cc["n_high_abnormal"]
    inst["attack_n_low"], inst["attack_n_high"] = ca["n_low_abnormal"], ca["n_high_abnormal"]
    inst["control_mean_abs_z"], inst["attack_mean_abs_z"] = cc["mean_abs_z"], ca["mean_abs_z"]
    for k, v in tr.items():
        inst[k] = v
    zdf = pd.DataFrame(np.hstack([Zc, Za]), columns=[f"z_control_{l}" for l in labels] + [f"z_attack_{l}" for l in labels])
    inst = pd.concat([inst, zdf], axis=1)
    inst.to_csv(stage / "paired_instances.csv.gz", index=False, float_format="%.6g")

    # ---- per group summaries ----------------------------------------------------------------
    ph, cs, dist, lh, pa, dsum, ls = [], [], [], [], [], [], []
    hs = P.heads()
    head_layer = np.array([h.layer for h in hs])
    docs = P.doc_level(inst)
    docs.to_csv(stage / "document_level.csv.gz", index=False)
    group_json = {}
    for gi, g in enumerate(P.GROUPS):
        m = (inst.relevance_group == g).values
        if not m.any():
            continue
        zc, za, sc_, sa_ = Zc[m], Za[m], Sc[m], Sa[m]
        a0, a1 = sc_ != 0, sa_ != 0
        for j, l in enumerate(labels):
            ph.append({"relevance_group": g, "head_label": l, "layer": hs[j].layer, "head": hs[j].head_idx,
                       "control_abnormal_rate": a0[:, j].mean(), "attack_abnormal_rate": a1[:, j].mean(),
                       "paired_rate_difference": a1[:, j].mean() - a0[:, j].mean(),
                       "control_low_rate": (sc_[:, j] < 0).mean(), "control_high_rate": (sc_[:, j] > 0).mean(),
                       "attack_low_rate": (sa_[:, j] < 0).mean(), "attack_high_rate": (sa_[:, j] > 0).mean(),
                       "rate_normal_to_normal": (~a0[:, j] & ~a1[:, j]).mean(),
                       "rate_normal_to_abnormal": (~a0[:, j] & a1[:, j]).mean(),
                       "rate_abnormal_to_normal": (a0[:, j] & ~a1[:, j]).mean(),
                       "rate_abnormal_to_abnormal": (a0[:, j] & a1[:, j]).mean(),
                       "rate_normal_to_abnormal_low": (~a0[:, j] & (sa_[:, j] < 0)).mean(),
                       "rate_normal_to_abnormal_high": (~a0[:, j] & (sa_[:, j] > 0)).mean(),
                       "rate_abnormal_low_to_normal": ((sc_[:, j] < 0) & ~a1[:, j]).mean(),
                       "rate_abnormal_high_to_normal": ((sc_[:, j] > 0) & ~a1[:, j]).mean(),
                       "mean_z_control": np.nanmean(zc[:, j]), "mean_z_attack": np.nanmean(za[:, j]),
                       "mean_paired_z_change": np.nanmean(za[:, j] - zc[:, j])})
        gi_ = inst[m]
        for L in P.LAYERS:                                  # where is any paired effect strongest?
            hl = head_layer == L
            c_L, a_L = a0[:, hl].sum(1), a1[:, hl].sum(1)
            d_L = a_L - c_L
            na, an = (~a0[:, hl] & a1[:, hl]), (a0[:, hl] & ~a1[:, hl])
            ls.append({"relevance_group": g, "layer": L, "n_heads": int(hl.sum()),
                       "mean_control_abnormal_count": c_L.mean(), "mean_attack_abnormal_count": a_L.mean(),
                       "mean_paired_change": d_L.mean(), "se_paired_change_query_cluster": P.cluster_se(d_L, gi_.qid.values),
                       "frac_change_positive": (d_L > 0).mean(), "frac_change_negative": (d_L < 0).mean(),
                       "control_abnormal_rate": a0[:, hl].mean(), "attack_abnormal_rate": a1[:, hl].mean(),
                       "rate_normal_to_abnormal": na.mean(), "rate_abnormal_to_normal": an.mean(),
                       "rate_normal_to_abnormal_low": (na & (sa_[:, hl] < 0)).mean(),
                       "rate_normal_to_abnormal_high": (na & (sa_[:, hl] > 0)).mean(),
                       "control_share_low": (sc_[:, hl] < 0).sum() / max(1, a0[:, hl].sum()),
                       "attack_share_low": (sa_[:, hl] < 0).sum() / max(1, a1[:, hl].sum()),
                       "mean_z_control": float(np.nanmean(zc[:, hl])), "mean_z_attack": float(np.nanmean(za[:, hl]))})
        ch = gi_.abnormal_count_change.values
        per_q = gi_.groupby("qid").abnormal_count_change.mean()
        sf = sign_flip_test(per_q.values, int(pc["n_sign_flips"]), int(cfg["statistics"]["sign_flip_seed"]), stream=gi)
        row = {"relevance_group": g, "n_successful_instances": int(m.sum()), "n_docs": int(gi_.pair_id.nunique()),
               "n_queries": int(gi_.qid.nunique()), "n_attack_configs": int(gi_.attack_name.nunique()),
               "mean_control_abnormal_count": gi_.control_abnormal_count.mean(),
               "mean_attack_abnormal_count": gi_.attack_abnormal_count.mean(),
               "median_control_abnormal_count": gi_.control_abnormal_count.median(),
               "median_attack_abnormal_count": gi_.attack_abnormal_count.median(),
               "mean_paired_change": ch.mean(), "median_paired_change": float(np.median(ch)),
               "sd_paired_change": ch.std(ddof=1) if len(ch) > 1 else np.nan,
               "se_paired_change_query_cluster": P.cluster_se(ch, gi_.qid.values),
               "frac_change_positive": (ch > 0).mean(), "frac_change_zero": (ch == 0).mean(),
               "frac_change_negative": (ch < 0).mean(),
               "frac_control_any_abnormal": (gi_.control_abnormal_count > 0).mean(),
               "frac_attack_any_abnormal": (gi_.attack_abnormal_count > 0).mean(),
               "query_level_mean_change": float(per_q.mean()),
               "query_level_signflip_p_one_sided_gt0": sf["p_one_sided"], "n_sign_flips": sf["n_flips"]}
        cs.append(row)
        vmax = P.N_HEADS
        for v in range(vmax + 1):
            dist.append({"relevance_group": g, "value": v, "kind": "abnormal_count",
                         "frac_control": (gi_.control_abnormal_count == v).mean(),
                         "frac_attack": (gi_.attack_abnormal_count == v).mean()})
        for v in range(-vmax, vmax + 1):
            dist.append({"relevance_group": g, "value": v, "kind": "paired_change", "frac": (ch == v).mean()})
        ev_c, ev_a = a0.sum(), a1.sum()
        na_low, na_high = (~a0 & (sa_ < 0)).sum(), (~a0 & (sa_ > 0)).sum()
        an_low, an_high = ((sc_ < 0) & ~a1).sum(), ((sc_ > 0) & ~a1).sum()
        lh.append({"relevance_group": g,
                   "control_abnormal_events": int(ev_c), "control_share_low": (sc_ < 0).sum() / ev_c if ev_c else np.nan,
                   "attack_abnormal_events": int(ev_a), "attack_share_low": (sa_ < 0).sum() / ev_a if ev_a else np.nan,
                   "normal_to_abnormal_events": int(na_low + na_high),
                   "normal_to_abnormal_share_low": na_low / (na_low + na_high) if na_low + na_high else np.nan,
                   "normal_to_abnormal_share_high": na_high / (na_low + na_high) if na_low + na_high else np.nan,
                   "abnormal_to_normal_events": int(an_low + an_high),
                   "abnormal_to_normal_share_from_low": an_low / (an_low + an_high) if an_low + an_high else np.nan,
                   "abnormal_to_abnormal_events": int((a0 & a1).sum()),
                   "abnormal_to_abnormal_sign_flips": int(((sc_ * sa_) < 0).sum())})
        pa.append(gi_.groupby(["attack_name", "attack_token", "attack_position", "repetitions"]).agg(
            n_successful=("pair_id", "size"), mean_control_abnormal_count=("control_abnormal_count", "mean"),
            mean_attack_abnormal_count=("attack_abnormal_count", "mean"),
            mean_paired_change=("abnormal_count_change", "mean"), mean_delta_score=("delta_score", "mean"))
            .reset_index().assign(relevance_group=g))
        d = docs[docs.relevance_group == g]
        n_docs_group = int(df[df.relevance_group == g].pair_id.nunique())
        dsum.append({"relevance_group": g, "n_docs_with_success": len(d), "n_docs_in_group": n_docs_group,
                     "mean_n_successful_attacks": d.n_successful_attacks.mean(),
                     "median_n_successful_attacks": d.n_successful_attacks.median(),
                     "mean_control_abnormal_count": d.mean_control_abnormal_count.mean(),
                     "mean_attack_abnormal_count": d.mean_attack_abnormal_count.mean(),
                     "mean_paired_change": d.mean_paired_change.mean(),
                     "median_of_doc_mean_change": d.mean_paired_change.median(),
                     "mean_of_doc_median_change": d.median_paired_change.mean(),
                     "se_paired_change_query_cluster": P.cluster_se(d.mean_paired_change.values, d.qid.values),
                     "frac_docs_mean_change_positive": (d.mean_paired_change > 0).mean(),
                     "frac_docs_mean_change_negative": (d.mean_paired_change < 0).mean()})
        group_json[g] = {"instance_level": row, "document_level": dsum[-1], "low_high": lh[-1]}

    pd.DataFrame(ph).to_csv(stage / "per_head_paired.csv", index=False)
    pd.DataFrame(ls).to_csv(stage / "layer_summary.csv", index=False)
    pd.DataFrame(cs).to_csv(stage / "paired_count_summary.csv", index=False)
    pd.DataFrame(dist).to_csv(stage / "count_distribution.csv", index=False)
    pd.DataFrame(lh).to_csv(stage / "low_high_summary.csv", index=False)
    pd.concat(pa, ignore_index=True).to_csv(stage / "per_attack_paired.csv", index=False)
    pd.DataFrame(dsum).to_csv(stage / "document_level_summary.csv", index=False)
    summary = {
        "created": now(), "scope": P.SCOPE, "layers": P.LAYERS, "k_abnormal": K, "n_heads": len(labels), "heads": labels, "n_folds": P.N_FOLDS,
        "fold_seed": P.FOLD_SEED, "n_aligned_instances": int(len(df)), "n_successful_instances": int(len(s)),
        "success_by_group": {g: {"n_instances": int((df.relevance_group == g).sum()),
                                 "n_successful": int((df.successful & (df.relevance_group == g)).sum()),
                                 "success_rate": float(df[df.relevance_group == g].successful.mean())}
                             for g in P.GROUPS if (df.relevance_group == g).any()},
        "reference": {"definition": "controls of successful qrel 2/3 instances of training queries, all attacks pooled",
                      "n_rows_per_fold": rf.groupby("fold").n_reference_rows.first().tolist(),
                      "n_docs_per_fold": rf.groupby("fold").n_reference_docs.first().tolist(),
                      "n_attack_configs_per_fold": rf.groupby("fold").n_reference_attack_configs.first().tolist(),
                      "n_invalid_heads_max": int((~rf.valid).groupby(rf.fold).sum().max())},
        "groups": group_json,
    }
    (stage / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_status(stage, {"status": "success", "finished": now()})
    print(pd.DataFrame(cs)[["relevance_group", "n_successful_instances", "mean_control_abnormal_count",
                            "mean_attack_abnormal_count", "mean_paired_change", "median_paired_change",
                            "frac_change_positive", "frac_change_negative"]].to_string(index=False))
    print(pd.DataFrame(dsum).to_string(index=False))
    print(f"[19] -> {stage}")


if __name__ == "__main__":
    main()
