"""
scripts/02_analyze.py
======================
Stage 02 — summaries, plots and compact tables from stage 01 (no model; CPU; seconds).

A. Genuine relevance: CLEAN sequences, qrel 2/3 vs qrel 0 (qrel 1 absent from the sample),
   metric x source region (query, doc) x head, full_visible scope: means, medians, difference,
   Cohen d (+ = larger for relevant), raw AUROC P(rel > nonrel), direction-free separation, and a
   length check (Cohen d after regressing the metric on the number of visible keys; Spearman with it).
B. Successful attacks (stage-22 definition: delta_score > 0) vs their OWN padded control,
   relevant / non-relevant base documents separately, key scopes full_visible and shared_key:
   delta = metric(attack, region, scope) - metric(control, region', full_visible)
   (the control's shared_key == full_visible by construction — validated in stage 01), for
   query->query, orig->doc, doc->doc. Instance-level + equal-document-weight summaries, paired
   dz, Spearman (primary) / Pearson with delta_score, and the same delta on UNSUCCESSFUL attacks
   (generic-insertion reference).
E. Injected tokens (attacked sequences only; no invented control delta): region-to-region mass
   (raw and token-count-normalised) for query<->inserted and original<->inserted (+ query<->original
   as a same-sequence reference), and the entropy / concentration of the inserted source rows vs
   the original-document rows of the same attacked sequence. Secondary: by attack token, position,
   repetitions.
C. Optional: Spearman across the 144 heads between attention-statistic effects and Exp 11's
   per-head causal combined effect (canonical relevant_start_5 run; sweep mean over the 12
   stage-22 attacks and over all 105). Read-only; skipped if missing.

Outputs (outputs/02_analysis/): relevance_attention_summary.csv, paired_attack_attention_summary.csv,
  inserted_attention_summary.csv, inserted_source_shape_summary.csv, inserted_by_factor.csv,
  causal_alignment.csv, summary_table.{csv,md}, injected_table.{csv,md}, summary.json, plots/*.png
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp17lib  # noqa: E402

from exp16lib import paired as P  # noqa: E402
from exp16lib.activation_stats import cohen_d, paired_dz  # noqa: E402
from exp16lib.anomaly import auroc  # noqa: E402
from exp16lib.heads import encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, now, write_status  # noqa: E402
from exp17lib import attention_stats as AT  # noqa: E402
from exp17lib.config import load_config, output_dir, resolve  # noqa: E402

_spec = importlib.util.spec_from_file_location("p21", exp17lib.EXP16_DIR / "scripts" / "21_plot_paired.py")
P21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P21)
P5 = P21.P5

STAGE, SRC = "02_analysis", "01_attention_stats"
REL, NON = "relevant", "nonrelevant"
LABELS = [f"L{L}H{h}" for L in range(12) for h in range(12)]
LAYER = np.repeat(np.arange(12), 12)
MAIN = ["entropy_norm", "max_attn", "top3", "neff_norm", "dist_norm"]
PAIRED_MAIN = ["entropy_norm", "top3", "dist_norm"]
NAMES = {"entropy_norm": "normalised attention entropy", "entropy": "attention entropy (nats)", "max_attn": "max attention (top-1)",
         "top3": "top-3 attention mass", "top5": "top-5 attention mass", "neff": "effective # attended keys",
         "neff_norm": "normalised effective support exp(H)/N", "dist": "mean attention distance (tokens)",
         "dist_norm": "normalised attention distance", "local_mass": "local mass |i-j|<=5"}
PAIRED_REGIONS = [("query", "query"), ("orig", "doc"), ("doc", "doc")]          # (attack region, control region)
RI = {r: i for i, r in enumerate(AT.REGIONS)}
SI = {s: i for i, s in enumerate(AT.SCOPES)}
MI = {m: i for i, m in enumerate(AT.METRICS)}
DI = {d: i for i, d in enumerate(AT.MASS_NAMES)}
FULL, SHARED = "full_visible", "shared_key"


def _cols(X, y, fn):
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j]) & ~np.isnan(y)
        if ok.sum() > 2 and np.std(X[ok, j]) > 0:
            out[j] = fn(X[ok, j], y[ok])[0]
    return out


def _heat(ax, v, title, vmax, canon, cmap="RdBu_r", center=0.0):
    im = ax.imshow(np.asarray(v, float).reshape(12, 12), cmap=cmap, vmin=center - vmax, vmax=center + vmax)
    for lab in canon:
        L, h = int(lab[1:lab.index("H")]), int(lab[lab.index("H") + 1:])
        ax.add_patch(Rectangle((h - 0.5, L - 0.5), 1, 1, fill=False, ec=P5.INK, lw=1.2))
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(12)); ax.set_yticks(range(12)); ax.grid(False)
    return im


def _vmax(*arrs, pct=100):
    v = np.nanpercentile(np.abs(np.concatenate([np.ravel(a) for a in arrs])), pct)
    return float(v) if np.isfinite(v) and v > 0 else 1.0


def _box(ax, vals, names, colors, title):
    bp = ax.boxplot(vals, widths=0.55, patch_artist=True, flierprops={"markersize": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=8, loc="left")


def _md(df):
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + \
           "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def causal_references(cfg, attacks):
    """Exp 11 per-head causal combined effects (144 heads, layer-major) or {} if unavailable."""
    d = resolve(cfg, cfg["exp11_outputs_dir"])
    out = {}
    f = d / "head_summary_canonical.csv"
    if f.exists():
        c = pd.read_csv(f).sort_values(["layer", "head_idx"])
        if len(c) == 144:
            out["exp11_canonical_relevant_start_5"] = c.combined_effect_mean.values
    f = d / "head_attack_summary_sweep.csv"
    if f.exists():
        s = pd.read_csv(f)
        for name, sub in (("exp11_sweep_mean_12_stage22_attacks", s[s.attack_name.isin(attacks)]), ("exp11_sweep_mean_105_attacks", s)):
            g = sub.groupby(["layer", "head_idx"]).combined_effect_mean.mean().sort_index()
            if len(g) == 144 and (name.endswith("105_attacks") or sub.attack_name.nunique() == len(set(attacks))):
                out[name] = g.values
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Exp 17 stage 02: analysis + plots")
    ap.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P.configure("all_layers")                         # head_axis: layers 0-11
    out = output_dir(cfg)
    src = out / SRC
    if not is_already_successful(src, ["attention_statistics.npz"]):
        raise FileNotFoundError(f"run stage 01 first ({src})")
    st = out / STAGE
    if is_already_successful(st, ["summary.json"]) and not args.force:
        print(f"[02] {st} already complete (use --force)")
        return
    pdir = st / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    write_status(st, {"status": "running", "started": now()})
    t0 = time.time()
    z = np.load(src / "attention_statistics.npz")
    R, M = z["R"].astype(np.float64), z["M"].astype(np.float64)
    meta = pd.read_csv(src / "sequences_meta.csv", dtype={"qid": str, "docid": str, "pair_id": str, "attack_name": str,
                                                          "attack_token": str, "attack_position": str}, low_memory=False)
    meta["successful"] = meta.successful.map({True: True, False: False, "True": True, "False": False})
    assert (meta.seq_id.values == z["seq_id"]).all() and list(z["metrics"]) == AT.METRICS and list(z["regions"]) == AT.REGIONS
    canon = [h.label for h in encoder_heads()]
    canon_set = set(canon)
    X = np.arange(144)

    # ---- A. genuine relevance --------------------------------------------------------------------------------
    cl = np.where(meta.kind.values == "clean")[0]
    cm = meta.iloc[cl]
    isrel = (cm.relevance_group == REL).values
    nkeys = cm.n_keys_full_visible.values.astype(float)
    rowsA = []
    for r in ("query", "doc"):
        for m in AT.METRICS:
            V = R[cl, RI[r], SI[FULL], MI[m]]
            rho_len = _cols(V, nkeys, spearmanr)
            Vres = np.column_stack([AT.residualize(V[:, j], nkeys) for j in range(144)])
            for j, lab in enumerate(LABELS):
                a, b = V[isrel, j], V[~isrel, j]
                au = auroc(a, b)
                rowsA.append({"metric": m, "source_region": r, "key_scope": FULL, "head_label": lab, "layer": LAYER[j],
                              "head": j % 12, "canonical_head": lab in canon_set, "n_rel": len(a), "n_nonrel": len(b),
                              "mean_rel": a.mean(), "mean_nonrel": b.mean(), "median_rel": np.median(a),
                              "median_nonrel": np.median(b), "difference": a.mean() - b.mean(), "cohen_d": cohen_d(a, b),
                              "auroc_raw": au, "auroc_separation": max(au, 1 - au),
                              "cohen_d_length_residualized": cohen_d(Vres[isrel, j], Vres[~isrel, j]),
                              "spearman_vs_n_visible_keys": rho_len[j]})
    A = pd.DataFrame(rowsA)
    A["abs_d"] = A.cohen_d.abs()
    A.to_csv(st / "relevance_attention_summary.csv", index=False)
    length = {"n_visible_keys_mean_rel": float(nkeys[isrel].mean()), "n_visible_keys_mean_nonrel": float(nkeys[~isrel].mean()),
              "n_visible_keys_cohen_d": cohen_d(nkeys[isrel], nkeys[~isrel]),
              "n_query_tokens_cohen_d": cohen_d(cm.n_tokens_query.values[isrel].astype(float), cm.n_tokens_query.values[~isrel].astype(float))}
    print(f"[02] A done ({time.time() - t0:.0f}s)", flush=True)

    # ---- B. paired control -> attack ----------------------------------------------------------------------------
    atk = meta[meta.kind == "attack"]
    ctl_idx = meta[meta.kind == "control"].reset_index().set_index(["attack_name", "pair_id"])["index"]
    ia_all = atk.index.values
    ic_all = ctl_idx.loc[list(zip(atk.attack_name, atk.pair_id))].values
    assert (meta.pair_id.values[ic_all] == atk.pair_id.values).all() and (meta.kind.values[ic_all] == "control").all()
    succ_all = atk.successful.values.astype(bool)
    grp_all = atk.relevance_group.values
    dsc_all = atk.delta_score.values
    rowsB = []
    for sc in (SHARED, FULL):
        for ra, rc in PAIRED_REGIONS:
            for m in AT.METRICS:
                Dall = R[ia_all, RI[ra], SI[sc], MI[m]] - R[ic_all, RI[rc], SI[FULL], MI[m]]
                for g in (REL, NON):
                    k = (grp_all == g) & succ_all
                    ku = (grp_all == g) & ~succ_all
                    D, pid, dsc = Dall[k], atk.pair_id.values[k], dsc_all[k]
                    docm = pd.DataFrame(D).groupby(pid).mean()
                    docs = pd.Series(dsc).groupby(pid).mean().loc[docm.index].values
                    rho, pr, rho_doc = _cols(D, dsc, spearmanr), _cols(D, dsc, pearsonr), _cols(docm.values, docs, spearmanr)
                    cmean = R[ic_all[k], RI[rc], SI[FULL], MI[m]].mean(0)
                    amean = R[ia_all[k], RI[ra], SI[sc], MI[m]].mean(0)
                    for j, lab in enumerate(LABELS):
                        x, xd, xu = D[:, j], docm.values[:, j], Dall[ku, j]
                        rowsB.append({"metric": m, "key_scope": sc, "source_region": ra, "control_region": rc,
                                      "relevance_group": g, "head_label": lab, "layer": LAYER[j], "head": j % 12,
                                      "canonical_head": lab in canon_set, "n_instances": len(x), "n_docs": len(xd),
                                      "control_mean": cmean[j], "attack_mean": amean[j], "mean_delta": x.mean(),
                                      "median_delta": np.median(x), "frac_delta_positive": (x > 0).mean(),
                                      "paired_dz": paired_dz(x), "spearman_delta_vs_delta_score": rho[j],
                                      "pearson_delta_vs_delta_score": pr[j], "mean_delta_doc_weighted": xd.mean(),
                                      "paired_dz_doc_weighted": paired_dz(xd), "frac_docs_delta_positive": (xd > 0).mean(),
                                      "spearman_doc_level": rho_doc[j], "n_unsuccessful": len(xu),
                                      "mean_delta_unsuccessful": xu.mean() if len(xu) else np.nan,
                                      "paired_dz_unsuccessful": paired_dz(xu)})
    B = pd.DataFrame(rowsB)
    B["abs_dz"] = B.paired_dz.abs()
    B.to_csv(st / "paired_attack_attention_summary.csv", index=False)
    print(f"[02] B done ({time.time() - t0:.0f}s)", flush=True)

    # ---- E. injected tokens ----------------------------------------------------------------------------------------
    rowsE, rowsS = [], []
    for g in (REL, NON):
        k = (grp_all == g) & succ_all
        ku = (grp_all == g) & ~succ_all
        pid = atk.pair_id.values[k]
        for dname in AT.MASS_NAMES:
            raw, nrm = M[ia_all[k], DI[dname], 0], M[ia_all[k], DI[dname], 1]
            nrm_u = M[ia_all[ku], DI[dname], 1]
            docn = pd.DataFrame(nrm).groupby(pid).mean().values
            for j, lab in enumerate(LABELS):
                rowsE.append({"direction": dname, "relevance_group": g, "head_label": lab, "layer": LAYER[j], "head": j % 12,
                              "canonical_head": lab in canon_set, "n_instances": int(k.sum()), "n_docs": len(docn),
                              "mean_raw_mass": raw[:, j].mean(), "mean_norm_mass": nrm[:, j].mean(),
                              "median_norm_mass": np.median(nrm[:, j]), "frac_norm_gt_1": (nrm[:, j] > 1).mean(),
                              "mean_norm_mass_doc_weighted": docn[:, j].mean(),
                              "spearman_norm_vs_delta_score": spearmanr(nrm[:, j], dsc_all[k])[0] if np.std(nrm[:, j]) > 0 else np.nan,
                              "mean_norm_mass_unsuccessful": nrm_u[:, j].mean() if len(nrm_u) else np.nan})
        for sc in (FULL, SHARED):
            for m in AT.METRICS:
                ins = R[ia_all[k], RI["ins"], SI[sc], MI[m]]
                org = R[ia_all[k], RI["orig"], SI[sc], MI[m]]
                d_ = ins - org
                for j, lab in enumerate(LABELS):
                    rowsS.append({"metric": m, "key_scope": sc, "relevance_group": g, "head_label": lab, "layer": LAYER[j],
                                  "head": j % 12, "canonical_head": lab in canon_set, "n_instances": int(k.sum()),
                                  "mean_ins_rows": ins[:, j].mean(), "mean_orig_rows": org[:, j].mean(),
                                  "mean_ins_minus_orig": d_[:, j].mean(), "frac_ins_gt_orig": (d_[:, j] > 0).mean(),
                                  "dz_ins_vs_orig": paired_dz(d_[:, j])})
    E = pd.DataFrame(rowsE)
    E.to_csv(st / "inserted_attention_summary.csv", index=False)
    ES = pd.DataFrame(rowsS)
    ES.to_csv(st / "inserted_source_shape_summary.csv", index=False)
    rowsF = []                                          # secondary: attack token / position / repetitions
    for g in (REL, NON):
        k = (grp_all == g) & succ_all
        sub = atk[k]
        for factor in ("attack_token", "attack_position", "repetitions"):
            for lev, idx in sub.groupby(factor).groups.items():
                pos = np.searchsorted(ia_all, idx.values)
                for dname in AT.MASS_NAMES[:4]:
                    hm = M[ia_all[pos], DI[dname], 1].mean(0)
                    rowsF.append({"relevance_group": g, "factor": factor, "level": lev, "direction": dname,
                                  "n_instances": len(idx), "n_docs": sub.loc[idx].pair_id.nunique(),
                                  "median_over_heads_mean_norm_mass": float(np.median(hm)),
                                  "max_over_heads_mean_norm_mass": float(hm.max()), "argmax_head": LABELS[int(hm.argmax())]})
    F = pd.DataFrame(rowsF)
    F.to_csv(st / "inserted_by_factor.csv", index=False)
    print(f"[02] E done ({time.time() - t0:.0f}s)", flush=True)

    # ---- C. causal alignment (optional) ----------------------------------------------------------------------------
    stage22_attacks = sorted(atk.attack_name.unique())
    causal = causal_references(cfg, stage22_attacks)
    rowsC = []
    if causal:
        feats = {}
        for sc in (SHARED, FULL):
            for m in PAIRED_MAIN:
                for ra in ("query", "orig"):
                    t = B[(B.metric == m) & (B.key_scope == sc) & (B.source_region == ra)]
                    feats[f"|mean delta| {m} {ra} {sc}"] = t.groupby("head_label").mean_delta.apply(lambda s: s.abs().mean()).loc[LABELS].values
                    feats[f"|paired dz| {m} {ra} {sc}"] = t.groupby("head_label").abs_dz.mean().loc[LABELS].values
        for dname in AT.MASS_NAMES[:4]:
            t = E[E.direction == dname].groupby("head_label")
            feats[f"mean norm mass {dname}"] = t.mean_norm_mass.mean().loc[LABELS].values
            feats[f"mean RAW mass {dname} (magnitude reference)"] = t.mean_raw_mass.mean().loc[LABELS].values
        for m in MAIN:
            t = A[(A.metric == m) & (A.source_region == "doc")].set_index("head_label").loc[LABELS]
            feats[f"|Cohen d| relevance {m} doc"] = t.abs_d.values
        for cname, cv in causal.items():
            for fname, fv in feats.items():
                rowsC.append({"causal_reference": cname, "attention_feature": fname,
                              "spearman_vs_causal_effect": spearmanr(fv, cv)[0],
                              "spearman_vs_abs_causal_effect": spearmanr(fv, np.abs(cv))[0],
                              "top18_overlap_with_canonical": len(set(np.array(LABELS)[np.argsort(-fv)[:18]]) & canon_set)})
    C = pd.DataFrame(rowsC)
    C.to_csv(st / "causal_alignment.csv", index=False)

    # ---- plots ------------------------------------------------------------------------------------------------------
    def a_vec(m, r, col):
        return A[(A.metric == m) & (A.source_region == r)].set_index("head_label").loc[LABELS][col].values

    fig, axes = plt.subplots(2, len(MAIN), figsize=(4.3 * len(MAIN), 9.2))
    for c_, m in enumerate(MAIN):
        dq, dd = a_vec(m, "query", "cohen_d"), a_vec(m, "doc", "cohen_d")
        v = _vmax(dq, dd)
        _heat(axes[0, c_], dq, f"{m}: query rows", v, canon)
        im = _heat(axes[1, c_], dd, f"{m}: document rows", v, canon)
        fig.colorbar(im, ax=axes[:, c_], shrink=0.5, location="bottom", label="Cohen d")
    fig.suptitle("Clean inputs, full-visible keys: Cohen d relevant (qrel 2/3) − non-relevant (qrel 0); red = larger for relevant; "
                 "boxed = canonical Exp 13 encoder heads", x=0.01, ha="left", fontsize=10)
    P5._save(fig, pdir / "A_relevance_cohen_d_heatmaps.png")

    for m in MAIN:
        fig, axes = plt.subplots(2, 1, figsize=(28, 8.4))
        for ax, r in zip(axes, ("query", "doc")):
            diff = a_vec(m, r, "difference")
            ax.bar(X, diff, color=[P5.BLUE if v_ > 0 else P5.ORANGE for v_ in diff])
            ax.axhline(0, color=P5.INK, lw=1)
            P21.head_axis(ax, LABELS, canon_set)
            ax.set_ylabel("relevant − non-relevant")
            ax.set_title(f"{r} source rows: {NAMES[m]}, difference of means (blue = larger for relevant; bold = canonical head)",
                         loc="left", fontsize=10, pad=16)
        fig.tight_layout(); P5._save(fig, pdir / f"A_relevance_heads_{m}.png")

    top = A[A.metric.isin(MAIN)].sort_values("abs_d", ascending=False).head(5)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, (_, row) in zip(axes, top.iterrows()):
        j = LABELS.index(row.head_label)
        V = R[cl, RI[row.source_region], SI[FULL], MI[row.metric], j]
        _box(ax, [V[isrel], V[~isrel]], ["relevant", "non-rel."], (P5.BLUE, P5.ORANGE),
             f"{row.metric} {row.source_region} {row.head_label}\nd={row.cohen_d:+.2f}, AUROC={row.auroc_raw:.2f}")
    fig.suptitle("Top-5 (metric, region, head) cells by |Cohen d| — clean relevant vs non-relevant", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); P5._save(fig, pdir / "A_top5_distributions.png")

    for sc in (SHARED, FULL):
        for m in PAIRED_MAIN:
            fig, axes = plt.subplots(2, 4, figsize=(19, 9.4))
            mats = {(g, ra): B[(B.metric == m) & (B.key_scope == sc) & (B.source_region == ra) & (B.relevance_group == g)]
                    .set_index("head_label").loc[LABELS] for g in (REL, NON) for ra in ("query", "orig")}
            vd = _vmax(*[t.mean_delta.values for t in mats.values()], pct=98)
            vr = _vmax(*[t.spearman_delta_vs_delta_score.values for t in mats.values()])
            for r_, g in enumerate((REL, NON)):
                for c_, ra in enumerate(("query", "orig")):
                    t = mats[(g, ra)]
                    imd = _heat(axes[r_, c_], t.mean_delta.values, f"{g}: mean Δ, {ra} rows (n={int(t.n_instances.iloc[0])})", vd, canon)
                    imr = _heat(axes[r_, 2 + c_], t.spearman_delta_vs_delta_score.values, f"{g}: ρ(Δ, Δscore), {ra} rows", vr, canon, cmap="PuOr_r")
            fig.colorbar(imd, ax=axes[:, :2], shrink=0.6, extend="both", label="mean(attack − own control), clipped at 98th pct")
            fig.colorbar(imr, ax=axes[:, 2:], shrink=0.6, label="Spearman ρ")
            fig.suptitle(f"{NAMES[m]}: successful attacks vs own padded control — {sc.upper()} key scope"
                         f"{' (injected tokens excluded as keys)' if sc == SHARED else ' (secondary diagnostic)'}; "
                         "orig rows compared with control document rows", x=0.01, ha="left", fontsize=10)
            P5._save(fig, pdir / f"B_paired_{m}_{sc}.png")

    topb = B[B.metric.isin(MAIN) & (B.key_scope == SHARED) & B.source_region.isin(["query", "orig"])]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4))
    for r_, g in enumerate((REL, NON)):
        k = (grp_all == g) & succ_all
        for c_, (_, row) in enumerate(topb[topb.relevance_group == g].sort_values("abs_dz", ascending=False).head(5).iterrows()):
            j = LABELS.index(row.head_label)
            vc = R[ic_all[k], RI[row.control_region], SI[FULL], MI[row.metric], j]
            va = R[ia_all[k], RI[row.source_region], SI[SHARED], MI[row.metric], j]
            _box(axes[r_, c_], [vc, va], ["control", "attack"], (P5.BLUE, P5.ORANGE),
                 f"{g[:6]}. {row.metric} {row.source_region} {row.head_label}\nΔ={row.mean_delta:+.3g}, dz={row.paired_dz:+.2f}, "
                 f"ρ={row.spearman_delta_vs_delta_score:+.2f}")
    fig.suptitle("Top-5 cells by |paired dz|, shared-key scope, successful attacks", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); P5._save(fig, pdir / "B_top5_distributions.png")

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.4))
    mats = {(g, d): E[(E.direction == d) & (E.relevance_group == g)].set_index("head_label").loc[LABELS]
            for g in (REL, NON) for d in AT.MASS_NAMES[:4]}
    vl = 3.0                                            # clip: directional heads (same head index in every layer, one shared
                                                        # relative-position bias) reach log2 ~ -12 and would wash out the 1-8x range
    for r_, g in enumerate((REL, NON)):
        for c_, d in enumerate(AT.MASS_NAMES[:4]):
            t = mats[(g, d)]
            im = _heat(axes[r_, c_], np.log2(t.mean_norm_mass.values),
                       f"{g}: {d.replace('_to_', ' → ')}\n(max {t.mean_norm_mass.max():.1f}× at {t.mean_norm_mass.idxmax()})", vl, canon)
    fig.colorbar(im, ax=axes, shrink=0.6, extend="both", label="log2(mean normalised mass), clipped at ±3; 0 = token-count-uniform baseline")
    fig.suptitle("Injected-token attention, successful attacks (full-visible keys): observed mass / (|target| / N visible)",
                 x=0.01, ha="left", fontsize=10)
    P5._save(fig, pdir / "D_injected_mass_heatmaps.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True)
    for ax, (a_, b_) in zip(axes, (("query_to_ins", "query_to_orig"), ("ins_to_query", "orig_to_query"), ("orig_to_ins", "ins_to_orig"))):
        for g, colr in ((REL, P5.BLUE), (NON, P5.ORANGE)):
            for d, ls in ((a_, "-"), (b_, "--")):
                lv = E[(E.direction == d) & (E.relevance_group == g)].groupby("layer").mean_norm_mass.median()
                ax.plot(lv.index, lv.values, color=colr, ls=ls, marker="o", ms=3, label=f"{g}: {d}")
        ax.axhline(1, color=P5.INK2, lw=1)
        ax.set_yscale("log"); ax.set_xticks(range(12)); ax.set_xlabel("layer"); ax.legend(fontsize=7)
    axes[0].set_ylabel("median over heads of mean normalised mass")
    fig.suptitle("Injected-token attention by layer (successful attacks); dashed = original-document reference", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); P5._save(fig, pdir / "D_injected_mass_by_layer.png")

    if causal:
        cref = "exp11_sweep_mean_12_stage22_attacks" if "exp11_sweep_mean_12_stage22_attacks" in causal else next(iter(causal))
        cv = causal[cref]
        shown = [f"|paired dz| entropy_norm query {SHARED}", f"|paired dz| top3 orig {SHARED}",
                 "mean norm mass query_to_ins", "mean RAW mass query_to_ins (magnitude reference)"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
        for ax, fname in zip(axes, shown):
            fv = feats[fname]
            cc = np.array([P5.ORANGE if lab in canon_set else P5.INK2 for lab in LABELS])
            ax.scatter(fv, cv, s=12, c=cc)
            ax.set_xlabel(fname, fontsize=8); ax.set_ylabel(f"causal combined effect ({cref})", fontsize=7)
            ax.set_title(f"ρ = {spearmanr(fv, cv)[0]:+.2f} (144 heads; orange = canonical)", fontsize=8, loc="left")
        fig.tight_layout(); P5._save(fig, pdir / "C_causal_alignment.png")
    print(f"[02] plots done ({time.time() - t0:.0f}s)", flush=True)

    # ---- compact tables ------------------------------------------------------------------------------------------
    rows = []
    for m in MAIN + ["local_mass"]:
        a = A[A.metric == m].sort_values("abs_d", ascending=False).iloc[0]
        bs = B[(B.metric == m) & (B.key_scope == SHARED) & B.source_region.isin(["query", "orig"])].sort_values("abs_dz", ascending=False).iloc[0]
        bf = B[(B.metric == m) & (B.key_scope == FULL) & (B.source_region == bs.source_region) & (B.relevance_group == bs.relevance_group)
               & (B.head_label == bs.head_label)].iloc[0]
        rows.append({"metric": m, "strongest rel/nonrel head": f"{a.head_label} ({a.source_region})", "Cohen d": f"{a.cohen_d:+.2f}",
                     "AUROC": f"{a.auroc_raw:.2f}",
                     "strongest paired attack head (shared-key)": f"{bs.head_label} ({bs.source_region}, {bs.relevance_group})",
                     "mean delta [dz]": f"{bs.mean_delta:+.3g} [{bs.paired_dz:+.2f}]",
                     "corr(delta, dscore)": f"{bs.spearman_delta_vs_delta_score:+.2f}",
                     "note": f"len-resid d {a.cohen_d_length_residualized:+.2f}; full-visible dz {bf.paired_dz:+.2f}; "
                             f"unsuccessful dz {bs.paired_dz_unsuccessful:+.2f}"})
    CT = pd.DataFrame(rows)
    CT.to_csv(st / "summary_table.csv", index=False)
    (st / "summary_table.md").write_text(_md(CT))
    rows = []
    for d in AT.MASS_NAMES:
        for g in (REL, NON):
            t = E[(E.direction == d) & (E.relevance_group == g)]
            b = t.sort_values("mean_norm_mass", ascending=False).iloc[0]
            rows.append({"direction": d.replace("_to_", " → "), "strongest head": b.head_label,
                         "normalized mass": f"{b.mean_norm_mass:.2f}", "relevance group": g,
                         "note": f"median over heads {t.mean_norm_mass.median():.2f}; heads >1: {(t.mean_norm_mass > 1).sum()}/144; "
                                 f"frac inst >1 at top head {b.frac_norm_gt_1:.2f}; unsuccessful {b.mean_norm_mass_unsuccessful:.2f}; "
                                 f"ρ(norm, Δscore) {b.spearman_norm_vs_delta_score:+.2f}"})
    IT = pd.DataFrame(rows)
    IT.to_csv(st / "injected_table.csv", index=False)
    (st / "injected_table.md").write_text(_md(IT))
    summary = {"created": now(), "n_sequences": len(meta), "n_clean": int(len(cl)), "n_rel_clean": int(isrel.sum()),
               "n_successful": int(succ_all.sum()),
               "n_successful_by_group": {g: int(((grp_all == g) & succ_all).sum()) for g in (REL, NON)},
               "n_successful_docs_by_group": {g: int(atk[(grp_all == g) & succ_all].pair_id.nunique()) for g in (REL, NON)},
               "length": length, "causal_references": sorted(causal), "canonical_heads": canon,
               "seconds": time.time() - t0}
    (st / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_status(st, {"status": "success", "finished": now()})
    print(CT.to_string(index=False))
    print(IT.to_string(index=False))
    print(f"[02] -> {st} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
