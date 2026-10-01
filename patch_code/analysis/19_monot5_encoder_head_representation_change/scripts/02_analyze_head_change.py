"""
scripts/02_analyze_head_change.py
==================================
Exp 19 stage 02 — analysis + figures of stage 01 (CPU, no model).

Inputs : 01_head_change/{instances.parquet, metrics/*.npy, cka.csv}
Causal heads (read-only sources of truth):
  binary      exp13lib.head_lists.load_senders()  18 encoder heads (Exp 11 canonical combined effect > 0.02)
  continuous  Exp 11 outputs/head_summary_canonical.csv combined_effect_mean (all 144 heads)

Per-head change values (main view = successful attacks, all base documents):
  omc / nl2 / tw_omc / tw_nl2   instance-weighted means, 95% query-bootstrap CI (run.n_boot, run.boot_seed)
  1 - CKA raw / query-centred   population level (stage 01)
Correlation with delta_score: Spearman within attack configuration (ranks within each attack, then Pearson),
  pooled Spearman (descriptive), and per-query Spearman (median, n positive).
Causal comparison per metric x region: causal vs other mean/median; global ranks; top-k enrichment
  (k = 5, 10, 18; hypergeometric tail, descriptive); Spearman across 144 heads with the Exp 11 effect;
  WITHIN-LAYER percentile of causal heads vs a layer-stratified permutation null (controls for layer
  magnitude differences), with a query-bootstrap CI for the pooled-metric version.
Layer-8 question: per-layer distribution / concentration of per-head document 1 - CKA at L6-L11.
Success contrast: per head, successful - unsuccessful mean change computed WITHIN each attack configuration and
  averaged over configurations (removes configuration composition), in units of the pooled SD.

Outputs: 02_analysis/{head_summary.csv, head_correlations.csv, cka.csv, causal_comparison.csv, causal_head_ranks.csv,
         topk_heads.csv, layer_concentration.csv, success_contrast.csv, relevance_contrast.csv, summary.json}
         02_analysis/plots/fig{1..7}_*.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp19lib  # noqa: E402,F401

from exp13lib.head_lists import load_senders  # noqa: E402
from exp16lib.run_utils import is_already_successful, now, write_status  # noqa: E402
from exp18lib import repr_change as R  # noqa: E402
from exp19lib import head_change as HC  # noqa: E402
from exp19lib.config import load_config, output_dir, resolve  # noqa: E402

NH = HC.N_LAYERS * HC.N_HEADS
LAYER_OF = np.repeat(np.arange(HC.N_LAYERS), HC.N_HEADS)
HEAD_OF = np.tile(np.arange(HC.N_HEADS), HC.N_LAYERS)
MLAB = {"one_minus_cosine": "1 − cos (mean-pooled)", "normalized_l2": "normalized L2 (mean-pooled)",
        "tw_mean_omc": "token-wise mean 1 − cos", "tw_mean_nl2": "token-wise mean normalized L2",
        "one_minus_cka_raw": "1 − linear CKA (raw)", "one_minus_cka_qc": "1 − linear CKA (query-centred)"}
RLAB = {"query": "query", "orig_doc": "original document"}
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
GROUPS = [("successful", "all"), ("unsuccessful", "all"), ("all", "all"),
          ("successful", "relevant"), ("successful", "nonrelevant")]


def load(stage01: pathlib.Path):
    inst = pd.read_parquet(stage01 / "instances.parquet")
    for c in ("qid", "docid", "attack_name", "relevance_group", "pair_id"):
        inst[c] = inst[c].astype(str)
    parts = []
    for a in pd.unique(inst.attack_name):
        m = np.load(stage01 / "metrics" / f"{a}.npy")                            # [n, 12, 12, 2, 4]
        if len(m) != (inst.attack_name == a).sum():
            raise RuntimeError(f"{a}: metrics rows != instances")
        parts.append(m.reshape(len(m), NH, HC.N_REG, len(HC.METRICS)))
    M = np.concatenate(parts)
    order_ok = (inst.groupby("attack_name", sort=False).row_in_attack.apply(lambda s: (s.values == np.arange(len(s))).all())).all()
    if not order_ok:
        raise RuntimeError("instances.parquet not in per-attack row order")
    return inst, M


def causal_sets(cfg):
    senders = load_senders()
    is_c = np.zeros(NH, bool)
    for s in senders:
        is_c[s.layer * HC.N_HEADS + s.head_idx] = True
    eff = pd.read_csv(resolve(cfg, cfg["exp11_head_summary"]))
    eff = eff.set_index(["layer", "head_idx"]).combined_effect_mean
    e = np.array([eff.loc[(l, h)] for l, h in zip(LAYER_OF, HEAD_OF)])
    if not np.array_equal(is_c, e > 0.02):
        raise RuntimeError("exp13 sender list != Exp 11 canonical combined_effect > 0.02")
    return is_c, e


def head_summary(inst, M, qids, n_boot, seed):
    q = pd.Categorical(inst.qid, categories=qids).codes
    Q = len(qids)
    rows, boots = [], {}
    for sg, rg in GROUPS:
        m = R.group_mask(inst.successful.values, inst.relevance_group.values, sg, rg)
        cnt = np.bincount(q[m], minlength=Q).astype(float)
        X = M[m].reshape(m.sum(), -1).astype(np.float64)                          # [n, 144*2*4]
        sums = np.stack([np.bincount(q[m], weights=X[:, j], minlength=Q) for j in range(X.shape[1])], 1)
        mean, lo, hi = R.query_bootstrap_mean(sums, cnt, n_boot, seed)
        med = np.median(X, axis=0)
        if (sg, rg) == ("successful", "all"):                                    # keep replicates for causal CI
            rng = np.random.default_rng(seed)
            keep = cnt > 0
            W = rng.multinomial(keep.sum(), np.full(keep.sum(), 1 / keep.sum()), size=n_boot).astype(float)
            boots = (W @ sums[keep]) / (W @ cnt[keep])[:, None]                   # [n_boot, 1152]
        k = 0
        for hi_ in range(NH):
            for r in range(HC.N_REG):
                for mi, met in enumerate(HC.METRICS):
                    j = (hi_ * HC.N_REG + r) * len(HC.METRICS) + mi
                    rows.append({"layer": LAYER_OF[hi_], "head": HEAD_OF[hi_], "head_label": HC.HEAD_LABELS[hi_],
                                 "region": HC.REGIONS[r], "metric": met, "success_group": sg, "relevance_group": rg,
                                 "mean": mean[j], "ci_lo": lo[j], "ci_hi": hi[j], "median": med[j],
                                 "n_instances": int(m.sum()), "n_queries": int((cnt > 0).sum())})
                    k += 1
    return pd.DataFrame(rows), boots


def _rank_within(V, groups):
    out = np.empty(V.shape, np.float32)
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        out[i] = stats.rankdata(V[i], axis=0)
    return out


def _col_corr(A, b):
    A = A - A.mean(0)
    b = b - b.mean()
    return (A * b[:, None]).sum(0) / np.sqrt((A ** 2).sum(0) * (b ** 2).sum())


def correlations(inst, M):
    rows = []
    mets = ["one_minus_cosine", "normalized_l2"]
    mi = [HC.METRICS.index(x) for x in mets]
    for sg, rg in GROUPS:
        m = R.group_mask(inst.successful.values, inst.relevance_group.values, sg, rg)
        V = M[m][..., mi].reshape(m.sum(), -1)                                   # [n, 144*2*2]
        y = inst.delta_score.values[m]
        att, qid = inst.attack_name.values[m], inst.qid.values[m]
        rho_w = _col_corr(_rank_within(V, att), _rank_within(y[:, None], att)[:, 0])
        rho_p = _col_corr(stats.rankdata(V, axis=0).astype(np.float32), stats.rankdata(y))
        qr = []
        for qq in np.unique(qid):
            i = qid == qq
            if i.sum() >= 10:
                qr.append(_col_corr(stats.rankdata(V[i], axis=0), stats.rankdata(y[i])))
        qr = np.array(qr) if qr else np.empty((0, V.shape[1]))
        j = 0
        for hi_ in range(NH):
            for r in range(HC.N_REG):
                for met in mets:
                    col = qr[:, j][np.isfinite(qr[:, j])]
                    rows.append({"layer": LAYER_OF[hi_], "head": HEAD_OF[hi_], "head_label": HC.HEAD_LABELS[hi_],
                                 "region": HC.REGIONS[r], "metric": met, "success_group": sg, "relevance_group": rg,
                                 "spearman_within_attack": rho_w[j], "spearman_pooled": rho_p[j],
                                 "median_query_rho": float(np.median(col)) if len(col) else np.nan,
                                 "n_queries_rho_pos": int((col > 0).sum()), "n_queries_rho": int(len(col)),
                                 "n_instances": int(m.sum()), "n_queries": int(len(np.unique(qid)))})
                    j += 1
    return pd.DataFrame(rows)


def head_values(hs, cka, sg="successful", rg="all"):
    """-> {(metric, region): [144]} head-level change values (larger = more change)."""
    out = {}
    s = hs[(hs.success_group == sg) & (hs.relevance_group == rg)]
    for met in HC.METRICS:
        for r in HC.REGIONS:
            x = s[(s.metric == met) & (s.region == r)].sort_values(["layer", "head"])["mean"].values
            out[(met, r)] = x
    c = cka[(cka.success_group == sg) & (cka.relevance_group == rg)]
    for r in HC.REGIONS:
        x = c[c.region == r].sort_values(["layer", "head"])
        out[("one_minus_cka_raw", r)] = 1 - x.cka_raw.values
        out[("one_minus_cka_qc", r)] = 1 - x.cka_query_centered.values
    return out


def causal_comparison(vals, is_c, eff, boots, n_perm, seed):
    rows, ranks = [], []
    for (met, r), v in vals.items():
        pct = HC.within_layer_percentile(v, LAYER_OF)
        obs, p_perm, null = HC.layer_stratified_permutation(v, LAYER_OF, is_c, n_perm, seed)
        late = is_c & (LAYER_OF >= 9)
        obs_late, p_late, _ = HC.layer_stratified_permutation(v[LAYER_OF >= 9], LAYER_OF[LAYER_OF >= 9],
                                                              is_c[LAYER_OF >= 9], n_perm, seed)
        ci = (np.nan, np.nan)
        if met in HC.METRICS and len(boots):
            j = (np.arange(NH) * HC.N_REG + HC.REGIONS.index(r)) * len(HC.METRICS) + HC.METRICS.index(met)
            bp = np.array([HC.within_layer_percentile(b[j], LAYER_OF)[is_c].mean() for b in boots[:500]])
            ci = tuple(np.percentile(bp, [2.5, 97.5]))
        grank = stats.rankdata(-v, method="min")
        row = {"metric": met, "region": r, "mean_causal": v[is_c].mean(), "median_causal": np.median(v[is_c]),
               "mean_other": v[~is_c].mean(), "median_other": np.median(v[~is_c]),
               "ratio_median": np.median(v[is_c]) / np.median(v[~is_c]),
               "median_global_rank_causal": float(np.median(grank[is_c])),
               "spearman_vs_exp11_effect": stats.spearmanr(v, eff)[0],
               "spearman_vs_exp11_effect_L9_L11": stats.spearmanr(v[LAYER_OF >= 9], eff[LAYER_OF >= 9])[0],
               "within_layer_pct_causal": obs, "within_layer_pct_ci_lo": ci[0], "within_layer_pct_ci_hi": ci[1],
               "within_layer_perm_p": p_perm, "within_layer_null": null,
               "within_layer_pct_causal_L9_L11": obs_late, "within_layer_perm_p_L9_L11": p_late,
               "n_causal": int(is_c.sum()), "n_causal_L9_L11": int(late.sum())}
        for e in HC.topk_enrichment(v, is_c):
            row[f"top{e['k']}_n_causal"] = e["n_causal_in_topk"]
            row[f"top{e['k']}_expected"] = e["expected"]
            row[f"top{e['k']}_p"] = e["p_hypergeom_ge"]
        rows.append(row)
        for i in np.flatnonzero(is_c):
            ranks.append({"metric": met, "region": r, "head_label": HC.HEAD_LABELS[i], "layer": LAYER_OF[i],
                          "exp11_effect": eff[i], "value": v[i], "global_rank": int(grank[i]),
                          "within_layer_rank": int(12 - round(pct[i] * 11)), "within_layer_pct": pct[i]})
    return pd.DataFrame(rows), pd.DataFrame(ranks)


def layer_concentration(vals):
    rows = []
    for r in HC.REGIONS:
        for met in ("one_minus_cka_raw", "one_minus_cka_qc", "one_minus_cosine", "normalized_l2"):
            v = vals[(met, r)]
            ref = np.median(v[LAYER_OF == 6])
            for L in range(6, 12):
                x = np.sort(v[LAYER_OF == L])[::-1]
                xs = np.clip(x, 0, None)
                g = (np.abs(xs[:, None] - xs[None, :]).sum() / (2 * len(xs) ** 2 * xs.mean())) if xs.mean() > 0 else np.nan
                rows.append({"region": r, "metric": met, "layer": L, "median": np.median(x), "max": x[0],
                             "max_head": HC.HEAD_LABELS[L * 12 + int(np.argmax(v[LAYER_OF == L]))],
                             "top1_share": xs[0] / xs.sum(), "top3_share": xs[:3].sum() / xs.sum(), "gini": g,
                             "n_heads_gt_2x_L6_median": int((x > 2 * ref).sum()),
                             "median_ratio_to_L6": np.median(x) / ref})
    return pd.DataFrame(rows)


def success_contrast(inst, M):
    """Per head/region/metric: mean over attack configs of (mean successful - mean unsuccessful) / pooled SD."""
    rows = []
    att = inst.attack_name.values
    succ = inst.successful.values.astype(bool)
    V = M.reshape(len(M), -1).astype(np.float64)
    sd = V.std(0)
    diffs, w = [], []
    for a in np.unique(att):
        i = att == a
        s, u = i & succ, i & ~succ
        if s.sum() >= 5 and u.sum() >= 5:
            diffs.append(V[s].mean(0) - V[u].mean(0))
            w.append(a)
    D = np.array(diffs) / sd
    mean, frac_pos = D.mean(0), (D > 0).mean(0)
    j = 0
    for hi_ in range(NH):
        for r in range(HC.N_REG):
            for met in HC.METRICS:
                rows.append({"layer": LAYER_OF[hi_], "head": HEAD_OF[hi_], "head_label": HC.HEAD_LABELS[hi_],
                             "region": HC.REGIONS[r], "metric": met, "stratified_d_succ_minus_unsucc": mean[j],
                             "frac_attacks_succ_higher": frac_pos[j], "n_attacks": len(w)})
                j += 1
    return pd.DataFrame(rows)


# ---- plots ------------------------------------------------------------------------------------

def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", ["#f7f9fc", "#b9d3f3", BLUE, "#123f78"])


def _heat(ax, v, title, is_c, log=True, cmap=None, vmin=None, vmax=None, fmt=None):
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    A = v.reshape(HC.N_LAYERS, HC.N_HEADS)
    norm = None
    if log:
        pos = A[A > 0]
        norm = mcolors.LogNorm(vmin=pos.min(), vmax=pos.max())
    im = ax.imshow(A, cmap=cmap or _seq_cmap(), norm=norm, vmin=None if log else vmin, vmax=None if log else vmax,
                   aspect="auto", origin="lower")
    for i in np.flatnonzero(is_c):
        ax.add_patch(Rectangle((HEAD_OF[i] - 0.5, LAYER_OF[i] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.2))
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"L{l}" for l in range(12)], fontsize=7)
    ax.set_xticklabels([f"H{h}" for h in range(12)], fontsize=7)
    ax.set_xlabel("head", fontsize=8, color=INK2)
    ax.set_ylabel("encoder layer", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9, color=INK)
    return im


def plot_all(vals, vals_u, is_c, cc, conc, sc, cor, pdir: pathlib.Path, n_succ, n_q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pdir.mkdir(parents=True, exist_ok=True)
    mark = Line2D([], [], marker="s", ls="", mfc="none", mec=ORANGE, mew=2, ms=9,
                  label="causal head (Exp 11 / Exp 13 list, 18)")
    sub = f"successful attacks: {n_succ:,} instances, {n_q} queries; pre-o_proj 64-d head outputs"

    def pair_fig(fname, title, keys, log=True, note=sub):
        fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.8))
        for ax, (k, t) in zip(np.atleast_1d(axes), keys):
            im = _heat(ax, vals[k], t, is_c, log=log)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.legend(handles=[mark], loc="lower center", fontsize=8, frameon=False, ncol=1)
        fig.suptitle(title, fontsize=11, color=INK)
        fig.text(0.5, 0.905, note, ha="center", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
        fig.savefig(pdir / fname, dpi=150)
        plt.close(fig)

    pair_fig("fig1_one_minus_cosine_heads.png", "Fig. 1 — Mean-pooled 1 − cosine per encoder head (control vs attack)",
             [(("one_minus_cosine", "query"), "query"), (("one_minus_cosine", "orig_doc"), "original document")])
    pair_fig("fig2_normalized_l2_heads.png", "Fig. 2 — Mean-pooled normalized L2 per encoder head",
             [(("normalized_l2", "query"), "query"), (("normalized_l2", "orig_doc"), "original document")])
    pair_fig("fig3_one_minus_cka_heads.png", "Fig. 3 — 1 − linear CKA per encoder head (population level)",
             [(("one_minus_cka_raw", "query"), "query — raw"), (("one_minus_cka_raw", "orig_doc"), "original document — raw")])
    pair_fig("fig3b_one_minus_cka_query_centred_heads.png", "Fig. 3b — 1 − linear CKA, query-centred",
             [(("one_minus_cka_qc", "query"), "query — query-centred"),
              (("one_minus_cka_qc", "orig_doc"), "original document — query-centred")])

    # Fig 4: causal vs other heads — within-layer percentile of causal heads per metric
    keys = [(m, r) for r in HC.REGIONS for m in ("one_minus_cosine", "normalized_l2", "tw_mean_omc",
                                                  "one_minus_cka_raw", "one_minus_cka_qc")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (m, r) in enumerate(keys):
        v = vals[(m, r)]
        pct = HC.within_layer_percentile(v, LAYER_OF)
        x = i + rng.uniform(-0.18, 0.18, is_c.sum())
        late = LAYER_OF[is_c] >= 9
        ax.scatter(x[late], pct[is_c][late], s=22, color=ORANGE, zorder=3, label="causal head, L9–L11" if i == 0 else None)
        ax.scatter(x[~late], pct[is_c][~late], s=22, facecolor="white", edgecolor=ORANGE, zorder=3,
                   label="causal head, L0–L8" if i == 0 else None)
        row = cc[(cc.metric == m) & (cc.region == r)].iloc[0]
        ax.plot([i - 0.3, i + 0.3], [row.within_layer_pct_causal] * 2, color=INK, lw=2)
        ax.text(i, 1.06, f"p={row.within_layer_perm_p:.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(0.5, color=GRAY, lw=1, ls="--")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{MLAB[m].replace(' (mean-pooled)', '')}\n{RLAB[r]}" for m, r in keys], fontsize=6.5, rotation=35, ha="right")
    ax.set_ylabel("within-layer percentile (1 = most changed head in its layer)", fontsize=8, color=INK2)
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("Causal heads ranked against the other heads OF THEIR OWN LAYER\n"
                 "(bar = mean; dashed = null 0.5; p = layer-stratified permutation)", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[1]
    for j, r in enumerate(HC.REGIONS):
        for k, (m, lab) in enumerate([("one_minus_cosine", "1 − cos"), ("one_minus_cka_raw", "1 − CKA raw")]):
            v = vals[(m, r)]
            pos = j * 2.6 + k * 1.1
            bp = ax.boxplot([v[~is_c], v[is_c]], positions=[pos - 0.22, pos + 0.22], widths=0.36, patch_artist=True,
                            showfliers=False, medianprops={"color": INK})
            for patch, c in zip(bp["boxes"], [GRAY, ORANGE]):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            ax.text(pos, -0.12, f"{lab}\n{RLAB[r]}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_title("All 144 heads, global values: other (gray) vs causal (orange)\n(confounded by layer — see left panel)",
                 fontsize=9, color=INK)
    fig.suptitle("Fig. 4 — Do the previously causal encoder heads show unusually strong representational divergence?",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(pdir / "fig4_causal_vs_other_heads.png", dpi=150)
    plt.close(fig)

    # Fig 5: top-20 heads per main metric
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    for ax, (m, r) in zip(axes, [("one_minus_cosine", "query"), ("one_minus_cosine", "orig_doc"),
                                 ("one_minus_cka_raw", "query"), ("one_minus_cka_raw", "orig_doc")]):
        v = vals[(m, r)]
        o = np.argsort(-v)[:20][::-1]
        ax.barh(range(20), v[o], color=[ORANGE if is_c[i] else BLUE for i in o], height=0.7)
        ax.set_yticks(range(20))
        ax.set_yticklabels([HC.HEAD_LABELS[i] for i in o], fontsize=7)
        ax.set_title(f"{MLAB[m]}\n{RLAB[r]}", fontsize=9, color=INK)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(handles=[Line2D([], [], color=ORANGE, lw=6, label="causal head"),
                            Line2D([], [], color=BLUE, lw=6, label="other head")], fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Fig. 5 — Top-20 most divergent encoder heads (successful attacks)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(pdir / "fig5_top_divergent_heads.png", dpi=150)
    plt.close(fig)

    # Fig 6: the L6-L11 question — per-head document 1 - CKA per layer
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (m, t) in zip(axes, [("one_minus_cka_raw", "1 − CKA raw"), ("one_minus_cosine", "1 − cos (mean-pooled)")]):
        v = vals[(m, "orig_doc")]
        for L in range(12):
            i = np.flatnonzero(LAYER_OF == L)
            x = L + rng.uniform(-0.2, 0.2, len(i))
            ax.scatter(x, v[i], s=16, color=[ORANGE if is_c[k] else BLUE for k in i], alpha=0.85, zorder=3)
            ax.plot([L - 0.3, L + 0.3], [np.median(v[i])] * 2, color=INK, lw=1.5)
        ax.axvspan(5.5, 11.5, color="#f1efe8", zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
        ax.set_title(f"original document — {t}; bar = layer median", fontsize=9, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="causal head"),
                            Line2D([], [], marker="o", ls="", color=BLUE, label="other head")], fontsize=7, frameon=False)
    fig.suptitle("Fig. 6 — Is the L7→L8 geometry change broad across heads or carried by a few? (successful attacks)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(pdir / "fig6_layer_transition_heads.png", dpi=150)
    plt.close(fig)

    # Fig 7: success contrast + correlation with delta_score (within attack configuration)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.4))
    for j, r in enumerate(HC.REGIONS):
        s = sc[(sc.metric == "one_minus_cosine") & (sc.region == r)].sort_values(["layer", "head"])
        im = _heat(axes[0, j], s.stratified_d_succ_minus_unsucc.values,
                   f"{RLAB[r]}: successful − unsuccessful 1 − cos\n(within attack config, SD units)", is_c,
                   log=False, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        fig.colorbar(im, ax=axes[0, j], shrink=0.85)
        c = cor[(cor.metric == "one_minus_cosine") & (cor.region == r) & (cor.success_group == "successful")
                & (cor.relevance_group == "all")].sort_values(["layer", "head"])
        im = _heat(axes[1, j], c.spearman_within_attack.values,
                   f"{RLAB[r]}: ρ(1 − cos, Δscore), successful,\nwithin attack config", is_c,
                   log=False, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        fig.colorbar(im, ax=axes[1, j], shrink=0.85)
    fig.legend(handles=[mark], loc="lower center", fontsize=8, frameon=False)
    fig.suptitle("Fig. 7 — Success-specific change and relation to attack strength, per head", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(pdir / "fig7_success_contrast_and_delta_score.png", dpi=150)
    plt.close(fig)


def plot_cka_by_layer(cka: pd.DataFrame, e18_cka: pathlib.Path, pdir: pathlib.Path, is_c: np.ndarray):
    """Fig 8: Exp 18 fig 3 at head level — per layer, median / IQR / range over the 12 heads of linear CKA
    (successful attacks), with Exp 18's residual-stream CKA (same regions, same population) as reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ck = cka[(cka.success_group == "successful") & (cka.relevance_group == "all")]
    e18 = pd.read_csv(e18_cka)
    e18 = e18[e18.state.isin(R.LAYER_STATES) & (e18.success_group == "successful") & (e18.relevance_group == "all")]
    color = {"query": BLUE, "orig_doc": ORANGE}
    lo = min(ck.cka_raw.min(), ck.cka_query_centered.min(), e18.cka_raw.min(), e18.cka_query_centered.min()) - 0.01
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, col, t in ((axes[0], "cka_raw", "raw"), (axes[1], "cka_query_centered", "query-centred")):
        for r in HC.REGIONS:
            g = ck[ck.region == r].groupby("layer")[col]
            q = g.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).unstack()
            x = q.index.values
            ax.fill_between(x, q[0.0], q[1.0], color=color[r], alpha=0.10, lw=0)
            ax.fill_between(x, q[0.25], q[0.75], color=color[r], alpha=0.28, lw=0)
            ax.plot(x, q[0.5], color=color[r], lw=2, marker="o", ms=4, label=f"{RLAB[r]}: head median (IQR, range)")
            ref = e18[e18.region == r].sort_values("layer")
            ax.plot(ref.layer, ref[col], color=color[r], lw=1.4, ls="--", label=f"{RLAB[r]}: residual stream (Exp 18)")
        ax.set_ylim(lo, 1.005)
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
        ax.set_xlabel("encoder layer", color=INK2, fontsize=9)
        ax.set_ylabel(f"linear CKA ({t})", color=INK2, fontsize=9)
        ax.set_title(t, fontsize=9, color=INK2)
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    r0 = ck.iloc[0]
    fig.suptitle("Fig. 8 — Linear CKA between control and attacked representations per layer: 12 heads vs residual stream",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.905, f"successful attacks: {int(r0.n_instances):,} instances, {int(r0.n_queries)} queries; "
             "heads = pre-o_proj 64-d outputs of block L's self-attention (solid, bands over the layer's 12 heads); "
             "dashed = 768-d residual stream after block L", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(pdir / "fig8_linear_cka_by_layer.png", dpi=160)
    plt.close(fig)

    # Fig 8b: same data, every head on the x axis (L0H0 ... L11H11); residual CKA as a per-layer step
    X = np.arange(NH)
    fig, axes = plt.subplots(2, 1, figsize=(26, 9.5))
    for ax, col, t in ((axes[0], "cka_raw", "raw"), (axes[1], "cka_query_centered", "query-centred")):
        for i in np.flatnonzero(is_c):
            ax.axvspan(i - 0.5, i + 0.5, color=ORANGE, alpha=0.12, lw=0, zorder=0)
        for r in HC.REGIONS:
            v = ck[ck.region == r].sort_values(["layer", "head"])[col].values
            ax.plot(X, v, color=color[r], lw=1.3, marker="o", ms=3.2, label=f"{RLAB[r]}: head", zorder=3)
            ref = e18[e18.region == r].set_index("layer")[col]
            for L in range(HC.N_LAYERS):
                ax.hlines(ref[L], 12 * L - 0.4, 12 * L + 11.4, color=color[r], lw=1.6, ls="--", zorder=2,
                          label=f"{RLAB[r]}: residual stream after block (Exp 18)" if L == 0 else None)
        for b in range(12, NH, 12):
            ax.axvline(b - 0.5, color=INK2, lw=0.8, ls=":", zorder=1)
        ax.set_xticks(X)
        ax.set_xticklabels(HC.HEAD_LABELS, rotation=90, fontsize=6)
        for tl, c in zip(ax.get_xticklabels(), is_c):
            if c:
                tl.set_fontweight("bold")
        ax.set_xlim(-0.6, NH - 0.4)
        ax.set_ylabel(f"linear CKA ({t})", color=INK2, fontsize=9)
        ax.set_title(f"{t} — shaded / bold = causal head (Exp 11 / Exp 13 list, 18)", loc="left", fontsize=9, color=INK2)
        ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    fig.suptitle("Fig. 8b — Linear CKA, control vs attacked representation, for every encoder head (successful attacks)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(pdir / "fig8b_linear_cka_all_heads.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Exp 19 stage 02: head-level analysis")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--plots-only", action="store_true", help="redraw figures from the saved CSVs")
    args = p.parse_args()
    cfg = load_config(args.config)
    rc = cfg["run"]
    out = output_dir(cfg)
    s01, stage = out / "01_head_change", out / "02_analysis"
    is_c, eff = causal_sets(cfg)
    if args.plots_only:
        hs = pd.read_csv(stage / "head_summary.csv")
        cka = pd.read_csv(stage / "cka.csv")
        pj = json.loads((stage / "summary.json").read_text())
        plot_all(head_values(hs, cka), head_values(hs, cka, "unsuccessful"), is_c,
                 pd.read_csv(stage / "causal_comparison.csv"), pd.read_csv(stage / "layer_concentration.csv"),
                 pd.read_csv(stage / "success_contrast.csv"), pd.read_csv(stage / "head_correlations.csv"),
                 stage / "plots", pj["n_successful"], pj["n_queries_successful"])
        plot_cka_by_layer(cka, resolve(cfg, cfg["exp18_outputs_dir"]) / "02_analysis" / "cka.csv", stage / "plots", is_c)
        print(f"[02] plots redrawn -> {stage / 'plots'}")
        return
    if not is_already_successful(s01, ["cka.csv", "population.json"]):
        raise FileNotFoundError(f"stage 01 not complete: {s01}")
    if not args.force and is_already_successful(stage, ["head_summary.csv", "causal_comparison.csv"]):
        print(f"[02] already complete in {stage}")
        return
    t0 = time.time()
    write_status(stage, {"status": "running", "started": now()})
    inst, M = load(s01)
    print(f"[02] loaded {len(inst):,} instances, metrics {M.shape} ({time.time() - t0:.0f}s)", flush=True)
    qids = sorted(inst.qid.unique())
    hs, boots = head_summary(inst, M, qids, int(rc["n_boot"]), int(rc["boot_seed"]))
    hs.to_csv(stage / "head_summary.csv", index=False, float_format="%.9g")
    print(f"[02] head summary ({time.time() - t0:.0f}s)", flush=True)
    cka = pd.read_csv(s01 / "cka.csv")
    cka["one_minus_cka_raw"] = 1 - cka.cka_raw
    cka["one_minus_cka_query_centered"] = 1 - cka.cka_query_centered
    cka.to_csv(stage / "cka.csv", index=False, float_format="%.9g")
    vals = head_values(hs, cka)
    cc, ranks = causal_comparison(vals, is_c, eff, boots, 20000, int(rc["boot_seed"]))
    cc.to_csv(stage / "causal_comparison.csv", index=False, float_format="%.6g")
    ranks.to_csv(stage / "causal_head_ranks.csv", index=False, float_format="%.6g")
    top = []
    for (m, r), v in vals.items():
        for k, i in enumerate(np.argsort(-v)[:18]):
            top.append({"metric": m, "region": r, "rank": k + 1, "head_label": HC.HEAD_LABELS[i],
                        "layer": LAYER_OF[i], "value": v[i], "causal": bool(is_c[i]), "exp11_effect": eff[i]})
    pd.DataFrame(top).to_csv(stage / "topk_heads.csv", index=False, float_format="%.6g")
    conc = layer_concentration(vals)
    conc.to_csv(stage / "layer_concentration.csv", index=False, float_format="%.6g")
    sc = success_contrast(inst, M)
    sc.to_csv(stage / "success_contrast.csv", index=False, float_format="%.6g")
    vr, vn = head_values(hs, cka, "successful", "relevant"), head_values(hs, cka, "successful", "nonrelevant")
    rel = []
    for (m, r) in vals:
        for i in range(NH):
            rel.append({"metric": m, "region": r, "head_label": HC.HEAD_LABELS[i], "layer": LAYER_OF[i],
                        "relevant": vr[(m, r)][i], "nonrelevant": vn[(m, r)][i],
                        "ratio_nonrel_over_rel": vn[(m, r)][i] / vr[(m, r)][i] if vr[(m, r)][i] else np.nan})
    pd.DataFrame(rel).to_csv(stage / "relevance_contrast.csv", index=False, float_format="%.6g")
    print(f"[02] causal comparison etc. ({time.time() - t0:.0f}s)", flush=True)
    cor = correlations(inst, M)
    cor.to_csv(stage / "head_correlations.csv", index=False, float_format="%.6g")
    print(f"[02] correlations ({time.time() - t0:.0f}s)", flush=True)
    ms = inst.successful.values.astype(bool)
    summ = {"n_instances": len(inst), "n_successful": int(ms.sum()), "n_queries_successful": int(inst.qid[ms].nunique()),
            "causal_heads": [HC.HEAD_LABELS[i] for i in np.flatnonzero(is_c)]}
    (stage / "summary.json").write_text(json.dumps(summ, indent=1))
    plot_all(vals, head_values(hs, cka, "unsuccessful"), is_c, cc, conc, sc, cor, stage / "plots",
             summ["n_successful"], summ["n_queries_successful"])
    plot_cka_by_layer(cka, resolve(cfg, cfg["exp18_outputs_dir"]) / "02_analysis" / "cka.csv", stage / "plots", is_c)
    write_status(stage, {"status": "success", "finished": now(), "seconds": time.time() - t0})
    print(f"[02] done in {time.time() - t0:.0f}s -> {stage}")


if __name__ == "__main__":
    main()
