"""
scripts/02_analyze_repr_change.py
==================================
Exp 18 stage 02 — analysis + figures of stage 01 (CPU; no model).

Inputs : 01_repr_change/per_attack/*.parquet (one row per instance x state x region), cka.csv
Groups : success {successful, unsuccessful, all} x relevance {all, relevant (qrel 2/3), nonrelevant (qrel 0)};
         never pooled into an unlabelled average. successful <=> delta_score > 0 (Exp 16 stage 18).
Uncertainty: instance-weighted means with a QUERY-level bootstrap (resample qids with replacement,
         run.n_boot replicates, seed run.boot_seed), 95% percentile CI.
Correlation with delta_score (per state x region x change metric x group), change metrics oriented so that
         larger = more change (1 - cos, normalized L2, and their token-wise means):
           spearman_rho / p_value   pooled over instances (p ignores query clustering -> descriptive only)
           spearman_within_attack   Pearson of ranks computed WITHIN each attack configuration (removes
                                    between-configuration differences such as repetition count)
           median_query_rho, n_queries_rho_pos / n_queries_rho   per-query Spearman (queries with >= 10 inst.)
CKA is population level: it is only tabulated/plotted, never correlated with delta_score.

Outputs: 02_analysis/{layer_summary.csv, correlations.csv, cka.csv, population.json, summary.json}
         02_analysis/plots/fig{1..6}_*.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp18lib  # noqa: E402,F401

from exp16lib.run_utils import is_already_successful, now, write_status  # noqa: E402
from exp18lib import repr_change as R  # noqa: E402
from exp18lib.config import load_config, output_dir  # noqa: E402

S, G = len(R.STATES), len(R.REGIONS)
METRICS = {  # summary column -> (parquet column, transform to "change" orientation)
    "one_minus_cosine": "one_minus_cosine",
    "normalized_l2": "normalized_l2",
    "tokenwise_mean_one_minus_cosine": "tokenwise_mean_one_minus_cosine",
    "tokenwise_median_one_minus_cosine": "tokenwise_median_cosine",        # 1 - median cos (monotone)
    "tokenwise_q10_one_minus_cosine": "tokenwise_q10_cosine",              # 1 - q10 cos = q90 of 1 - cos
    "tokenwise_mean_normalized_l2": "tokenwise_mean_normalized_l2",
    "tokenwise_median_normalized_l2": "tokenwise_median_normalized_l2",
}
CORR_METRICS = ["one_minus_cosine", "normalized_l2", "tokenwise_mean_one_minus_cosine", "tokenwise_mean_normalized_l2"]
META = ["attack_name", "qid", "docid", "relevance_group", "successful", "delta_score", "repetitions", "attack_position"]

# reference categorical slots 1-4 (dataviz palette, light), fixed order = region order
REGION_COLOR = {"query": "#2a78d6", "orig_doc": "#eb6834", "whole_shared": "#1baf7a", "whole_full_visible": "#eda100"}
REGION_LABEL = {"query": "query", "orig_doc": "original document", "whole_shared": "whole (shared tokens)",
                "whole_full_visible": "whole (full visible)"}
REGION_STYLE = {"query": "-", "orig_doc": "-", "whole_shared": "-", "whole_full_visible": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(stage25: pathlib.Path):
    """-> meta DataFrame [N] and metric arrays [N, S, G] (instance-major order of the stage-25 shards)."""
    metas, arrs = [], {k: [] for k in METRICS}
    for f in sorted((stage25 / "per_attack").glob("*.parquet")):
        t = pq.read_table(f, columns=META + ["state", "region"] + sorted(set(METRICS.values()))).to_pandas()
        n = len(t) // (S * G)
        if n * S * G != len(t):
            raise RuntimeError(f"{f.name}: row count not a multiple of states x regions")
        st = t.state.astype(str).values.reshape(n, S, G)
        rg = t.region.astype(str).values.reshape(n, S, G)
        if not ((st == np.array(R.STATES)[None, :, None]).all() and (rg == np.array(R.REGIONS)[None, None, :]).all()):
            raise RuntimeError(f"{f.name}: unexpected state/region layout")
        m = t.iloc[::S * G][META].reset_index(drop=True)
        for c in ("attack_name", "qid", "docid", "relevance_group", "attack_position"):
            m[c] = m[c].astype(str)
        metas.append(m)
        for k, col in METRICS.items():
            v = t[col].values.astype(np.float64).reshape(n, S, G)
            arrs[k].append(1.0 - v if col.endswith("cosine") and k != col else v)
    meta = pd.concat(metas, ignore_index=True)
    return meta, {k: np.concatenate(v) for k, v in arrs.items()}


def layer_summary(meta, A, qids, n_boot, seed):
    q = pd.Categorical(meta.qid, categories=qids).codes
    Q = len(qids)
    rows = []
    for sg in R.SUCCESS_GROUPS:
        for rg in R.RELEVANCE_GROUPS:
            m = R.group_mask(meta.successful.values, meta.relevance_group.values, sg, rg)
            cnt = np.bincount(q[m], minlength=Q).astype(float)
            for k, X in A.items():
                x = X[m].reshape(m.sum(), -1)
                sums = np.stack([np.bincount(q[m], weights=x[:, j], minlength=Q) for j in range(S * G)], 1)
                mean, lo, hi = R.query_bootstrap_mean(sums.reshape(Q, S, G), cnt, n_boot, seed)
                med = np.median(X[m], axis=0) if m.sum() else np.full((S, G), np.nan)
                for si, s in enumerate(R.STATES):
                    for gi, r in enumerate(R.REGIONS):
                        if np.isnan(mean[si, gi]):
                            continue
                        rows.append({"state": s, "layer": R.STATE_LAYER[s], "region": r, "metric": k,
                                     "success_group": sg, "relevance_group": rg, "mean": mean[si, gi],
                                     "ci_lo": lo[si, gi], "ci_hi": hi[si, gi], "median": med[si, gi],
                                     "n_instances": int(m.sum()), "n_queries": int((cnt > 0).sum())})
    df = pd.DataFrame(rows)
    # cosine rows (mean cos = 1 - mean(1 - cos); CI flipped)
    c = df[df.metric.isin(["one_minus_cosine", "tokenwise_mean_one_minus_cosine", "tokenwise_median_one_minus_cosine"])].copy()
    c["metric"] = c.metric.str.replace("one_minus_cosine", "cosine")
    c["mean"], c["ci_lo"], c["ci_hi"], c["median"] = 1 - c["mean"], 1 - c["ci_hi"], 1 - c["ci_lo"], 1 - c["median"]
    return pd.concat([df, c], ignore_index=True)


def within_ranks(x, groups):
    r = np.empty(len(x))
    for g in np.unique(groups):
        i = groups == g
        r[i] = stats.rankdata(x[i])
    return r


def correlations(meta, A):
    rows = []
    att = meta.attack_name.values
    qid = meta.qid.values
    dsc = meta.delta_score.values
    for sg in R.SUCCESS_GROUPS:
        for rg in R.RELEVANCE_GROUPS:
            m = R.group_mask(meta.successful.values, meta.relevance_group.values, sg, rg)
            y = dsc[m]
            y_wr = within_ranks(y, att[m])
            qs = [qq for qq in np.unique(qid[m]) if (qid[m] == qq).sum() >= 10]
            qmask = {qq: qid[m] == qq for qq in qs}
            for k in CORR_METRICS:
                X = A[k][m]
                for si, s in enumerate(R.STATES):
                    for gi, r in enumerate(R.REGIONS):
                        x = X[:, si, gi]
                        if np.isnan(x).all() or len(x) < 10:
                            continue
                        rho, p = stats.spearmanr(x, y)
                        rw = np.corrcoef(within_ranks(x, att[m]), y_wr)[0, 1]
                        qr = np.array([stats.spearmanr(x[i], y[i])[0] for i in qmask.values()])
                        qr = qr[np.isfinite(qr)]
                        rows.append({"state": s, "layer": R.STATE_LAYER[s], "region": r, "metric": k,
                                     "success_group": sg, "relevance_group": rg, "spearman_rho": rho,
                                     "p_value": p, "spearman_within_attack": rw,
                                     "median_query_rho": float(np.median(qr)) if len(qr) else np.nan,
                                     "n_queries_rho_pos": int((qr > 0).sum()), "n_queries_rho": int(len(qr)),
                                     "n_instances": int(m.sum()), "n_queries": int(len(np.unique(qid[m])))})
    return pd.DataFrame(rows)


# ---- plots ------------------------------------------------------------------------------------

def _style(ax, ylabel):
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
    ax.set_xlabel("encoder layer (block output, before final norm)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _lines(ax, d, regions, direct=True, log=False):
    ends = []
    for r in regions:
        x = d[d.region == r].sort_values("layer")
        if x.empty:
            continue
        c = REGION_COLOR[r]
        ax.plot(x.layer, x["mean"], REGION_STYLE[r], color=c, lw=2, marker="o", ms=4, label=REGION_LABEL[r])
        if "ci_lo" in x:
            ax.fill_between(x.layer, x.ci_lo, x.ci_hi, color=c, alpha=0.18, lw=0)
        ends.append((r, float(x.layer.iloc[-1]), float(x["mean"].iloc[-1])))
    if log:
        ax.set_yscale("log")
    ax.set_xlim(-0.3, 13.8 if direct else 11.3)
    if direct and ends:
        _direct_labels(ax, ends)


def _direct_labels(ax, ends, gap=0.06):
    """End-of-line labels, dodged vertically (axes-fraction units) so they never overlap."""
    ax.figure.canvas.draw()
    to_ax = ax.transAxes.inverted()
    pts = sorted(((to_ax.transform(ax.transData.transform((x, y)))[1], r, x, y) for r, x, y in ends))
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = max(0.0, ys[-1] - 0.97)
    for (y0, r, x, y), ya in zip(pts, ys):
        ax.annotate(REGION_LABEL[r], (x, y), xytext=(x + 0.25, 0), textcoords=("data", "axes fraction"),
                    fontsize=7, color=INK, va="center")
        ax.texts[-1].set_position((x + 0.25, ya - shift))


def plot_all(ls, cka, cor, pdir: pathlib.Path, n_info: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir.mkdir(parents=True, exist_ok=True)
    L = ls[ls.state.isin(R.LAYER_STATES)]
    base = L[(L.success_group == "successful") & (L.relevance_group == "all")]
    sub = (f"successful attacks: {n_info['succ']:,} instances, {n_info['succ_q']} queries; "
           f"mean ± 95% query-bootstrap CI")

    # Fig 1: cosine (linear) + 1 - cosine (log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _lines(axes[0], base[base.metric == "cosine"], R.REGIONS)
    _style(axes[0], "cosine(control, attack) of mean-pooled state")
    _lines(axes[1], base[base.metric == "one_minus_cosine"], R.REGIONS, log=True)
    _style(axes[1], "1 − cosine (log scale)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Fig. 1 — Cosine similarity, padded control vs attacked input, per encoder layer", fontsize=11, color=INK)
    fig.text(0.5, 0.905, sub, ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(pdir / "fig1_cosine_by_layer.png", dpi=160)
    plt.close(fig)

    # Fig 2: normalized L2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _lines(axes[0], base[base.metric == "normalized_l2"], R.REGIONS)
    _style(axes[0], "‖R_attack − R_control‖ / ‖R_control‖")
    _lines(axes[1], base[base.metric == "normalized_l2"], R.REGIONS, log=True)
    _style(axes[1], "normalized L2 (log scale)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Fig. 2 — Normalized L2 displacement of the mean-pooled state, per encoder layer", fontsize=11, color=INK)
    fig.text(0.5, 0.905, sub, ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(pdir / "fig2_normalized_l2_by_layer.png", dpi=160)
    plt.close(fig)

    # Fig 3: linear CKA raw + query-centred (successful, all relevance)
    ck = cka[cka.state.isin(R.LAYER_STATES) & (cka.success_group == "successful") & (cka.relevance_group == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    lo = min(ck.cka_raw.min(), ck.cka_query_centered.min()) - 0.01
    for ax in axes:
        ax.set_ylim(lo, 1.005)                               # common y-range, fixed BEFORE direct labels are placed
    for ax, col, t in ((axes[0], "cka_raw", "raw"), (axes[1], "cka_query_centered", "query-centred")):
        _lines(ax, ck.rename(columns={col: "mean"}).drop(columns=[c for c in ("ci_lo",) if c in ck]), R.REGIONS)
        _style(ax, f"linear CKA ({t})")
        ax.set_title(f"{t}", fontsize=9, color=INK2)
        ax.tick_params(labelleft=True)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    r0 = ck.iloc[0]
    fig.suptitle("Fig. 3 — Linear CKA between control and attacked representation matrices (population level)",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.905, f"successful attacks: {int(r0.n_instances):,} instances x 768-d mean-pooled states, "
             f"{int(r0.n_queries)} queries; query-centred = each condition centred on its own per-query mean",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(pdir / "fig3_linear_cka_by_layer.png", dpi=160)
    plt.close(fig)

    # Fig 4: successful vs unsuccessful (1 - cos, log), per region
    def split_fig(fname, title, filt, groups, colors, labels, metric="one_minus_cosine", ylabel="1 − cosine (log)"):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
        for ax, r in zip(axes, R.REGIONS):
            for g, c, lab in zip(groups, colors, labels):
                x = L[filt(g) & (L.metric == metric) & (L.region == r)].sort_values("layer")
                ax.plot(x.layer, x["mean"], color=c, lw=2, marker="o", ms=3.5, label=lab)
                ax.fill_between(x.layer, x.ci_lo, x.ci_hi, color=c, alpha=0.18, lw=0)
            ax.set_yscale("log")
            ax.set_title(REGION_LABEL[r], fontsize=9, color=INK)
            _style(ax, ylabel if r == "query" else "")
            ax.set_xlabel("encoder layer", fontsize=8, color=INK2)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(title, fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(pdir / fname, dpi=150)
        plt.close(fig)

    split_fig("fig4_successful_vs_unsuccessful.png",
              "Fig. 4 — Successful vs unsuccessful attacks (all eligible base documents); 1 − cosine, 95% query-bootstrap CI",
              lambda g: (L.success_group == g) & (L.relevance_group == "all"),
              ["successful", "unsuccessful"], ["#2a78d6", "#eb6834"], ["successful", "unsuccessful"])
    split_fig("fig5_relevant_vs_nonrelevant.png",
              "Fig. 5 — Successful attacks: relevant (qrel 2/3) vs non-relevant (qrel 0) base documents; 1 − cosine",
              lambda g: (L.success_group == "successful") & (L.relevance_group == g),
              ["relevant", "nonrelevant"], ["#2a78d6", "#eb6834"], ["relevant base doc", "non-relevant base doc"])

    # Fig 6: Spearman(change, delta_score) heatmaps, all instances and successful only
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.2), gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    for i, sg in enumerate(["all", "successful"]):
        for j, (k, col, t) in enumerate([("one_minus_cosine", "spearman_rho", "pooled Spearman"),
                                         ("one_minus_cosine", "spearman_within_attack", "within attack configuration")]):
            ax = axes[i, j]
            c = cor[(cor.metric == k) & (cor.success_group == sg) & (cor.relevance_group == "all")
                    & cor.state.isin(R.LAYER_STATES)]
            M = c.pivot(index="region", columns="layer", values=col).reindex(R.REGIONS)
            im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
            for (a, b), v in np.ndenumerate(M.values):
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.4 else INK)
            ax.set_yticks(range(G))
            ax.set_yticklabels([REGION_LABEL[r] for r in R.REGIONS] if j == 0 else [], fontsize=8)
            ax.set_xticks(range(12))
            ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=8)
            ax.set_title(f"{sg} attacks — {t}", fontsize=9, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman ρ(1 − cos, Δscore)")
    fig.suptitle("Fig. 6 — Does representation change track attack strength? ρ between 1 − cosine and Δscore",
                 fontsize=11, color=INK)
    fig.savefig(pdir / "fig6_change_vs_delta_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Exp 18 stage 02: representation-change analysis")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--plots-only", action="store_true", help="redraw figures from the saved CSVs")
    args = p.parse_args()
    cfg = load_config(args.config)
    rc = cfg["run"]
    out = output_dir(cfg)
    s01 = out / "01_repr_change"
    stage = out / "02_analysis"
    if args.plots_only:                                       # redraw from the saved tables (no recompute)
        ls = pd.read_csv(stage / "layer_summary.csv", dtype={"state": str})
        pj = json.loads((stage / "population.json").read_text())["by_group"]["successful/all"]
        plot_all(ls, pd.read_csv(stage / "cka.csv", dtype={"state": str}),
                 pd.read_csv(stage / "correlations.csv", dtype={"state": str}), stage / "plots",
                 {"succ": pj["n_instances"], "succ_q": pj["n_queries"]})
        print(f"[02] plots redrawn -> {stage / 'plots'}")
        return
    if not is_already_successful(s01, ["cka.csv", "population.json"]):
        raise FileNotFoundError(f"stage 01 not complete: {s01}")
    if not args.force and is_already_successful(stage, ["layer_summary.csv", "correlations.csv"]):
        print(f"[02] already complete in {stage}")
        return
    t0 = time.time()
    write_status(stage, {"status": "running", "started": now()})
    meta, A = load(s01)
    print(f"[02] loaded {len(meta):,} instances ({time.time() - t0:.0f}s)", flush=True)
    if meta.duplicated(["attack_name", "qid", "docid"]).any():
        raise RuntimeError("duplicate instance")
    if not (meta.successful.values == (meta.delta_score.values > 0)).all():
        raise RuntimeError("successful != delta_score > 0")
    qids = sorted(meta.qid.unique())
    ls = layer_summary(meta, A, qids, int(rc["n_boot"]), int(rc["boot_seed"]))
    ls.to_csv(stage / "layer_summary.csv", index=False, float_format="%.9g")
    print(f"[02] layer summary ({time.time() - t0:.0f}s)", flush=True)
    cor = correlations(meta, A)
    cor.to_csv(stage / "correlations.csv", index=False, float_format="%.6g")
    print(f"[02] correlations ({time.time() - t0:.0f}s)", flush=True)
    cka = pd.read_csv(s01 / "cka.csv", dtype={"state": str})
    cka.to_csv(stage / "cka.csv", index=False, float_format="%.9g")
    popn = {"n_instances": len(meta), "n_queries": len(qids), "n_pairs": int(meta.qid.str.cat(meta.docid, "_").nunique()),
            "n_attacks": int(meta.attack_name.nunique()),
            "by_group": {f"{sg}/{rg}": {"n_instances": int(m.sum()), "n_queries": int(meta.qid[m].nunique())}
                         for sg in R.SUCCESS_GROUPS for rg in R.RELEVANCE_GROUPS
                         for m in [R.group_mask(meta.successful.values, meta.relevance_group.values, sg, rg)]}}
    (stage / "population.json").write_text(json.dumps(popn, indent=1))
    ms = R.group_mask(meta.successful.values, meta.relevance_group.values, "successful", "all")
    plot_all(ls, cka, cor, stage / "plots",
             {"succ": int(ms.sum()), "succ_q": int(meta.qid[ms].nunique())})
    write_status(stage, {"status": "success", "finished": now(), "seconds": time.time() - t0})
    print(f"[02] done in {time.time() - t0:.0f}s -> {stage}")


if __name__ == "__main__":
    main()
