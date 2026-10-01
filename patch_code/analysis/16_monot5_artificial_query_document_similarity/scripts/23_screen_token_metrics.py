"""
scripts/23_screen_token_metrics.py
===================================
Stage 23 — SCREEN alternative query/document similarity metrics on the stage-22 token
sample only (no forward pass, no full-population inference). Metric definitions:
exp16lib/metrics_screen.py.

Validation first (stops on failure): baseline cos_full == stage-22 capture cosines (fp16
storage tolerance) and == TokenSample.pooled_cosine; mask integrity (query/doc disjoint,
no template/pad/masked slots in pools, inserted region == injected tokens only); decoded
token inspection file.

A. Genuine relevance separation — CLEAN sequences, qrel 2/3 vs qrel 0 (one sequence per
   document), per metric x head: means, medians, difference, Cohen d, raw AUROC
   (relevant = positive; > 0.5 = higher for relevant), CV-oriented AUROC (sign chosen on
   training queries, scored on held-out queries), and per metric a CV best-head AUROC
   (head AND sign chosen on training queries only). Folds = stage-17 fold map (5, seed 42).
B. Successful attacks vs their own padded control: delta = metric(attack, region) -
   metric(control, document), per group; instance-level and equal-document-weight means,
   medians, query-cluster SE, Spearman(delta_metric, delta_score).

Outputs: 23_metric_screen/{metrics.npz, validation.json, mask_inspection.txt,
  A_separation_per_head.csv, A_separation_layer.csv, A_cv_best_head.csv,
  B_paired_delta_per_head.csv, B_paired_delta_layer.csv, B_region_levels_per_head.csv,
  comparison_table.{csv,md}, summary.json, plots/*.png}
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import metrics_screen as MS  # noqa: E402
from exp16lib import paired as P  # noqa: E402
from exp16lib.anomaly import auroc, query_folds  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_tokenizer, now, stage_argparser, write_status  # noqa: E402
from exp16lib.token_sample import TokenSample  # noqa: E402

_spec = importlib.util.spec_from_file_location("p21", EXP_DIR / "scripts" / "21_plot_paired.py")
P21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P21)
P5 = P21.P5

STAGE = "23_metric_screen"
MAIN = ["cos", "ccos", "dot", "qnorm", "dnorm", "ms_mean", "ms_median", "ms_top3"]
NAMES = {"cos": "mean-pooled cosine (baseline)", "ccos": "centered cosine", "dot": "dot product (q̄·d̄)",
         "qnorm": "query norm ||q̄||", "dnorm": "document norm ||d̄||", "ms_mean": "MaxSim, mean over query tokens",
         "ms_median": "MaxSim, median", "ms_top3": "MaxSim, top-3 mean"}
REL, NON = "relevant", "nonrelevant"


def col(m, r="full"):
    return "qnorm" if m == "qnorm" else f"{m}_{r}"


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def cluster_se_mat(X, clusters):
    n = len(X)
    r = pd.DataFrame(X - X.mean(0)).groupby(np.asarray(clusters)).sum().values
    G = len(r)
    return np.sqrt(G / (G - 1) * (r ** 2).sum(0)) / n


def main():
    args = stage_argparser("Exp 16 stage 23: screen alternative similarity metrics on the token sample").parse_args()
    cfg = load_config(args.config)
    P.configure("all_layers")                  # 144 heads: plot helpers (head_axis) must use layers 0-11
    out = output_dir(cfg)
    src = out / "22_token_sample"
    if not is_already_successful(src, ["heads.npy", "tokens.npz", "sequences.csv"]):
        raise FileNotFoundError(f"run stage 22 first ({src})")
    st = out / STAGE
    pdir = st / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    write_status(st, {"status": "running", "started": now()})
    t0 = time.time()
    ts = TokenSample(src)
    meta = ts.meta.copy()
    n = len(meta)
    fm = json.loads((out / P.MANIFEST_DIR / "fold_map.json").read_text())
    if query_folds(fm["qids"], 5, 42) != fm["fold_of"]:
        raise RuntimeError("stage-17 fold map is not query_folds(qids, 5, 42)")
    meta["fold"] = meta.qid.map(fm["fold_of"]).astype(int)
    labels = [f"L{L}H{h}" for L in range(12) for h in range(12)]
    layer_of = np.repeat(np.arange(12), 12)
    canon = {h.label for h in encoder_heads()}
    tok = load_tokenizer(cfg)
    T = ts.tok

    # ---- validation 1: masks -----------------------------------------------------------------------
    val = {}
    qm, dm, im, am = (T[k].astype(bool) for k in ("query_mask", "doc_mask", "inserted_mask", "attention_mask"))
    ids = T["input_ids"]
    kind_tok = np.repeat(meta.kind.values, meta.n_tokens.values)
    pad = tok.pad_token_id
    template = set(tok.convert_tokens_to_ids(["▁", "Query", ":", "▁Document", "▁Relevan", "t", "</s>"]))
    val["query_doc_disjoint"] = bool(not (qm & dm).any())
    val["pools_attention_1"] = bool(am[qm | dm].all())
    val["pad_in_pools"] = int(((ids == pad) & (qm | dm)).sum())
    val["control_inserted_in_doc_pool"] = int((im & dm & (kind_tok == "control")).sum())
    val["control_inserted_slots_are_masked_pads"] = bool(((ids[im & (kind_tok == "control")] == pad).all()
                                                          and not am[im & (kind_tok == "control")].any()))
    val["clean_has_inserted"] = int((im & (kind_tok == "clean")).sum())
    ins_outside_doc = im & ~dm & (kind_tok == "attack")
    val["attack_inserted_outside_doc_field"] = int(ins_outside_doc.sum())     # Exp 01 boundary shift (template ':')
    # every token outside both pools is a template/pad token (query/doc are complete spans)
    outside = ~(qm | dm)
    val["non_template_tokens_outside_pools"] = int((~np.isin(ids[outside], list(template | {pad}))).sum())
    # inserted doc tokens decode to the attack token
    bad = 0
    for i in meta.index[meta.kind == "attack"]:
        a, b = int(meta.token_offset[i]), int(meta.token_offset[i] + meta.n_tokens[i])
        txt = tok.decode(ids[a:b][(im & dm)[a:b]]).lower()
        if meta.attack_token[i].lower() not in txt:
            bad += 1
    val["attack_inserted_region_not_attack_token"] = bad
    ok = (val["query_doc_disjoint"] and val["pools_attention_1"] and val["pad_in_pools"] == 0
          and val["control_inserted_in_doc_pool"] == 0 and val["control_inserted_slots_are_masked_pads"]
          and val["clean_has_inserted"] == 0 and val["non_template_tokens_outside_pools"] == 0 and bad == 0)
    if not ok:
        (st / "validation.json").write_text(json.dumps(val, indent=2))
        raise SystemExit(f"STOP: mask validation failed: {val}")
    # manual inspection file
    lines = []
    for kind, grp in (("clean", REL), ("control", NON), ("attack", REL), ("attack", NON)):
        i = meta.index[(meta.kind == kind) & (meta.relevance_group == grp) &
                       (meta.attack_name.fillna("").str.contains("random") | (kind == "clean"))][0]
        s = ts[int(i)]
        reg = MS.region_masks(s["query_mask"], s["doc_mask"], s["inserted_mask"], kind)
        dec = lambda m: " ".join(tok.convert_ids_to_tokens(s["input_ids"][m].tolist()))  # noqa: E731
        lines += [f"=== seq {i} kind={kind} group={grp} pair={s['meta']['pair_id']} attack={s['meta']['attack_name']}",
                  f"QUERY  ({s['query_mask'].sum()}): {dec(s['query_mask'])}",
                  f"FULL   ({reg['full'].sum()}): {dec(reg['full'])}",
                  f"ORIG   ({reg['orig'].sum()}): {dec(reg['orig'])}",
                  f"INS    ({0 if reg['ins'] is None else reg['ins'].sum()}): {'' if reg['ins'] is None else dec(reg['ins'])}",
                  f"OUTSIDE POOLS: {dec(~(s['query_mask'] | s['doc_mask']))}", ""]
    (st / "mask_inspection.txt").write_text("\n".join(lines))

    # ---- centering background μ (clean, training queries, label-free) ------------------------------
    qsum, qcnt = {}, {}
    for i in meta.index[meta.kind == "clean"]:
        s = ts[int(i)]
        m = s["query_mask"] | s["doc_mask"]
        H = s["heads"][m].astype(np.float64).reshape(int(m.sum()), 144, 64)
        q = s["meta"]["qid"]
        qsum[q] = qsum.get(q, 0) + H.sum(0)
        qcnt[q] = qcnt.get(q, 0) + int(m.sum())
    mu = {}
    for f in range(5):
        tr = [q for q in qsum if fm["fold_of"][q] != f]
        mu[f] = (sum(qsum[q] for q in tr) / sum(qcnt[q] for q in tr)).astype(np.float32)

    # ---- metrics ------------------------------------------------------------------------------------
    M = np.full((n, len(MS.COLUMNS), 144), np.nan, dtype=np.float32)
    for i in range(n):
        s = ts[i]
        reg = MS.region_masks(s["query_mask"], s["doc_mask"], s["inserted_mask"], s["meta"]["kind"])
        H = s["heads"].reshape(len(s["input_ids"]), 144, 64)
        M[i] = MS.sequence_metrics(H, s["query_mask"], reg, mu[int(meta.fold[i])])
        if i % 1000 == 0:
            print(f"[23] metrics {i}/{n} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(st / "metrics.npz", M=M, columns=np.array(MS.COLUMNS), heads=np.array(labels),
                        seq_id=meta.seq_id.values)
    ci = {c: j for j, c in enumerate(MS.COLUMNS)}

    # ---- validation 2: baseline reproduction --------------------------------------------------------
    cap = meta[[f"cos_{l}" for l in labels]].values
    d_cap = float(np.nanmax(np.abs(M[:, ci["cos_full"]] - cap)))
    rng = np.random.default_rng(0)
    d_pc = max(abs(TokenSample.pooled_cosine(ts[int(i)], L, h) - M[int(i), ci["cos_full"], L * 12 + h])
               for i in rng.choice(n, 200, replace=False) for L, h in ((0, 3), (6, 6), (11, 2), (11, 11)))
    val.update({"baseline_vs_stage22_capture_max_abs_diff": d_cap, "baseline_vs_pooled_cosine_max_abs_diff": float(d_pc),
                "orig_equals_full_for_control": bool(np.allclose(M[meta.kind != "attack"][:, ci["cos_orig"]],
                                                                 M[meta.kind != "attack"][:, ci["cos_full"]], equal_nan=True)),
                "ins_defined_only_for_attack": bool(np.isnan(M[meta.kind != "attack"][:, ci["cos_ins"]]).all()
                                                    and not np.isnan(M[meta.kind == "attack"][:, ci["cos_ins"]]).any())})
    (st / "validation.json").write_text(json.dumps(val, indent=2))
    if d_cap > 5e-4 or d_pc > 1e-6 or not val["orig_equals_full_for_control"] or not val["ins_defined_only_for_attack"]:
        raise SystemExit(f"STOP: baseline reproduction failed: {val}")
    print(json.dumps(val, indent=1))

    # ---- A. genuine relevance separation (clean) -----------------------------------------------------
    cl = meta.kind == "clean"
    cm = meta[cl].reset_index(drop=True)
    Mc = M[cl.values]
    isrel = (cm.relevance_group == REL).values
    fold = cm.fold.values
    rowsA, cvbest = [], []
    for m in MAIN:
        V = Mc[:, ci[col(m)]]                                            # [docs, 144]
        for j, l in enumerate(labels):
            a, b = V[isrel, j], V[~isrel, j]
            ori = np.empty(len(V))
            for f in range(5):                                           # CV-oriented scores
                tr = fold != f
                sgn = np.sign(V[tr & isrel, j].mean() - V[tr & ~isrel, j].mean()) or 1.0
                ori[fold == f] = sgn * V[fold == f, j]
            rowsA.append({"metric": m, "head_label": l, "layer": layer_of[j], "mean_rel": a.mean(), "mean_nonrel": b.mean(),
                          "median_rel": np.median(a), "median_nonrel": np.median(b), "difference": a.mean() - b.mean(),
                          "cohen_d": cohen_d(a, b), "auroc_raw": auroc(a, b),
                          "auroc_cv_oriented": auroc(ori[isrel], ori[~isrel])})
        pos, neg, picks = [], [], []                                     # CV best head (head + sign on training only)
        for f in range(5):
            tr, te = fold != f, fold == f
            d_tr = np.array([cohen_d(V[tr & isrel, j], V[tr & ~isrel, j]) for j in range(144)])
            j = int(np.nanargmax(np.abs(d_tr)))
            sgn = np.sign(d_tr[j])
            pos.append(sgn * V[te & isrel, j]); neg.append(sgn * V[te & ~isrel, j]); picks.append(labels[j])
        cvbest.append({"metric": m, "cv_best_head_auroc": auroc(np.concatenate(pos), np.concatenate(neg)),
                       "heads_chosen_per_fold": ",".join(picks)})
    A = pd.DataFrame(rowsA)
    A["abs_d"] = A.cohen_d.abs()
    A["rank_in_metric"] = A.groupby("metric").abs_d.rank(ascending=False, method="first").astype(int)
    A.to_csv(st / "A_separation_per_head.csv", index=False)
    AL = A.groupby(["metric", "layer"]).agg(mean_abs_d=("abs_d", "mean"), max_abs_d=("abs_d", "max"),
                                            max_auroc_cv=("auroc_cv_oriented", "max")).reset_index()
    AL.to_csv(st / "A_separation_layer.csv", index=False)
    CVB = pd.DataFrame(cvbest)
    CVB.to_csv(st / "A_cv_best_head.csv", index=False)

    # ---- B. successful attacks vs own control ---------------------------------------------------------
    atk = meta[meta.kind == "attack"]
    ctl = meta[meta.kind == "control"].set_index(["attack_name", "pair_id"])
    succ = atk[atk.successful == True]  # noqa: E712
    ia = succ.index.values
    ic = ctl.loc[list(zip(succ.attack_name, succ.pair_id)), "seq_id"].values
    assert (meta.loc[ic, "pair_id"].values == succ.pair_id.values).all() and (meta.loc[ic, "kind"] == "control").all()
    assert np.allclose(meta.loc[ia, "score"].values - meta.loc[ic, "score"].values, succ.delta_score.values,
                       rtol=0, atol=1e-6)                             # CSV stores both at 9 significant digits
    rowsB, levels = [], []
    grp = succ.relevance_group.values
    for m in MAIN:
        regions = ["full"] if m == "qnorm" else MS.REGIONS
        for r in regions:
            D = M[ia, ci[col(m, r)]].astype(np.float64) - M[ic, ci[col(m, "full")]].astype(np.float64)   # [inst, 144]
            for g in (REL, NON):
                k = grp == g
                Dg, sub = D[k], succ[k]
                se = cluster_se_mat(Dg, sub.qid.values)
                docm = pd.DataFrame(Dg).groupby(sub.pair_id.values).mean().values
                rho = [spearmanr(Dg[:, j], sub.delta_score.values)[0] for j in range(144)]
                for j, l in enumerate(labels):
                    rowsB.append({"metric": m, "region": r, "relevance_group": g, "head_label": l, "layer": layer_of[j],
                                  "n_instances": int(k.sum()), "n_docs": docm.shape[0],
                                  "mean_delta": Dg[:, j].mean(), "median_delta": np.median(Dg[:, j]),
                                  "se_delta_query_cluster": se[j], "mean_delta_doc_weighted": docm[:, j].mean(),
                                  "frac_delta_positive": (Dg[:, j] > 0).mean(),
                                  "q10": np.quantile(Dg[:, j], 0.1), "q90": np.quantile(Dg[:, j], 0.9),
                                  "spearman_delta_vs_delta_score": rho[j]})
            if m in ("cos", "ms_mean", "ccos", "dot"):
                for g in (REL, NON):
                    k = grp == g
                    lv = {"control": M[ic[k], ci[col(m, "full")]].mean(0), f"attack_{r}": M[ia[k], ci[col(m, r)]].mean(0)}
                    for j, l in enumerate(labels):
                        levels.append({"metric": m, "relevance_group": g, "head_label": l, "layer": layer_of[j],
                                       "series": f"attack_{r}", "mean": lv[f"attack_{r}"][j]})
                        if r == "full":
                            levels.append({"metric": m, "relevance_group": g, "head_label": l, "layer": layer_of[j],
                                           "series": "control", "mean": lv["control"][j]})
    B = pd.DataFrame(rowsB)
    B.to_csv(st / "B_paired_delta_per_head.csv", index=False)
    BL = B.groupby(["metric", "region", "relevance_group", "layer"]).agg(
        mean_delta=("mean_delta", "mean"), mean_abs_delta=("mean_delta", lambda x: x.abs().mean()),
        mean_rho=("spearman_delta_vs_delta_score", "mean"),
        max_abs_rho=("spearman_delta_vs_delta_score", lambda x: x.abs().max())).reset_index()
    BL.to_csv(st / "B_paired_delta_layer.csv", index=False)
    LV = pd.DataFrame(levels)
    LV.to_csv(st / "B_region_levels_per_head.csv", index=False)

    # ---- plots ----------------------------------------------------------------------------------------
    X = np.arange(144)
    W = 28
    for m in MAIN:                                                    # 1. separation across heads, overlay + difference
        t = A[A.metric == m].set_index("head_label").loc[labels]
        fig, axes = plt.subplots(2, 1, figsize=(W, 8.4), gridspec_kw={"height_ratios": [1.3, 1]})
        axes[0].plot(X, t.mean_rel, color=P5.BLUE, lw=1.6, marker="o", ms=3, label=f"relevant (qrel 2/3), n={int(isrel.sum())}")
        axes[0].plot(X, t.mean_nonrel, color=P5.ORANGE, lw=1.6, marker="s", ms=3, label=f"non-relevant (qrel 0), n={int((~isrel).sum())}")
        P21.head_axis(axes[0], labels, canon)
        axes[0].set_ylabel(NAMES[m]); axes[0].legend(fontsize=9, loc="lower left")
        axes[0].set_title(f"Clean inputs: {NAMES[m]} per head (group means)", loc="left", fontsize=10, pad=16)
        axes[1].bar(X, t.cohen_d, color=[P5.BLUE if v > 0 else P5.ORANGE for v in t.cohen_d])
        axes[1].axhline(0, color=P5.INK, lw=1)
        P21.head_axis(axes[1], labels, canon)
        axes[1].set_ylabel("Cohen d (relevant − non-relevant)"); axes[1].set_ylim(-1.8, 1.8)
        axes[1].set_title("separation (blue = higher for relevant)", loc="left", fontsize=10, pad=16)
        fig.tight_layout(); P5._save(fig, pdir / f"A_sep_{m}.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))                  # 2. layer summaries
    cols = {"cos": P5.INK, "ccos": P5.BLUE, "dot": P5.AQUA, "qnorm": "#9b59b6", "dnorm": "#8c6d31",
            "ms_mean": P5.ORANGE, "ms_median": "#d62728", "ms_top3": "#e7a33e"}
    for m in MAIN:
        v = AL[AL.metric == m]
        ls = "-" if m == "cos" else "--" if m in ("qnorm", "dnorm") else "-"
        axes[0].plot(v.layer, v.mean_abs_d, color=cols[m], lw=2.6 if m == "cos" else 1.6, ls=ls, marker="o", ms=3, label=NAMES[m])
        axes[1].plot(v.layer, v.max_auroc_cv, color=cols[m], lw=2.6 if m == "cos" else 1.6, ls=ls, marker="o", ms=3)
    axes[0].set_ylabel("mean |Cohen d| over the layer's 12 heads"); axes[1].set_ylabel("best CV-oriented AUROC in layer")
    axes[1].axhline(0.5, color=P5.INK2, lw=0.8, ls="-.")
    for ax in axes:
        ax.set_xlabel("encoder layer"); ax.set_xticks(range(12))
    axes[0].legend(fontsize=8)
    fig.suptitle("Genuine relevance separation by layer (clean qrel 2/3 vs qrel 0; thick black = baseline cosine)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); P5._save(fig, pdir / "A_layer_summary.png")

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))                     # 3. ranked top heads
    for ax, m in zip(axes.ravel(), MAIN):
        t = A[A.metric == m].sort_values("abs_d", ascending=False).head(12)[::-1]
        ax.barh(t.head_label, t.auroc_cv_oriented, color=[P5.BLUE if d > 0 else P5.ORANGE for d in t.cohen_d])
        ax.axvline(0.5, color=P5.INK2, lw=0.8, ls="-.")
        ax.set_xlim(0.4, 1.0)
        cb = CVB.set_index("metric").loc[m, "cv_best_head_auroc"]
        ax.set_title(f"{NAMES[m]}\nCV best-head AUROC {cb:.3f}", loc="left", fontsize=9)
        ax.set_xlabel("CV-oriented AUROC (top 12 heads by |d|)")
    fig.suptitle("Top heads per metric (blue = higher for relevant, orange = higher for non-relevant)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); P5._save(fig, pdir / "A_top_heads.png")

    for m in ("cos", "ccos", "dot", "ms_mean", "ms_top3"):             # 4. paired deltas
        fig, axes = plt.subplots(2, 1, figsize=(W, 8.4))
        for ax, g in zip(axes, (REL, NON)):
            t = B[(B.metric == m) & (B.region == "full") & (B.relevance_group == g)].set_index("head_label").loc[labels]
            ax.bar(X, t.mean_delta, color=[P5.ORANGE if v > 0 else P5.BLUE for v in t.mean_delta])
            ax.errorbar(X, t.mean_delta, yerr=t.se_delta_query_cluster, fmt="none", ecolor=P5.INK2, lw=0.6)
            ax.axhline(0, color=P5.INK, lw=1)
            P21.head_axis(ax, labels, canon)
            ax.set_ylabel(f"Δ {m} (attack − own control)")
            ax.set_title(f"{P.GROUP_LABEL[g]}: successful attacks (n={int(t.n_instances.iloc[0])}, {int(t.n_docs.iloc[0])} docs); "
                         f"full document; ±1 SE query clusters", loc="left", fontsize=10, pad=16)
        fig.tight_layout(); P5._save(fig, pdir / f"B_delta_{m}.png")
    fig, axes = plt.subplots(2, 4, figsize=(20, 7.5))                   # 4b. corr with delta score (12x12 grid)
    for c_, m in enumerate(("cos", "ccos", "dot", "ms_mean")):
        for r_, g in enumerate((REL, NON)):
            t = B[(B.metric == m) & (B.region == "full") & (B.relevance_group == g)].set_index("head_label").loc[labels]
            im_ = axes[r_, c_].imshow(t.spearman_delta_vs_delta_score.values.reshape(12, 12), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            axes[r_, c_].set_title(f"{m}, {g}: ρ(Δmetric, Δscore)", fontsize=9, loc="left")
            axes[r_, c_].set_xlabel("head"); axes[r_, c_].set_ylabel("layer")
            axes[r_, c_].set_xticks(range(12)); axes[r_, c_].set_yticks(range(12))
    fig.colorbar(im_, ax=axes, shrink=0.6)
    P5._save(fig, pdir / "B_corr_delta_score.png")

    for m in ("cos", "ms_mean"):                                         # 5. regions
        fig, axes = plt.subplots(2, 1, figsize=(W, 8.4))
        for ax, g in zip(axes, (REL, NON)):
            for series, c_, mk in (("control", P5.BLUE, "o"), ("attack_orig", P5.AQUA, "D"), ("attack_full", P5.ORANGE, "s"),
                                   ("attack_ins", "#d62728", "^")):
                t = LV[(LV.metric == m) & (LV.relevance_group == g) & (LV.series == series)].set_index("head_label").loc[labels]
                ax.plot(X, t["mean"], color=c_, lw=1.4, marker=mk, ms=3, label=series.replace("_", ": ").replace("attack: ins", "attack: inserted tokens only"))
            P21.head_axis(ax, labels, canon)
            ax.set_ylabel(NAMES[m]); ax.legend(fontsize=8, loc="lower left")
            ax.set_title(f"{P.GROUP_LABEL[g]}, successful attacks: control vs attacked original / full / inserted-only document",
                         loc="left", fontsize=10, pad=16)
        fig.tight_layout(); P5._save(fig, pdir / f"B_regions_{m}.png")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))                     # 5b. region layer summary
    for c_, m in enumerate(("cos", "ms_mean")):
        for r_, g in enumerate((REL, NON)):
            ax = axes[r_, c_]
            for r, c2 in (("orig", P5.AQUA), ("full", P5.ORANGE), ("ins", "#d62728")):
                v = BL[(BL.metric == m) & (BL.region == r) & (BL.relevance_group == g)]
                ax.plot(v.layer, v.mean_delta, color=c2, marker="o", ms=3, label=f"{r} − control")
            ax.axhline(0, color=P5.INK, lw=1)
            ax.set_title(f"{NAMES[m]}, {g}: mean Δ per layer", loc="left", fontsize=9)
            ax.set_xticks(range(12)); ax.set_xlabel("encoder layer"); ax.legend(fontsize=8)
    fig.tight_layout(); P5._save(fig, pdir / "B_regions_layer_summary.png")

    # ---- comparison table ---------------------------------------------------------------------------------
    rows = []
    for m in MAIN:
        a = A[A.metric == m].sort_values("abs_d", ascending=False)
        top = a.head(5)
        best = top.iloc[0]
        bf = B[(B.metric == m) & (B.region == "full")]
        bt = bf[bf.head_label.isin(top.head_label)]
        gap = top.set_index("head_label").difference
        rel_d = bt.assign(gap=bt.head_label.map(gap)).assign(frac=lambda x: x.mean_delta / x.gap)
        rows.append({"metric": m, "best |d| (head)": f"{best.cohen_d:+.2f} ({best.head_label})",
                     "AUROC in-sample (best head, oriented)": round(max(best.auroc_raw, 1 - best.auroc_raw), 3),
                     "CV best-head AUROC": round(float(CVB.set_index("metric").loc[m, "cv_best_head_auroc"]), 3),
                     "top-5 heads": ",".join(top.head_label), "layer with max mean|d|":
                         int(AL[AL.metric == m].sort_values("mean_abs_d").iloc[-1].layer),
                     "paired Δ at top-5 heads, as fraction of genuine gap (rel / nonrel)":
                         f"{rel_d[rel_d.relevance_group == REL].frac.mean():+.2f} / {rel_d[rel_d.relevance_group == NON].frac.mean():+.2f}",
                     "max |ρ(Δmetric, Δscore)| any head (rel / nonrel)":
                         f"{bf[bf.relevance_group == REL].spearman_delta_vs_delta_score.abs().max():.2f} / "
                         f"{bf[bf.relevance_group == NON].spearman_delta_vs_delta_score.abs().max():.2f}"})
    CT = pd.DataFrame(rows)
    CT.to_csv(st / "comparison_table.csv", index=False)
    try:
        md = CT.to_markdown(index=False)
    except ImportError:                                               # tabulate not installed
        md = "| " + " | ".join(CT.columns) + " |\n|" + "---|" * len(CT.columns) + "\n" + \
             "\n".join("| " + " | ".join(map(str, r)) + " |" for r in CT.itertuples(index=False))
    (st / "comparison_table.md").write_text(md)
    summary = {"created": now(), "n_sequences": n, "n_clean_docs": int(cl.sum()), "n_rel_docs": int(isrel.sum()),
               "n_successful_attacks": int(len(succ)), "n_successful_by_group": succ.relevance_group.value_counts().to_dict(),
               "validation": val, "cv_best_head": CVB.to_dict("records"), "seconds": time.time() - t0,
               "centering": "mu_f = mean of all query-text + document token vectors of CLEAN sequences of queries not in fold f "
                            "(label-free, per head); ccos = cos(qbar - mu_f, dbar - mu_f)",
               "maxsim_top_k": "mean of the min(3, n_query_tokens) largest per-query-token maxima"}
    (st / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_status(st, {"status": "success", "finished": now()})
    print(CT.to_string(index=False))
    print(CVB.to_string(index=False))
    print(f"[23] -> {st} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
