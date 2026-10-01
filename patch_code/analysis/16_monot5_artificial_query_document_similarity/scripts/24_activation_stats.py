"""
scripts/24_activation_stats.py
===============================
Stage 24 — exploratory SCREEN of activation magnitude / structure statistics on the stage-22
token sample only (no forward pass, no full population, no learned classifier). Statistic and
region definitions: exp16lib/activation_stats.py.

Validation first (stops on failure): stage-22 counts and shape; mask integrity (query/doc
disjoint, no template/pad/masked control slots in any region, attack original region excludes
inserted tokens, inserted region = exactly the injected string: identical token ids for every
instance of an attack, attack word occurs `repetitions` times, nothing else except ':');
decoded mask inspection file; statistics re-computed by an independent direct NumPy
implementation on random (sequence, region, head) cells; PCA input = 2304 dims and equal to
a direct mean of the stored tokens.

A. Genuine relevance — CLEAN sequences (one per document), qrel 2/3 vs qrel 0, per statistic x
   region (query, doc) x head: means, medians, difference, Cohen d (+ = larger for relevant),
   raw AUROC (P(relevant > non-relevant); > 0.5 = larger for relevant) and direction-free
   separation max(AUROC, 1 - AUROC).
B. Successful attacks (delta_score > 0) vs their own padded control, relevant / non-relevant
   base documents separately: delta = stat(attack, region) - stat(control, region'), for
   query->query, orig->doc, doc->doc (control orig == doc). Instance-level and equal-document-
   weight summaries, paired d_z, Spearman (primary) and Pearson with delta_score.
E. Inserted tokens (successful attacks): stat(ins) vs stat(orig) of the same attacked sequence,
   and vs a size-matched baseline (mean over 16 random subsets of the original-document tokens
   with the same token count, seed 42) because top_share / entropy / maxabs / eff_dim depend on
   the region size.
PCA (L9-L11 "late-layer pooled document head representation", 3 x 12 x 64 = 2304 dims; NOT the
   residual stream). Fitted on CLEAN documents only (raw, and query-centred by each qid's clean
   mean); padded controls and successful attacks projected into the same bases.

Outputs: 24_activation_stats/{activation_statistics.npz, activation_statistics.csv.gz,
  sequences_meta.csv, late_doc_reps.npz, validation.json, mask_inspection.txt,
  relevance_head_summary.csv, attack_paired_summary.csv, inserted_region_summary.csv,
  pca_clean.csv, pca_attack_pairs.csv, pca_movement_summary.csv, pca_metadata.json,
  summary_table.{csv,md}, summary.json, plots/*.png}
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
from scipy.stats import pearsonr, spearmanr

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import activation_stats as AS  # noqa: E402
from exp16lib import paired as P  # noqa: E402
from exp16lib.anomaly import auroc  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_tokenizer, now, stage_argparser, write_status  # noqa: E402
from exp16lib.token_sample import TokenSample  # noqa: E402

_spec = importlib.util.spec_from_file_location("p21", EXP_DIR / "scripts" / "21_plot_paired.py")
P21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P21)
P5 = P21.P5

STAGE = "24_activation_stats"
REL, NON = "relevant", "nonrelevant"
NAMES = {"l2": "mean token L2 norm", "mean": "mean activation", "var": "mean per-token variance over dims",
         "maxabs": "max |activation|", "eff_dim": "effective dimensionality (participation ratio)",
         "top_share": "top-token energy share", "entropy": "normalised token-energy entropy"}
RI = {r: i for i, r in enumerate(AS.REGIONS)}
SI = {s: i for i, s in enumerate(AS.STATS)}
PAIRED_REGIONS = [("query", "query"), ("orig", "doc"), ("doc", "doc")]      # (attack region, control region)
N_SIZE_DRAWS = 16
N_ARROWS = 100
N_PC_KEEP = 10


def _heatmap(ax, M, title, vmax, cmap="RdBu_r"):
    im = ax.imshow(M.reshape(12, 12), cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(12)); ax.set_yticks(range(12)); ax.grid(False)
    return im


def _vmax(*arrs, pct=100):
    v = np.nanpercentile(np.abs(np.concatenate([np.ravel(a) for a in arrs])), pct)
    return float(v) if np.isfinite(v) and v > 0 else 1.0


def _col_spearman(X, y):
    """Spearman of each column of X [n, h] with y (NaN rows of a column dropped)."""
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j])
        if ok.sum() > 2 and np.nanstd(X[ok, j]) > 0:
            out[j] = spearmanr(X[ok, j], y[ok])[0]
    return out


def _col_pearson(X, y):
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j])
        if ok.sum() > 2 and np.nanstd(X[ok, j]) > 0:
            out[j] = pearsonr(X[ok, j], y[ok])[0]
    return out


def _pca(X):
    """Centred PCA via SVD -> (mean, components [k, d], explained-variance ratio [k]); sign: largest |loading| > 0."""
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(1)])
    Vt = Vt * sgn[:, None]
    return mu, Vt, s ** 2 / (s ** 2).sum()


def main():
    args = stage_argparser("Exp 16 stage 24: activation-statistics screen on the token sample").parse_args()
    cfg = load_config(args.config)
    P.configure("all_layers")                  # 144 heads: plot helpers (head_axis) must use layers 0-11
    out = output_dir(cfg)
    src = out / "22_token_sample"
    if not is_already_successful(src, ["heads.npy", "tokens.npz", "sequences.csv"]):
        raise FileNotFoundError(f"run stage 22 first ({src})")
    st = out / STAGE
    if is_already_successful(st, ["summary.json"]) and not args.force:
        print(f"[24] {st} already complete (use --force to recompute)")
        return
    pdir = st / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    write_status(st, {"status": "running", "started": now()})
    t0 = time.time()
    ts = TokenSample(src)
    meta = ts.meta.copy()
    n = len(meta)
    labels = [f"L{L}H{h}" for L in range(12) for h in range(12)]
    layer_of = np.repeat(np.arange(12), 12)
    canon = {h.label for h in encoder_heads()}
    tok = load_tokenizer(cfg)
    T = ts.tok
    val = {}

    # ---- validation 1: counts and shape ----------------------------------------------------------------
    cl_meta = meta[meta.kind == "clean"]
    val["counts"] = {"n_sequences": n, "by_kind": meta.kind.value_counts().to_dict(),
                     "n_pairs": int(meta.pair_id.nunique()), "n_queries": int(meta.qid.nunique()),
                     "clean_by_group": cl_meta.relevance_group.value_counts().to_dict(),
                     "clean_qrel_grades": sorted(int(g) for g in cl_meta.qrel_grade.unique()),
                     "n_successful_attacks": int((meta[meta.kind == "attack"].successful == True).sum()),  # noqa: E712
                     "heads_shape": list(ts.heads.shape), "heads_dtype": str(ts.heads.dtype)}
    c = val["counts"]
    ok_counts = (n == 6325 and c["n_pairs"] == 253 and c["n_queries"] == 43 and c["clean_by_group"] == {REL: 126, NON: 127}
                 and c["by_kind"] == {"control": 3036, "attack": 3036, "clean": 253} and c["n_successful_attacks"] == 1026
                 and tuple(ts.heads.shape[1:]) == (12, 12, 64) and ts.heads.shape[0] == int(meta.n_tokens.sum())
                 and 1 not in c["clean_qrel_grades"] and (meta.seq_id.values == np.arange(n)).all()
                 and (meta.index.values == np.arange(n)).all())
    atk_all = meta[meta.kind == "attack"]
    ok_counts &= bool(((atk_all.delta_score > 0) == atk_all.successful.astype(bool)).all())
    val["counts_ok"] = bool(ok_counts)
    if not ok_counts:
        (st / "validation.json").write_text(json.dumps(val, indent=2, default=str))
        raise SystemExit(f"STOP: stage-22 counts/shape differ from the expected sample: {c}")

    # ---- validation 2: masks ------------------------------------------------------------------------------
    qm, dm, im, am = (T[k].astype(bool) for k in ("query_mask", "doc_mask", "inserted_mask", "attention_mask"))
    ids = T["input_ids"]
    kind_tok = np.repeat(meta.kind.values, meta.n_tokens.values)
    pad = tok.pad_token_id
    template = set(tok.convert_tokens_to_ids(["▁", "Query", ":", "▁Document", "▁Relevan", "t", "</s>"]))
    ctl_slots = im & (kind_tok == "control")
    ins_real = im & dm & (kind_tok == "attack")
    val["query_doc_disjoint"] = bool(not (qm & dm).any())
    val["regions_attention_1"] = bool(am[qm | dm].all())
    val["pad_in_regions"] = int(((ids == pad) & (qm | dm)).sum())
    val["control_inserted_slots_in_any_region"] = int((ctl_slots & (qm | dm)).sum())
    val["control_inserted_slots_are_masked_pads"] = bool((ids[ctl_slots] == pad).all() and not am[ctl_slots].any())
    val["n_control_masked_slots"] = int(ctl_slots.sum())
    val["clean_has_inserted"] = int((im & (kind_tok == "clean")).sum())
    val["attack_inserted_outside_doc"] = int((im & ~dm & (kind_tok == "attack")).sum())
    val["attack_inserted_is_pad_or_unattended"] = int(((ids == pad) | ~am)[ins_real].sum())
    val["non_template_tokens_outside_regions"] = int((~np.isin(ids[~(qm | dm)], list(template | {pad}))).sum())
    # attack orig region never contains an inserted token (computed through the stage-24 region code path)
    bad_orig, bad_word, bad_extra, seqs_by_attack = 0, 0, 0, {}
    colon = tok.convert_tokens_to_ids(":")
    for i in atk_all.index:
        s = ts[int(i)]
        reg = AS.region_masks(s["query_mask"], s["doc_mask"], s["inserted_mask"], "attack")
        bad_orig += int((reg["orig"] & s["inserted_mask"]).sum()) + int((reg["orig"] | reg["ins"]).sum() != s["doc_mask"].sum())
        ins_ids = s["input_ids"][reg["ins"]].tolist()
        seqs_by_attack.setdefault(meta.attack_name[i], set()).add(tuple(ins_ids))
        pieces = tok.convert_ids_to_tokens(ins_ids)
        word = "▁" + meta.attack_token[i]
        bad_word += int(sum(p == word for p in pieces) != int(meta.repetitions[i]))
        bad_extra += int(any(p != word and t_ != colon for p, t_ in zip(pieces, ins_ids)))
    val["attack_orig_contains_inserted_or_partition_broken"] = bad_orig
    val["attack_inserted_word_count_ne_repetitions"] = bad_word
    val["attack_inserted_has_other_tokens_than_word_or_colon"] = bad_extra
    val["attack_inserted_token_sequence_identical_within_attack"] = {k: len(v) == 1 for k, v in seqs_by_attack.items()}
    val["attack_inserted_strings"] = {k: tok.decode(list(next(iter(v)))) for k, v in seqs_by_attack.items()}
    ok_mask = (val["query_doc_disjoint"] and val["regions_attention_1"] and val["pad_in_regions"] == 0
               and val["control_inserted_slots_in_any_region"] == 0 and val["control_inserted_slots_are_masked_pads"]
               and val["clean_has_inserted"] == 0 and val["attack_inserted_outside_doc"] == 0
               and val["attack_inserted_is_pad_or_unattended"] == 0 and val["non_template_tokens_outside_regions"] == 0
               and bad_orig == 0 and bad_word == 0 and bad_extra == 0
               and all(val["attack_inserted_token_sequence_identical_within_attack"].values()))
    val["masks_ok"] = bool(ok_mask)
    if not ok_mask:
        (st / "validation.json").write_text(json.dumps(val, indent=2, default=str))
        raise SystemExit(f"STOP: mask validation failed: {val}")
    lines = []                                                              # manual inspection file
    picks = [int(cl_meta.index[cl_meta.relevance_group == REL][0]), int(cl_meta.index[cl_meta.relevance_group == NON][0])]
    for name in ("relevant_start_1", "false_start_5", "relevant_end_5", "true_random_1", "bar_random_5"):
        ia_ = int(atk_all.index[(atk_all.attack_name == name)][0])
        ic_ = int(meta.index[(meta.kind == "control") & (meta.attack_name == name) & (meta.pair_id == meta.pair_id[ia_])][0])
        picks += [ic_, ia_]
    for i in picks:
        s = ts[i]
        k = s["meta"]["kind"]
        reg = AS.region_masks(s["query_mask"], s["doc_mask"], s["inserted_mask"], k)
        dec = lambda m: " ".join(tok.convert_ids_to_tokens(s["input_ids"][m].tolist()))  # noqa: E731
        slots = s["inserted_mask"] & ~s["doc_mask"]
        lines += [f"=== seq {i} kind={k} group={s['meta']['relevance_group']} pair={s['meta']['pair_id']} "
                  f"attack={s['meta']['attack_name']} successful={s['meta']['successful']}",
                  f"QUERY ({s['query_mask'].sum()}): {dec(s['query_mask'])}",
                  f"DOC   ({reg['full'].sum()}): {dec(reg['full'])}",
                  f"ORIG  ({reg['orig'].sum()}): {dec(reg['orig'])}",
                  f"INS   ({0 if reg['ins'] is None else reg['ins'].sum()}): {'NA' if reg['ins'] is None else dec(reg['ins'])}",
                  f"MASKED CONTROL SLOTS ({slots.sum()}, attention={s['attention_mask'][slots].tolist()}): {dec(slots)}",
                  f"OUTSIDE REGIONS: {dec(~(s['query_mask'] | s['doc_mask']))}", ""]
    (st / "mask_inspection.txt").write_text("\n".join(lines))
    print(f"[24] validation passed ({time.time() - t0:.0f}s)", flush=True)

    # ---- statistics ----------------------------------------------------------------------------------------
    S = np.full((n, len(AS.REGIONS), len(AS.STATS), 144), np.nan, dtype=np.float32)
    NT = np.zeros((n, len(AS.REGIONS)), dtype=np.int32)
    REP = np.zeros((n, 2304), dtype=np.float32)                            # pooled L9-L11, full document
    REPO = np.zeros((n, 2304), dtype=np.float32)                           # pooled L9-L11, original-document tokens
    SM = np.full((n, len(AS.STATS), 144), np.nan, dtype=np.float32)        # size-matched original-document baseline
    succ_mask = (meta.kind == "attack").values & (meta.successful == True).fillna(False).values  # noqa: E712
    rng = np.random.default_rng(42)
    for i in range(n):
        s = ts[i]
        k = s["meta"]["kind"]
        H = np.asarray(s["heads"], dtype=np.float32)
        H144 = H.reshape(len(H), 144, 64)
        o = AS.sequence_stats(H144, s["query_mask"], s["doc_mask"], s["inserted_mask"], k)
        for r in AS.REGIONS:
            NT[i, RI[r]] = o[f"n_{r}"]
            if o[r] is not None:
                S[i, RI[r]] = o[r]
        reg = AS.region_masks(s["query_mask"], s["doc_mask"], s["inserted_mask"], k)
        REP[i] = AS.pooled_late_doc(H, reg["full"])
        REPO[i] = AS.pooled_late_doc(H, reg["orig"])
        if succ_mask[i]:
            SM[i] = AS.size_matched_stats(H144[reg["orig"]], o["n_ins"], N_SIZE_DRAWS, rng)
        if i % 1000 == 0:
            print(f"[24] stats {i}/{n} ({time.time() - t0:.0f}s)", flush=True)

    # ---- validation 3: direct NumPy recomputation + PCA input -----------------------------------------------
    chk = np.random.default_rng(7)
    worst = 0.0
    n_chk = 0
    for i in chk.choice(n, 40, replace=False):
        a, b = int(meta.token_offset[i]), int(meta.token_offset[i] + meta.n_tokens[i])
        q_, d_, in_ = qm[a:b], dm[a:b], im[a:b]
        direct = {"query": q_, "doc": d_}
        if meta.kind[i] == "attack":
            direct.update({"orig": d_ & ~in_, "ins": d_ & in_})
        else:
            direct["orig"] = d_
        for r, m in direct.items():
            L, h = int(chk.integers(12)), int(chk.integers(12))
            X = ts.heads[a:b][m, L, h].astype(np.float64)                   # [n, 64]
            e = (X ** 2).mean(axis=0)
            Et = np.array([x @ x for x in X])
            p = Et / Et.sum()
            ref = {"l2": np.mean([np.sqrt(x @ x) for x in X]), "mean": X.sum() / X.size,
                   "var": np.mean([((x - x.mean()) ** 2).mean() for x in X]), "maxabs": np.max(np.abs(X)),
                   "eff_dim": e.sum() ** 2 / (e ** 2).sum(), "top_share": p.max(),
                   "entropy": (-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(X))) if len(X) > 1 else np.nan}
            for sname, v in ref.items():
                got = float(S[i, RI[r], SI[sname], L * 12 + h])
                if np.isnan(v) and np.isnan(got):
                    continue
                worst = max(worst, abs(got - v) / max(abs(v), 1e-3))
                n_chk += 1
    val["direct_numpy_recompute_max_rel_diff"] = worst
    val["direct_numpy_recompute_n_values"] = n_chk
    i0 = int(cl_meta.index[0])
    s0 = ts[i0]
    direct_rep = np.concatenate([s0["heads"][s0["doc_mask"], L, h].astype(np.float32).mean(0)
                                 for L in (9, 10, 11) for h in range(12)])
    val["pca_input_dim"] = int(REP.shape[1])
    val["pca_input_vs_direct_max_abs_diff"] = float(np.abs(direct_rep - REP[i0]).max())
    val["control_orig_equals_doc"] = bool(np.array_equal(S[meta.kind != "attack"][:, RI["orig"]],
                                                         S[meta.kind != "attack"][:, RI["doc"]], equal_nan=True))
    val["ins_defined_only_for_attack"] = bool(np.isnan(S[meta.kind != "attack"][:, RI["ins"]]).all()
                                              and (NT[(meta.kind == "attack").values, RI["ins"]] > 0).all())
    (st / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    if worst > 1e-4 or val["pca_input_dim"] != 2304 or val["pca_input_vs_direct_max_abs_diff"] > 1e-6 \
            or not val["control_orig_equals_doc"] or not val["ins_defined_only_for_attack"]:
        raise SystemExit(f"STOP: statistic / PCA-input validation failed: {val}")
    print(f"[24] stats done, direct recompute max rel diff {worst:.2e} over {n_chk} values ({time.time() - t0:.0f}s)", flush=True)

    # ---- save per-sequence statistics ---------------------------------------------------------------------------
    keep_meta = ["seq_id", "kind", "attack_name", "pair_id", "qid", "docid", "qrel_grade", "relevance_group",
                 "attack_token", "attack_position", "repetitions", "score", "delta_score", "successful", "n_tokens"]
    SMETA = meta[keep_meta].copy()
    for r in AS.REGIONS:
        SMETA[f"n_tokens_{r}"] = NT[:, RI[r]]
    SMETA.to_csv(st / "sequences_meta.csv", index=False)
    np.savez_compressed(st / "activation_statistics.npz", S=S, size_matched_orig=SM, n_tokens=NT, regions=np.array(AS.REGIONS),
                        stats=np.array(AS.STATS), heads=np.array(labels), seq_id=meta.seq_id.values)
    np.savez_compressed(st / "late_doc_reps.npz", full=REP, orig=REPO, seq_id=meta.seq_id.values,
                        layout=np.array("layers 9,10,11 x heads 0-11 x 64 dims, mean over region tokens, fp32"))
    long = []                                                   # one row per (sequence, defined region, head)
    for r in AS.REGIONS:
        rows = np.where(meta.kind.values == "attack")[0] if r in ("orig", "ins") else np.arange(n)
        blk = S[rows, RI[r]]                                    # [m, stats, 144]
        df = pd.DataFrame({"seq_id": np.repeat(meta.seq_id.values[rows], 144), "region": r,
                           "head_label": np.tile(labels, len(rows)), "n_tokens_region": np.repeat(NT[rows, RI[r]], 144)})
        for sname in AS.STATS:
            df[sname] = blk[:, SI[sname]].reshape(-1)
        long.append(df)
    pd.concat(long).to_csv(st / "activation_statistics.csv.gz", index=False, float_format="%.6g", compression="gzip")
    print(f"[24] saved per-sequence statistics ({time.time() - t0:.0f}s)", flush=True)

    # ---- A. genuine relevance (clean) --------------------------------------------------------------------------
    cl = (meta.kind == "clean").values
    cm = meta[cl].reset_index(drop=True)
    isrel = (cm.relevance_group == REL).values
    rowsA = []
    for r in ("query", "doc"):
        for sname in AS.STATS:
            V = S[cl, RI[r], SI[sname]].astype(np.float64)
            for j, l in enumerate(labels):
                a, b = V[isrel, j], V[~isrel, j]
                a, b = a[~np.isnan(a)], b[~np.isnan(b)]
                au = auroc(a, b) if len(a) and len(b) else np.nan
                rowsA.append({"statistic": sname, "region": r, "head_label": l, "layer": layer_of[j], "head": j % 12,
                              "n_rel": len(a), "n_nonrel": len(b), "mean_rel": a.mean(), "mean_nonrel": b.mean(),
                              "median_rel": np.median(a), "median_nonrel": np.median(b), "difference": a.mean() - b.mean(),
                              "cohen_d": AS.cohen_d(a, b), "auroc_raw": au, "auroc_separation": max(au, 1 - au),
                              "larger_for": REL if a.mean() > b.mean() else NON})
    A = pd.DataFrame(rowsA)
    A["abs_d"] = A.cohen_d.abs()
    A["rank_abs_d_in_stat_region"] = A.groupby(["statistic", "region"]).abs_d.rank(ascending=False, method="first").astype(int)
    A.to_csv(st / "relevance_head_summary.csv", index=False)
    # confound: document / query length (token-count-sensitive statistics)
    confound = {}
    for r in ("query", "doc"):
        ntok = NT[cl, RI[r]].astype(float)
        confound[f"n_tokens_{r}"] = {"mean_rel": ntok[isrel].mean(), "mean_nonrel": ntok[~isrel].mean(),
                                     "cohen_d": AS.cohen_d(ntok[isrel], ntok[~isrel]), "auroc_raw": auroc(ntok[isrel], ntok[~isrel])}
        for sname in AS.STATS:
            rho = _col_spearman(S[cl, RI[r], SI[sname]].astype(np.float64), ntok)
            confound[f"n_tokens_{r}"][f"median_abs_spearman_{sname}_vs_ntokens"] = float(np.nanmedian(np.abs(rho)))

    # ---- B. successful attacks vs own control -------------------------------------------------------------------
    atk = meta[meta.kind == "attack"]
    ctl = meta[meta.kind == "control"].set_index(["attack_name", "pair_id"])
    succ = atk[atk.successful == True]  # noqa: E712
    ia = succ.index.values
    ic = ctl.loc[list(zip(succ.attack_name, succ.pair_id)), "seq_id"].values
    assert (meta.loc[ic, "pair_id"].values == succ.pair_id.values).all() and (meta.loc[ic, "kind"] == "control").all()
    assert np.allclose(meta.loc[ia, "score"].values - meta.loc[ic, "score"].values, succ.delta_score.values, rtol=0, atol=1e-6)
    grp = succ.relevance_group.values
    dsc = succ.delta_score.values
    rowsB = []
    for ra, rc in PAIRED_REGIONS:
        for sname in AS.STATS:
            D = S[ia, RI[ra], SI[sname]].astype(np.float64) - S[ic, RI[rc], SI[sname]].astype(np.float64)
            for g in (REL, NON):
                k = grp == g
                Dg, sub = D[k], succ[k]
                docm = pd.DataFrame(Dg).groupby(sub.pair_id.values).mean()
                docs = pd.Series(dsc[k]).groupby(sub.pair_id.values).mean().loc[docm.index].values
                rho, pr = _col_spearman(Dg, dsc[k]), _col_pearson(Dg, dsc[k])
                rho_doc = _col_spearman(docm.values, docs)
                ctl_mean = S[ic[k], RI[rc], SI[sname]].mean(0)
                att_mean = S[ia[k], RI[ra], SI[sname]].mean(0)
                for j, l in enumerate(labels):
                    x = Dg[:, j][~np.isnan(Dg[:, j])]
                    xd = docm.values[:, j][~np.isnan(docm.values[:, j])]
                    rowsB.append({"statistic": sname, "region": ra, "control_region": rc, "relevance_group": g,
                                  "head_label": l, "layer": layer_of[j], "head": j % 12, "n_instances": len(x),
                                  "n_docs": len(xd), "control_mean": ctl_mean[j], "attack_mean": att_mean[j],
                                  "mean_delta": x.mean(), "median_delta": np.median(x), "frac_delta_positive": (x > 0).mean(),
                                  "paired_dz": AS.paired_dz(x), "relative_mean_delta": x.mean() / abs(ctl_mean[j]),
                                  "spearman_delta_vs_delta_score": rho[j], "pearson_delta_vs_delta_score": pr[j],
                                  "mean_delta_doc_weighted": xd.mean(), "median_delta_doc_weighted": np.median(xd),
                                  "frac_docs_delta_positive": (xd > 0).mean(), "paired_dz_doc_weighted": AS.paired_dz(xd),
                                  "spearman_doc_level": rho_doc[j]})
    B = pd.DataFrame(rowsB)
    B["abs_dz"] = B.paired_dz.abs()
    B.to_csv(st / "attack_paired_summary.csv", index=False)
    print(f"[24] A/B summaries done ({time.time() - t0:.0f}s)", flush=True)

    # ---- E. inserted tokens (successful attacks) ---------------------------------------------------------------
    rowsE = []
    for sname in AS.STATS:
        ins = S[ia, RI["ins"], SI[sname]].astype(np.float64)
        org = S[ia, RI["orig"], SI[sname]].astype(np.float64)
        full = S[ia, RI["doc"], SI[sname]].astype(np.float64)
        ctl_doc = S[ic, RI["doc"], SI[sname]].astype(np.float64)
        smt = SM[ia, SI[sname]].astype(np.float64)
        for g in (REL, NON):
            k = grp == g
            pid = succ.pair_id.values[k]
            d_io, d_is = ins[k] - org[k], ins[k] - smt[k]
            doc_io = pd.DataFrame(d_io).groupby(pid).mean().values
            with np.errstate(invalid="ignore"):
                for j, l in enumerate(labels):
                    a, b2 = d_io[:, j], d_is[:, j]
                    a, b2 = a[~np.isnan(a)], b2[~np.isnan(b2)]
                    xd = doc_io[:, j][~np.isnan(doc_io[:, j])]
                    rowsE.append({"statistic": sname, "relevance_group": g, "head_label": l, "layer": layer_of[j],
                                  "head": j % 12, "n_instances": int(k.sum()), "n_valid_ins": len(a),
                                  "mean_ins": np.nanmean(ins[k, j]), "mean_orig": np.nanmean(org[k, j]),
                                  "mean_full_doc": np.nanmean(full[k, j]), "mean_control_doc": np.nanmean(ctl_doc[k, j]),
                                  "mean_orig_size_matched": np.nanmean(smt[k, j]),
                                  "median_ins": np.nanmedian(ins[k, j]), "median_orig": np.nanmedian(org[k, j]),
                                  "mean_ins_minus_orig": a.mean() if len(a) else np.nan,
                                  "frac_ins_gt_orig": (a > 0).mean() if len(a) else np.nan,
                                  "dz_ins_vs_orig": AS.paired_dz(a), "ratio_ins_over_orig": np.nanmean(ins[k, j]) / np.nanmean(org[k, j]),
                                  "mean_ins_minus_orig_doc_weighted": xd.mean() if len(xd) else np.nan,
                                  "mean_ins_minus_size_matched": b2.mean() if len(b2) else np.nan,
                                  "frac_ins_gt_size_matched": (b2 > 0).mean() if len(b2) else np.nan,
                                  "dz_ins_vs_size_matched": AS.paired_dz(b2)})
    E = pd.DataFrame(rowsE)
    E.to_csv(st / "inserted_region_summary.csv", index=False)

    # ---- PCA (L9-L11 late-layer pooled document head representation) --------------------------------------------
    Xc = REP[cl].astype(np.float64)
    qids = cm.qid.values
    qcenter = {q: Xc[qids == q].mean(0) for q in np.unique(qids)}
    Xcc = Xc - np.stack([qcenter[q] for q in qids])
    bases = {}
    for name, X in (("raw", Xc), ("query_centered", Xcc)):
        mu, Vt, evr = _pca(X)
        bases[name] = {"mu": mu, "V": Vt[:N_PC_KEEP], "evr": evr}

    def project(name, R, q):
        Z = R.astype(np.float64) - (np.stack([qcenter[x] for x in q]) if name == "query_centered" else 0.0)
        return (Z - bases[name]["mu"]) @ bases[name]["V"].T, Z

    pcs = pd.DataFrame({"seq_id": cm.seq_id, "qid": cm.qid, "docid": cm.docid, "pair_id": cm.pair_id,
                        "qrel_grade": cm.qrel_grade, "relevance_group": cm.relevance_group})
    pc_sep = []
    rel_dir = {}
    for name in bases:
        Zc, Zfull = project(name, Xc, qids)
        for c_ in range(N_PC_KEEP):
            pcs[f"{name}_PC{c_ + 1}"] = Zc[:, c_]
            au = auroc(Zc[isrel, c_], Zc[~isrel, c_])
            pc_sep.append({"basis": name, "pc": c_ + 1, "explained_variance_ratio": bases[name]["evr"][c_],
                           "cohen_d_rel_minus_nonrel": AS.cohen_d(Zc[isrel, c_], Zc[~isrel, c_]),
                           "auroc_raw": au, "auroc_separation": max(au, 1 - au)})
        rel_dir[name] = {"rel": Zfull[isrel].mean(0), "non": Zfull[~isrel].mean(0)}
    pcs.to_csv(st / "pca_clean.csv", index=False)
    PCS = pd.DataFrame(pc_sep)

    pairs = pd.DataFrame({"attack_seq_id": ia, "control_seq_id": ic, "attack_name": succ.attack_name.values,
                          "pair_id": succ.pair_id.values, "qid": succ.qid.values, "relevance_group": grp,
                          "delta_score": dsc})
    clean_of_pair = cm.set_index("pair_id").seq_id
    iclean = clean_of_pair.loc[succ.pair_id.values].values
    mov = []
    for variant, R in (("full_doc", REP), ("orig_doc", REPO)):
        for name in bases:
            q = succ.qid.values
            Pc, Zc_ = project(name, R[ic], q)
            Pa, Za_ = project(name, R[ia], q)
            _, Zcl_ = project(name, REP[iclean], q)
            g_ = rel_dir[name]["rel"] - rel_dir[name]["non"]
            dZ = Za_ - Zc_
            proj = dZ @ g_ / (g_ @ g_)                         # movement along the genuine relevance direction (fraction of gap)
            ctl_off = (Zc_ - Zcl_) @ g_ / (g_ @ g_)            # sanity: control vs its own clean document
            dist_gap = lambda Z: np.linalg.norm(Z - rel_dir[name]["rel"], axis=1) - np.linalg.norm(Z - rel_dir[name]["non"], axis=1)  # noqa: E731
            closer_rel = dist_gap(Za_) < dist_gap(Zc_)
            if variant == "full_doc":
                for c_ in range(2):
                    pairs[f"{name}_PC{c_ + 1}_control"] = Pc[:, c_]
                    pairs[f"{name}_PC{c_ + 1}_attack"] = Pa[:, c_]
            pairs[f"{name}_{variant}_proj_rel_direction"] = proj
            pairs[f"{name}_{variant}_moved_closer_to_rel_centroid"] = closer_rel
            for g in (REL, NON):
                k = grp == g
                docp = pd.Series(proj[k]).groupby(succ.pair_id.values[k]).mean()
                mdz = dZ[k].mean(0)
                mov.append({"representation": variant, "basis": name, "relevance_group": g, "n_instances": int(k.sum()),
                            "n_docs": len(docp), "mean_dPC1": (Pa - Pc)[k, 0].mean(), "mean_dPC2": (Pa - Pc)[k, 1].mean(),
                            "mean_control_PC1": Pc[k, 0].mean(), "mean_control_PC2": Pc[k, 1].mean(),
                            "mean_attack_PC1": Pa[k, 0].mean(), "mean_attack_PC2": Pa[k, 1].mean(),
                            "mean_proj_on_rel_direction_frac_of_gap": proj[k].mean(),
                            "median_proj_on_rel_direction_frac_of_gap": np.median(proj[k]),
                            "frac_proj_positive": (proj[k] > 0).mean(),
                            "mean_proj_doc_weighted": docp.mean(), "frac_docs_proj_positive": (docp > 0).mean(),
                            "spearman_proj_vs_delta_score": spearmanr(proj[k], dsc[k])[0],
                            "cos_mean_movement_vs_rel_direction": float(mdz @ g_ / (np.linalg.norm(mdz) * np.linalg.norm(g_))),
                            "mean_movement_norm_over_gap_norm": float(np.linalg.norm(dZ[k], axis=1).mean() / np.linalg.norm(g_)),
                            "frac_moved_closer_to_rel_centroid": closer_rel[k].mean(),
                            "sanity_mean_abs_control_vs_clean_proj": float(np.abs(ctl_off[k]).mean())})
    pairs.to_csv(st / "pca_attack_pairs.csv", index=False)
    MOV = pd.DataFrame(mov)
    MOV.to_csv(st / "pca_movement_summary.csv", index=False)
    PCS.to_csv(st / "pca_component_relevance.csv", index=False)
    pca_meta = {"representation": "late-layer pooled document head representation: mean over document tokens (doc_mask; "
                                  "attacked = original + injected tokens) of the pre-o_proj output of every head of encoder "
                                  "layers 9, 10, 11, concatenated layer-major/head-major -> 3 x 12 x 64 = 2304 dims. "
                                  "NOT a residual-stream representation.",
                "input_dim": int(REP.shape[1]), "fit_on": "CLEAN sequences only (126 relevant + 127 non-relevant documents)",
                "scaling": "centred only (no per-feature standardisation)",
                "sign_convention": "each component's largest-|loading| coordinate is positive",
                "query_centered": "rep - mean clean rep of the same qid (all clean docs of that qid, both groups)",
                "n_queries_with_both_groups": int(sum(len(set(cm.relevance_group[cm.qid == q])) == 2 for q in np.unique(qids))),
                "explained_variance_ratio": {k: [float(x) for x in v["evr"][:N_PC_KEEP]] for k, v in bases.items()},
                "relevance_direction": "mean(clean relevant) - mean(clean non-relevant) in the full 2304-dim space of each basis; "
                                       "movement projection = (attack - control) . g / ||g||^2 (fraction of the genuine gap). "
                                       "Descriptive: g is estimated on the same clean documents (no held-out split).",
                "orig_doc_variant": "same, but attacked sequences pooled over original-document tokens only",
                "arrow_sample": f"<= {N_ARROWS} pairs per group, numpy default_rng(42); all pairs used for numbers"}
    (st / "pca_metadata.json").write_text(json.dumps(pca_meta, indent=2))
    print(f"[24] PCA done ({time.time() - t0:.0f}s)", flush=True)

    # ---- plots ----------------------------------------------------------------------------------------------------
    X = np.arange(144)
    W = 28
    fig, axes = plt.subplots(2, len(AS.STATS), figsize=(4.2 * len(AS.STATS), 9.4))       # A0 overview
    fig.subplots_adjust(hspace=0.35)
    for c_, sname in enumerate(AS.STATS):
        dq = A[(A.statistic == sname) & (A.region == "query")].set_index("head_label").loc[labels].cohen_d.values
        dd = A[(A.statistic == sname) & (A.region == "doc")].set_index("head_label").loc[labels].cohen_d.values
        v = _vmax(dq, dd)
        _heatmap(axes[0, c_], dq, f"{sname}: query", v)
        im_ = _heatmap(axes[1, c_], dd, f"{sname}: document", v)
        fig.colorbar(im_, ax=axes[:, c_], shrink=0.5, location="bottom", label="Cohen d")
    fig.suptitle("Clean inputs: Cohen d, relevant (qrel 2/3) − non-relevant (qrel 0); red = larger for relevant",
                 x=0.01, ha="left", fontsize=10)
    P5._save(fig, pdir / "A_heatmaps_cohen_d_overview.png")
    for sname in AS.STATS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))                                    # A1 per-stat heatmaps
        dq = A[(A.statistic == sname) & (A.region == "query")].set_index("head_label").loc[labels].cohen_d.values
        dd = A[(A.statistic == sname) & (A.region == "doc")].set_index("head_label").loc[labels].cohen_d.values
        v = _vmax(dq, dd)
        _heatmap(axes[0], dq, "query tokens", v)
        im_ = _heatmap(axes[1], dd, "document tokens", v)
        fig.colorbar(im_, ax=axes, shrink=0.8, label="Cohen d (+ = larger for relevant)")
        fig.suptitle(f"{NAMES[sname]}: relevant vs non-relevant (clean)", x=0.01, ha="left", fontsize=10)
        P5._save(fig, pdir / f"A_heatmap_{sname}.png")

        fig, axes = plt.subplots(4, 1, figsize=(W, 15), gridspec_kw={"height_ratios": [1.3, 1, 1.3, 1]})  # A2 144 heads
        for k_, r in enumerate(("query", "doc")):
            t = A[(A.statistic == sname) & (A.region == r)].set_index("head_label").loc[labels]
            ax = axes[2 * k_]
            ax.plot(X, t.mean_rel, color=P5.BLUE, lw=1.6, marker="o", ms=3, label=f"relevant (qrel 2/3), n={int(isrel.sum())}")
            ax.plot(X, t.mean_nonrel, color=P5.ORANGE, lw=1.6, marker="s", ms=3, label=f"non-relevant (qrel 0), n={int((~isrel).sum())}")
            P21.head_axis(ax, labels, canon)
            ax.set_ylabel(sname); ax.legend(fontsize=9, loc="upper left")
            ax.set_title(f"{r} tokens: {NAMES[sname]} per head (group means)", loc="left", fontsize=10, pad=16)
            ax = axes[2 * k_ + 1]
            ax.bar(X, t.difference, color=[P5.BLUE if v_ > 0 else P5.ORANGE for v_ in t.difference])
            ax.axhline(0, color=P5.INK, lw=1)
            P21.head_axis(ax, labels, canon)
            ax.set_ylabel("relevant − non-relevant")
            ax.set_title(f"{r} tokens: difference of means (blue = larger for relevant)", loc="left", fontsize=10, pad=16)
        fig.tight_layout(); P5._save(fig, pdir / f"A_heads_{sname}.png")

        fig, axes = plt.subplots(2, 5, figsize=(17, 7))                                      # A3 top-5 distributions
        for k_, r in enumerate(("query", "doc")):
            top = A[(A.statistic == sname) & (A.region == r)].sort_values("abs_d", ascending=False).head(5)
            for c_, (_, row) in enumerate(top.iterrows()):
                j = labels.index(row.head_label)
                V = S[cl, RI[r], SI[sname], j]
                ax = axes[k_, c_]
                bp = ax.boxplot([V[isrel], V[~isrel]], widths=0.55, patch_artist=True, showfliers=True,
                                flierprops={"markersize": 2})
                for patch, colr in zip(bp["boxes"], (P5.BLUE, P5.ORANGE)):
                    patch.set_facecolor(colr); patch.set_alpha(0.45)
                ax.set_xticks([1, 2]); ax.set_xticklabels(["relevant", "non-rel."])
                ax.set_title(f"{r} {row.head_label}\nd={row.cohen_d:+.2f}, AUROC={row.auroc_raw:.2f}", fontsize=9, loc="left")
        fig.suptitle(f"{NAMES[sname]}: top-5 heads by |Cohen d| (clean)", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); P5._save(fig, pdir / f"A_top_heads_{sname}.png")

        fig, axes = plt.subplots(4, 3, figsize=(14, 18))                                     # B1 paired heatmaps
        mats = {(g, ra): B[(B.statistic == sname) & (B.region == ra) & (B.relevance_group == g)].set_index("head_label").loc[labels]
                for g in (REL, NON) for ra, _ in PAIRED_REGIONS}
        vd = _vmax(*[m.mean_delta.values for m in mats.values()], pct=98)      # clipped: single outlier heads (L8H6) saturate
        vr = _vmax(*[m.spearman_delta_vs_delta_score.values for m in mats.values()])
        for c_, (ra, _) in enumerate(PAIRED_REGIONS):
            for r_, g in enumerate((REL, NON)):
                t = mats[(g, ra)]
                imd = _heatmap(axes[r_, c_], t.mean_delta.values, f"{g}: mean Δ, {ra} (n={int(t.n_instances.iloc[0])})", vd)
                imr = _heatmap(axes[2 + r_, c_], t.spearman_delta_vs_delta_score.values, f"{g}: ρ(Δstat, Δscore), {ra}", vr, cmap="PuOr_r")
        fig.colorbar(imd, ax=axes[:2], shrink=0.6, extend="both", label="mean(attack − control), scale clipped at 98th pct |Δ|")
        fig.colorbar(imr, ax=axes[2:], shrink=0.6, label="Spearman ρ")
        fig.suptitle(f"{NAMES[sname]}: successful attacks vs own padded control (orig/doc Δ vs control document)",
                     x=0.01, ha="left", fontsize=10)
        P5._save(fig, pdir / f"B_delta_heatmap_{sname}.png")

        fig, axes = plt.subplots(2, 5, figsize=(17, 7))                                      # B2 top paired heads
        for r_, g in enumerate((REL, NON)):
            top = B[(B.statistic == sname) & (B.relevance_group == g)].sort_values("abs_dz", ascending=False).head(5)
            k = grp == g
            for c_, (_, row) in enumerate(top.iterrows()):
                j = labels.index(row.head_label)
                vc = S[ic[k], RI[row.control_region], SI[sname], j]
                va = S[ia[k], RI[row.region], SI[sname], j]
                ax = axes[r_, c_]
                bp = ax.boxplot([vc, va], widths=0.55, patch_artist=True, flierprops={"markersize": 2})
                for patch, colr in zip(bp["boxes"], (P5.BLUE, P5.ORANGE)):
                    patch.set_facecolor(colr); patch.set_alpha(0.45)
                ax.set_xticks([1, 2]); ax.set_xticklabels(["control", "attack"])
                ax.set_title(f"{g[:6]}. {row.region} {row.head_label}\nΔ={row.mean_delta:+.3g}, dz={row.paired_dz:+.2f}, "
                             f"ρ={row.spearman_delta_vs_delta_score:+.2f}", fontsize=8, loc="left")
        fig.suptitle(f"{NAMES[sname]}: top-5 (region, head) by |paired dz|, successful attacks", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); P5._save(fig, pdir / f"B_top_heads_{sname}.png")

        fig, axes = plt.subplots(2, 1, figsize=(W, 8.4))                                     # E1 inserted per head
        for ax, g in zip(axes, (REL, NON)):
            t = E[(E.statistic == sname) & (E.relevance_group == g)].set_index("head_label").loc[labels]
            for col_, colr, mk, lab in (("mean_control_doc", P5.BLUE, "o", "control: document"),
                                        ("mean_orig", P5.AQUA, "D", "attack: original-document tokens"),
                                        ("mean_full_doc", P5.ORANGE, "s", "attack: full document"),
                                        ("mean_ins", "#d62728", "^", "attack: inserted tokens only"),
                                        ("mean_orig_size_matched", P5.INK2, "x", "attack: original tokens, size-matched to inserted")):
                ax.plot(X, t[col_], color=colr, lw=1.3, marker=mk, ms=3, label=lab)
            P21.head_axis(ax, labels, canon)
            ax.set_ylabel(sname); ax.legend(fontsize=8, loc="upper left")
            ax.set_title(f"{P.GROUP_LABEL[g]}, successful attacks (n={int(t.n_instances.iloc[0])}): {NAMES[sname]} by region",
                         loc="left", fontsize=10, pad=16)
        fig.tight_layout(); P5._save(fig, pdir / f"E_inserted_{sname}.png")

    fig, axes = plt.subplots(1, len(AS.STATS), figsize=(4 * len(AS.STATS), 3.8), sharey=True)   # E2 overview
    for ax, sname in zip(axes, AS.STATS):
        for g, colr in ((REL, P5.BLUE), (NON, P5.ORANGE)):
            t = E[(E.statistic == sname) & (E.relevance_group == g)]
            lv = t.groupby("layer")[["dz_ins_vs_orig", "dz_ins_vs_size_matched"]].median()
            ax.plot(lv.index, lv.dz_ins_vs_orig, color=colr, marker="o", ms=3, label=f"{g}: ins vs orig")
            ax.plot(lv.index, lv.dz_ins_vs_size_matched, color=colr, marker="x", ms=4, ls="--", label=f"{g}: ins vs size-matched orig")
        ax.axhline(0, color=P5.INK, lw=1)
        ax.set_title(sname, loc="left", fontsize=9); ax.set_xticks(range(12)); ax.set_xlabel("layer")
    axes[0].set_ylabel("median over the layer's heads of paired dz")
    axes[0].legend(fontsize=7)
    fig.suptitle("Inserted tokens vs original-document tokens of the same successful attack (+ = larger on inserted tokens)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); P5._save(fig, pdir / "E_inserted_overview.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))                                        # PCA A/B
    for ax, name, title in ((axes[0], "raw", "clean, raw"), (axes[1], "query_centered", "clean, query-centred")):
        ev = bases[name]["evr"]
        for g, colr, mk in ((REL, P5.BLUE, "o"), (NON, P5.ORANGE, "s")):
            t = pcs[pcs.relevance_group == g]
            ax.scatter(t[f"{name}_PC1"], t[f"{name}_PC2"], s=18, color=colr, marker=mk, alpha=0.7, edgecolor="white",
                       linewidth=0.5, label=f"{g} (n={len(t)})")
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var)")
        ax.set_title(f"L9–L11 pooled document head representation (2304-d), {title}", loc="left", fontsize=9)
        ax.legend(fontsize=8)
    P5._save(fig, pdir / "PCA_clean.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))                                         # PCA C
    arrow_rng = np.random.default_rng(42)
    for r_, name in enumerate(bases):
        for c_, g in enumerate((REL, NON)):
            ax = axes[r_, c_]
            ax.scatter(pcs[f"{name}_PC1"], pcs[f"{name}_PC2"], s=8, color=P5.GRID if hasattr(P5, "GRID") else "#e4e3df",
                       zorder=1, label="clean (all)")
            t = pairs[pairs.relevance_group == g]
            smp = t.iloc[np.sort(arrow_rng.choice(len(t), min(N_ARROWS, len(t)), replace=False))]
            for _, row in smp.iterrows():
                ax.annotate("", xy=(row[f"{name}_PC1_attack"], row[f"{name}_PC2_attack"]),
                            xytext=(row[f"{name}_PC1_control"], row[f"{name}_PC2_control"]),
                            arrowprops={"arrowstyle": "->", "color": P5.INK2, "lw": 0.5, "alpha": 0.6}, zorder=2)
            for lab, (x_, y_), colr, mk in (
                    ("clean relevant centroid", pcs[pcs.relevance_group == REL][[f"{name}_PC1", f"{name}_PC2"]].mean(), P5.BLUE, "o"),
                    ("clean non-relevant centroid", pcs[pcs.relevance_group == NON][[f"{name}_PC1", f"{name}_PC2"]].mean(), P5.ORANGE, "s"),
                    ("control centroid", t[[f"{name}_PC1_control", f"{name}_PC2_control"]].mean(), P5.AQUA, "D"),
                    ("attack centroid", t[[f"{name}_PC1_attack", f"{name}_PC2_attack"]].mean(), "#d62728", "^")):
                ax.scatter([x_], [y_], s=110, color=colr, marker=mk, edgecolor=P5.INK, linewidth=1, zorder=5, label=lab)
            cx, cy = t[f"{name}_PC1_control"].mean(), t[f"{name}_PC2_control"].mean()
            ax.annotate("", xy=(t[f"{name}_PC1_attack"].mean(), t[f"{name}_PC2_attack"].mean()), xytext=(cx, cy),
                        arrowprops={"arrowstyle": "-|>", "color": "#d62728", "lw": 2.2}, zorder=6)
            m_ = MOV[(MOV.representation == "full_doc") & (MOV.basis == name) & (MOV.relevance_group == g)].iloc[0]
            ax.set_title(f"{name}, {g}: control → successful attack ({len(smp)} of {len(t)} arrows)\n"
                         f"mean proj. on rel. direction = {m_.mean_proj_on_rel_direction_frac_of_gap:+.3f} of genuine gap; "
                         f"frac > 0 = {m_.frac_proj_positive:.2f}", loc="left", fontsize=9)
            ev = bases[name]["evr"]
            ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
            ax.legend(fontsize=7, loc="best")
    P5._save(fig, pdir / "PCA_attack_movement.png")
    print(f"[24] plots done ({time.time() - t0:.0f}s)", flush=True)

    # ---- compact summary table -------------------------------------------------------------------------------------
    rows = []
    for sname in AS.STATS:
        a = A[A.statistic == sname].sort_values("abs_d", ascending=False).iloc[0]
        cells = {}
        for g in (REL, NON):
            b = B[(B.statistic == sname) & (B.relevance_group == g)].sort_values("abs_dz", ascending=False).iloc[0]
            cells[g] = b
        e = E[E.statistic == sname]
        rows.append({"statistic": sname,
                     "strongest rel/nonrel head": f"{a.head_label} ({a.region})",
                     "Cohen d": round(a.cohen_d, 2), "AUROC (raw, rel>nonrel)": round(a.auroc_raw, 3),
                     "strongest attack-change head (rel / nonrel)":
                         " / ".join(f"{cells[g].head_label} ({cells[g].region})" for g in (REL, NON)),
                     "paired delta [dz] (rel / nonrel)":
                         " / ".join(f"{cells[g].mean_delta:+.3g} [{cells[g].paired_dz:+.2f}]" for g in (REL, NON)),
                     "Spearman with delta_score (rel / nonrel)":
                         " / ".join(f"{cells[g].spearman_delta_vs_delta_score:+.2f}" for g in (REL, NON)),
                     "note": f"ins>orig in {100 * (e.dz_ins_vs_orig > 0).mean():.0f}% of head×group cells; "
                             f"ins>size-matched in {100 * (e.dz_ins_vs_size_matched > 0).mean():.0f}%"})
    CT = pd.DataFrame(rows)
    CT.to_csv(st / "summary_table.csv", index=False)
    md = "| " + " | ".join(CT.columns) + " |\n|" + "---|" * len(CT.columns) + "\n" + \
         "\n".join("| " + " | ".join(map(str, r)) + " |" for r in CT.itertuples(index=False))
    (st / "summary_table.md").write_text(md)
    summary = {"created": now(), "n_sequences": n, "n_clean_docs": int(cl.sum()), "n_rel_docs": int(isrel.sum()),
               "n_successful_attacks": int(len(succ)), "n_successful_by_group": succ.relevance_group.value_counts().to_dict(),
               "n_successful_docs_by_group": succ.groupby("relevance_group").pair_id.nunique().to_dict(),
               "validation": val, "length_confound": confound, "pca_components": PCS.to_dict("records"),
               "pca_movement": MOV.to_dict("records"), "seconds": time.time() - t0,
               "conventions": {"var": "population variance (ddof 0) over the 64 dims per token, averaged over tokens",
                               "one_token_region": "top_share = 1, entropy = NaN", "zero_token_region": "all NaN",
                               "size_matched": f"mean over {N_SIZE_DRAWS} random original-document token subsets of the inserted "
                                               "region's size, numpy default_rng(42), successful attacks only",
                               "auroc_raw": "P(stat_relevant > stat_nonrelevant); auroc_separation = max(raw, 1 - raw)",
                               "successful": "delta_score = score_attack - score_control > 0 (stage 22)"}}
    (st / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_status(st, {"status": "success", "finished": now()})
    print(CT.to_string(index=False))
    print(f"[24] -> {st} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
