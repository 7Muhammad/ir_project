"""
scripts/diag_paired_all_layers.py
==================================
DIAGNOSTIC ONLY (changes nothing in stages 17-21): sanity checks of the all-layers
(144-head) paired pipeline after relevant and non-relevant looked alike.

  1  label / row-filter integrity of 18_paired_forward_all_layers
  2  clean qrel 2/3 vs qrel 0 (stage-00 balanced sample) re-extracted with the NEW
     code path (paired.run_paired_sequences, 144 heads) AND the stage-13 path;
     L9-L11 compared with the stage-13 cache (max/mean |diff|, correlation)
  3  relevant vs non-relevant padded controls: ALL vs SUCCESSFUL only
  4  raw example rows
  5  query/document pool tokens (clean, control, attack)
  6  head indexing: pre-o_proj capture == SelfAttention.o input per layer; zeroing
     the value rows of head h in layer L zeroes exactly slice h; no stale captures
  7  per-head / per-layer relevant-vs-non-relevant effect sizes

Outputs: outputs/diag_paired_all_layers/ (+ plots). Needs the model for 2, 5, 6.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import EncoderHeadCapture, all_encoder_heads, encoder_heads, run_encoder_head_sequences  # noqa: E402
from exp16lib.inputs import collate, encode_clean  # noqa: E402
from exp16lib.run_utils import load_model, load_status, read_jsonl  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("p21", EXP_DIR / "scripts" / "21_plot_paired.py")
P21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P21)
P5 = P21.P5
REL, NON = P.RELEVANT, P.NONRELEVANT
SHOW = ["L9H4", "L10H0", "L10H2", "L11H10", "L11H11"]


def effect_table(df, cols, labels, unit="pair_id"):
    """Per head: doc-level (mean per `unit` first) rel/nonrel means, diff, Cohen d, equal-query-weight diff."""
    d = df.groupby([unit, "qid", "relevance_group"])[cols].mean().reset_index()
    r, n = d[d.relevance_group == REL], d[d.relevance_group == NON]
    qr = r.groupby("qid")[cols].mean(); qn = n.groupby("qid")[cols].mean()
    both = qr.index.intersection(qn.index)
    rows = []
    for c, l in zip(cols, labels):
        a, b = r[c].values, n[c].values
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        rows.append({"head_label": l, "layer": int(l[1:l.index("H")]), "mean_rel": a.mean(), "mean_nonrel": b.mean(),
                     "difference": a.mean() - b.mean(), "cohen_d": (a.mean() - b.mean()) / sp,
                     "within_query_difference": float((qr.loc[both, c] - qn.loc[both, c]).mean()),
                     "n_rel_docs": len(a), "n_nonrel_docs": len(b), "n_queries_both": len(both)})
    t = pd.DataFrame(rows)
    t["rank_abs_d"] = t.cohen_d.abs().rank(ascending=False, method="first").astype(int)
    return t


def plot_rel_non(tabs, title, png, labels, canon):
    fig, axes = plt.subplots(len(tabs), 1, figsize=(28, 4.2 * len(tabs)), squeeze=False)
    X = np.arange(len(labels))
    for ax, (name, t) in zip(axes[:, 0], tabs.items()):
        t = t.set_index("head_label").loc[labels]
        ax.plot(X, t.mean_rel, color=P5.BLUE, lw=1.6, marker="o", ms=3, label="relevant (qrel 2/3)")
        ax.plot(X, t.mean_nonrel, color=P5.ORANGE, lw=1.6, marker="s", ms=3, label="non-relevant (qrel 0)")
        ax2 = ax.twinx()
        ax2.bar(X, t.cohen_d, color=P5.INK2, alpha=0.25, width=0.8, label="Cohen d (rel − nonrel)")
        ax2.axhline(0, color=P5.INK2, lw=0.6)
        ax2.set_ylabel("Cohen d", color=P5.INK2)
        P21.head_axis(ax, labels, canon)
        ax.set_ylabel("head query–doc cosine")
        ax.set_title(f"{name}: {int(t.n_rel_docs.iloc[0])} rel / {int(t.n_nonrel_docs.iloc[0])} non-rel docs "
                     f"(doc-level means); |d| max {t.cohen_d.abs().max():.2f}", loc="left", fontsize=9, pad=16)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    P5._save(fig, png)


def main():
    cfg = load_config(EXP_DIR / "configs" / "paired_all_layers.yaml")
    P.validate_cfg(cfg)
    out = output_dir(cfg)
    dd = out / "diag_paired_all_layers"
    dd.mkdir(exist_ok=True)
    rep = {}
    labels = P.head_labels()
    canon = {h.label for h in encoder_heads()}
    hc_c, hc_a = P.head_cols("control"), P.head_cols("attack")
    attacks = list(load_status(out / P.MANIFEST_DIR)["sha256"]["attacks"])

    # ---- 1. labels / filtering ------------------------------------------------------------
    df = P.load_forward_rows(out / P.FORWARD_DIR, attacks)
    P.check_forward_frame(df)
    base = pd.DataFrame(read_jsonl(out / P.MANIFEST_DIR / "base_pairs.jsonl.gz"))
    import ir_datasets
    qrels = {(q.query_id, q.doc_id): int(q.relevance) for q in ir_datasets.load(cfg["qrels"]["irds_qrels_id"]).qrels_iter()}
    truth = pd.Series([qrels.get(k) for k in zip(df.qid, df.docid)], index=df.index)
    c1 = {"all_rows_grade_matches_nist_qrels": bool((truth.values == df.qrel_grade.values).all()),
          "grades_present": sorted(df.qrel_grade.unique().tolist()),
          "relevance_group_consistent": bool(((df.qrel_grade.isin([2, 3])) == (df.relevance_group == REL)).all()),
          "pair_id_is_qid_docid": bool((df.pair_id == df.qid + "_" + df.docid).all()),
          "duplicate_attack_pair_rows": int(df.duplicated(["attack_name", "pair_id"]).sum()),
          "pairs_in_both_groups": int(len(set(df[df.relevance_group == REL].pair_id) & set(df[df.relevance_group == NON].pair_id))),
          "base_manifest_grades_match_nist": bool(all(qrels.get((r.qid, r.docid)) == r.qrel_grade for r in base.itertuples())),
          "grade1_or_unjudged_in_manifest": int((~base.qrel_grade.isin([0, 2, 3])).sum())}
    for g in P.GROUPS:
        for name, sub in (("all", df), ("successful", df[df.successful])):
            x = sub[sub.relevance_group == g]
            c1[f"{g}_{name}"] = {"rows": len(x), "unique_qid_docid": int(x.pair_id.nunique()), "qids": int(x.qid.nunique()),
                                 "grade_counts": x.drop_duplicates("pair_id").qrel_grade.value_counts().to_dict()}
    # identical head rows across different documents would indicate reused encodings/rows
    first = df.drop_duplicates("pair_id")
    c1["duplicate_control_head_vectors_across_docs"] = int(first[hc_c].round(10).duplicated().sum())
    rep["1_labels"] = c1
    print(json.dumps(c1, indent=1, default=str))

    # ---- 7 / 3. relevant vs non-relevant controls: all vs successful ------------------------
    t_all = effect_table(df, hc_c, labels)
    t_succ = effect_table(df[df.successful], hc_c, labels)
    t_all.to_csv(dd / "rel_vs_nonrel_controls_all.csv", index=False)
    t_succ.to_csv(dd / "rel_vs_nonrel_controls_successful.csv", index=False)

    # ---- 2. clean qrel sample through the NEW extraction ----------------------------------
    model, tok, tid, fid, dev = load_model(cfg)
    qm = read_jsonl(out / "manifests" / "qrel_balanced.jsonl")
    seqs = [encode_clean(tok, r["query"], r["passage"], int(cfg["model"]["max_length"])) for r in qm]
    hs = all_encoder_heads(P.LAYERS)
    _, Hn, Sn = P.run_paired_sequences(model, seqs, tok.pad_token_id, dev, 128, hs, tid, fid)      # new code path
    He = run_encoder_head_sequences(model, seqs, tok.pad_token_id, dev, 64, hs)                    # stage-13 path
    clean = pd.DataFrame({k: [r[k] for r in qm] for k in ("qid", "docid", "pair_id", "qrel_grade", "relevance_group")})
    clean["in_bm25_top1000"] = [r["in_bm25_top1000"] for r in qm]
    clean["score_clean"] = Sn
    clean = pd.concat([clean, pd.DataFrame(Hn, columns=[f"clean_head_{l}" for l in labels])], axis=1)
    clean.to_csv(dd / "clean_qrel_144_heads.csv.gz", index=False, float_format="%.9g")
    s13 = pd.read_csv(out / "13_late_encoder_heads" / "qrel_late_encoder_heads.csv.gz", dtype={"qid": str, "docid": str, "pair_id": str})
    late = [l for l in labels if int(l[1:l.index("H")]) >= 9]
    m = clean.reset_index().rename(columns={"index": "_row"}).merge(s13, on=["pair_id", "relevance_group"], validate="one_to_one")
    A = m[[f"clean_head_{l}" for l in late]].values; B = m[[f"enc_{l}" for l in late]].values
    idx = [labels.index(l) for l in late]
    c2 = {"n_rows_overlap": len(m), "n_heads_compared": len(late),
          "new_vs_stage13_max_abs_diff": float(np.abs(A - B).max()), "new_vs_stage13_mean_abs_diff": float(np.abs(A - B).mean()),
          "new_vs_stage13_corr": float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
          "fullforward_vs_encoderonly_144_max_abs_diff": float(np.abs(Hn - He).max()),
          "stage13_path_vs_cache_max_abs_diff": float(np.abs(He[m._row.values][:, idx] - B).max())}
    rep["2_clean_reproduction"] = c2
    print(json.dumps(c2, indent=1))
    if c2["new_vs_stage13_max_abs_diff"] > 1e-4:
        (dd / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        raise SystemExit("STOP: new extraction does not reproduce stage-13 cache")
    cc = [f"clean_head_{l}" for l in labels]
    t_clean = effect_table(clean, cc, labels)
    t_clean_bm = effect_table(clean[clean.in_bm25_top1000], cc, labels)
    t_clean.to_csv(dd / "rel_vs_nonrel_clean.csv", index=False)
    t_clean_bm.to_csv(dd / "rel_vs_nonrel_clean_bm25_subset.csv", index=False)
    # same documents: clean vs mean padded control (does padding itself change the head sims?)
    ctl_doc = df.groupby("pair_id")[hc_c].mean()
    mm = clean.set_index("pair_id").join(ctl_doc, how="inner")
    cl, ct = mm[cc].values, mm[hc_c].values
    rep["2b_clean_vs_mean_control_same_doc"] = {"n_docs": len(mm), "mean_abs_diff": float(np.abs(cl - ct).mean()),
                                               "max_abs_diff": float(np.abs(cl - ct).max()),
                                               "corr": float(np.corrcoef(cl.ravel(), ct.ravel())[0, 1])}

    plot_rel_non({"clean qrel sample (stage-00 balanced, all judged docs), NEW 144-head extraction": t_clean,
                  "same, restricted to docs in the BM25 top-1000 (= paired population)": t_clean_bm},
                 "Clean genuine relevant vs non-relevant, all 144 encoder heads", dd / "fig_diag_clean_rel_vs_nonrel.png", labels, canon)
    plot_rel_non({"ALL padded controls (no success filter)": t_all},
                 "Padded controls, relevant vs non-relevant (no success filter)", dd / "fig_diag_controls_all.png", labels, canon)
    plot_rel_non({"SUCCESSFUL-attack padded controls": t_succ},
                 "Padded controls of successful attacks, relevant vs non-relevant", dd / "fig_diag_controls_successful.png",
                 labels, canon)

    def layer_sum(t):
        return t.groupby("layer").agg(mean_abs_d=("cohen_d", lambda x: x.abs().mean()), max_abs_d=("cohen_d", lambda x: x.abs().max()),
                                      mean_diff=("difference", "mean")).round(3)
    comp = pd.concat({"clean_all_judged": layer_sum(t_clean), "clean_bm25_subset": layer_sum(t_clean_bm),
                      "controls_all": layer_sum(t_all), "controls_successful": layer_sum(t_succ)}, axis=1)
    comp.to_csv(dd / "layer_summary_rel_vs_nonrel.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    for (name, t), c, mk in zip({"clean, all judged docs": t_clean, "clean, BM25 top-1000 subset": t_clean_bm,
                                 "padded controls, all": t_all, "padded controls, successful": t_succ}.items(),
                                [P5.BLUE, P5.AQUA, P5.INK2, P5.ORANGE], "oD^s"):
        v = layer_sum(t)
        ax.plot(v.index, v.mean_abs_d, color=c, marker=mk, label=name)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("mean |Cohen d| over the layer's 12 heads")
    ax.set_title("Relevant vs non-relevant separation by layer", loc="left", fontsize=9); ax.legend(fontsize=8)
    ax.set_xticks(range(12))
    P5._save(fig, dd / "fig_diag_layer_separation.png")

    # ---- 4. raw examples ----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    ex = []
    bmap = base.set_index("pair_id")
    for g in P.GROUPS:
        pids = rng.choice(df[df.relevance_group == g].pair_id.unique(), 4, replace=False)
        for pid in pids:
            r = df[(df.pair_id == pid) & (df.attack_name == "relevant_start_5")].iloc[0]
            b = bmap.loc[pid]
            ex.append({"group": g, "qid": r.qid, "docid": r.docid, "qrel": int(r.qrel_grade), "query": b.query,
                       "passage": b.passage[:90], "score_control": round(r.score_control, 3),
                       "score_attack": round(r.score_attack, 3), "delta": round(r.delta_score, 3),
                       **{f"ctl_{h}": round(r[f"control_head_{h}"], 4) for h in SHOW},
                       **{f"atk_{h}": round(r[f"attack_head_{h}"], 4) for h in SHOW}})
    exdf = pd.DataFrame(ex)
    exdf.to_csv(dd / "raw_examples.csv", index=False)
    rep["4_examples"] = ex

    # ---- 5. pools ------------------------------------------------------------------------------
    from exp16lib.inputs import encode_attack_and_control
    pools = []
    for e in ex[:2] + ex[4:6]:
        b = bmap.loc[f"{e['qid']}_{e['docid']}"]
        rr = [x for x in read_jsonl(out / P.MANIFEST_DIR / "attacks" / "relevant_end_3.jsonl.gz")
              if x["pair_id"] == f"{e['qid']}_{e['docid']}"][0]
        atk, ctl, info = encode_attack_and_control(tok, b.query, b.passage, rr["attacked_passage"], 512)
        cln = encode_clean(tok, b.query, b.passage, 512)
        dec = lambda s, mask: tok.convert_ids_to_tokens([i for i, m_ in zip(s.input_ids, mask) if m_])  # noqa: E731
        pools.append({"group": e["group"], "pair_id": f"{e['qid']}_{e['docid']}", "attack": "relevant_end_3",
                      "query_span": info["query_span"], "doc_span": info["doc_span"],
                      "inserted_positions": info["inserted_positions"],
                      "n_query_tokens": ctl.n_query_tokens, "n_doc_tokens_control": ctl.n_doc_tokens,
                      "n_doc_tokens_attack": atk.n_doc_tokens, "n_doc_tokens_clean": cln.n_doc_tokens,
                      "query_pool": dec(ctl, ctl.query_mask),
                      "control_doc_pool_equals_clean_doc_pool": dec(ctl, ctl.doc_mask) == dec(cln, cln.doc_mask),
                      "control_doc_pool_has_pad": tok.pad_token in dec(ctl, ctl.doc_mask),
                      "inserted_slots_in_control_pool": any(ctl.doc_mask[i] for i in info["inserted_positions"]),
                      "control_doc_pool_first_last": dec(ctl, ctl.doc_mask)[:6] + ["…"] + dec(ctl, ctl.doc_mask)[-4:],
                      "attack_doc_pool_first": dec(atk, atk.doc_mask)[:12],
                      "tokens_outside_pools": tok.convert_ids_to_tokens([i for i, q, d in zip(ctl.input_ids, ctl.query_mask, ctl.doc_mask)
                                                                         if not q and not d])})
    rep["5_pools"] = pools

    # ---- 6. head indexing --------------------------------------------------------------------------
    batch = collate(seqs[:6], tok.pad_token_id, dev)
    enc = model.encoder
    o_in, attn_out = {}, {}
    hooks = [enc.block[L].layer[0].SelfAttention.o.register_forward_hook(
        (lambda L: lambda mod, a, o: o_in.__setitem__(L, (a[0].detach().clone(), o.detach().clone())))(L)) for L in range(12)]
    with torch.inference_mode(), EncoderHeadCapture(enc, hs, keep_inputs=True) as ec:
        ec.set_masks(batch["query_mask"], batch["doc_mask"])
        enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        cap = {L: ec.inputs[L] for L in range(12)}
        cos = dict(ec.cos)
    for h in hooks:
        h.remove()
    from exp16lib.pooling import pooled_cosine
    c6 = {"n_layers_captured": len(cap), "n_prehooks_registered_per_layer": 1,
          "capture_equals_o_input_all_layers": all(torch.equal(cap[L], o_in[L][0]) for L in range(12)),
          "layers_distinct": all(not torch.allclose(cap[L], cap[L + 1]) for L in range(11)),
          "slice_recompute_max_diff": max(float((pooled_cosine(cap[h.layer][..., h.head_idx * 64:(h.head_idx + 1) * 64],
                                                               batch["query_mask"], batch["doc_mask"]) - cos[h.label]).abs().max())
                                          for h in hs),
          "d_kv": model.config.d_kv, "num_heads": model.config.num_heads}
    ablate = []
    for L, h in ((0, 0), (0, 11), (4, 7), (9, 4), (11, 11)):
        v = enc.block[L].layer[0].SelfAttention.v.weight
        saved = v.data.clone()
        v.data[h * 64:(h + 1) * 64] = 0
        with torch.inference_mode(), EncoderHeadCapture(enc, hs, keep_inputs=True) as ec:
            ec.set_masks(batch["query_mask"], batch["doc_mask"])
            enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            x = ec.inputs[L]
        v.data.copy_(saved)
        others = [k for k in range(12) if k != h]
        ablate.append({"layer": L, "head": h, "slice_h_all_zero": bool((x[..., h * 64:(h + 1) * 64] == 0).all()),
                       "other_slices_unchanged": bool(all(torch.equal(x[..., k * 64:(k + 1) * 64], cap[L][..., k * 64:(k + 1) * 64])
                                                          for k in others))})
    c6["value_zeroing"] = ablate
    b2 = collate(seqs[6:12], tok.pad_token_id, dev)
    with torch.inference_mode(), EncoderHeadCapture(enc, hs) as ec:
        ec.set_masks(b2["query_mask"], b2["doc_mask"])
        enc(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"])
        c6["no_stale_capture"] = bool(not torch.allclose(ec.cos["L5H3"], cos["L5H3"][:len(ec.cos["L5H3"])]))
    rep["6_head_indexing"] = c6
    print(json.dumps(c6, indent=1))

    top = {"clean_all_judged": t_clean, "clean_bm25_subset": t_clean_bm, "controls_all": t_all, "controls_successful": t_succ}
    rep["7_top_heads"] = {k: t.sort_values("rank_abs_d").head(10)[["head_label", "mean_rel", "mean_nonrel", "difference",
                                                                  "cohen_d", "within_query_difference"]].round(4).to_dict("records")
                          for k, t in top.items()}
    rep["7_layer_summary"] = {f"{a}/{b}": v for (a, b), v in comp.to_dict().items()}
    (dd / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(comp.to_string())
    print(f"[diag] -> {dd}")


if __name__ == "__main__":
    main()
