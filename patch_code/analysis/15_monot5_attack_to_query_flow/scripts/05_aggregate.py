#!/usr/bin/env python3
"""
scripts/05_aggregate.py
========================
Collect the per-attack raw outputs of stages 02-04, verify completeness
against the manifest (every example x head [x query token] exactly once —
fail loudly otherwise), and write:

  raw_all_query_edges.csv            (1) raw all-query edge results
  raw_single_query_edges.csv.gz      (2) raw individual-query-token results
  raw_whole_head.csv                 (3) raw whole-head reference results
  per_attack_head_all_query.csv      (4) per attack x head means (all-query)
  global_head_summary.csv            (5) equal-weight 18-head summary (edge + whole-head)
  per_attack_head_whole_head.csv         per attack x head means (whole-head)
  edge_vs_whole_head.csv             (6) SECONDARY edge/whole-head comparison (global per head)
  edge_vs_whole_head_per_attack.csv      same per attack x head
  token_summary_by_head.csv          (7) descriptive single-token summary
  token_per_example.csv                  per example x head: top token, frac positive,
                                         all-query vs sum-of-single (descriptive only)
  baseline_consistency.json              fresh live vs cached Exp 01 baselines

Equal weight (DECISIONS.md item 25): mean within attack, then unweighted mean
of attack means. No clipping, no filtering of rows anywhere.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401  (adds sibling experiment dirs to sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.patching import SKIP_EPSILON  # noqa: E402

from exp15lib.config import load_config, output_dir  # noqa: E402
from exp15lib.heads import load_heads  # noqa: E402
from exp15lib.run_utils import (  # noqa: E402
    cheap_stage_is_current, load_manifest, load_status, now, upstream_fingerprint, write_status,
)

METRICS = ["e_fwd", "e_rev", "e_combined", "raw_fwd", "raw_rev"]
WH_METRICS = ["whole_head_fwd", "whole_head_rev", "whole_head_combined", "whole_head_raw_fwd", "whole_head_raw_rev"]
STAGES = ["02_all_query", "03_single_query", "04_whole_head"]


def parse_args():
    p = argparse.ArgumentParser(description="Exp 15 stage 05: aggregation")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_stage(out, stage, attacks, sha, labels):
    frames, status_files = [], []
    for a in attacks:
        unit = out / stage / "per_attack" / a
        st_path = unit / "status.json"
        if not st_path.exists():
            raise RuntimeError(f"{stage}: attack {a} has not been run (missing {st_path})")
        st = load_status(unit)
        if st.get("status") != "success" or st.get("manifest_sha256") != sha or st.get("head_labels") != labels:
            raise RuntimeError(f"{stage}: attack {a} incomplete or from a different manifest/head set")
        frames.append(pd.read_csv(unit / "rows.csv.gz", dtype={"qid": str, "docid": str}))
        status_files.append(st_path)
    return pd.concat(frames, ignore_index=True), status_files


def check_complete(df, manifest, labels, stage, per_token=False):
    exp = []
    for r in manifest:
        for lab in labels:
            if per_token:
                exp += [(r["attack_name"], r["example_id"], lab, i) for i in range(r["n_query_tokens"])]
            else:
                exp.append((r["attack_name"], r["example_id"], lab))
    cols = ["attack_name", "example_id", "head_label"] + (["query_token_index"] if per_token else [])
    got = list(df[cols].itertuples(index=False, name=None))
    if len(got) != len(set(got)):
        raise RuntimeError(f"{stage}: duplicate rows")
    if set(got) != set(exp):
        raise RuntimeError(f"{stage}: rows do not match manifest x heads "
                           f"(missing {len(set(exp) - set(got))}, extra {len(set(got) - set(exp))})")


def equal_weight(df, metrics, head_order):
    per_attack = df.groupby(["head_label", "attack_name"], sort=False)[metrics].mean().reset_index()
    g = per_attack.groupby("head_label")[metrics]
    res = g.mean().add_prefix("mean_")
    res["n_attacks"] = per_attack.groupby("head_label").size()
    return res.reindex(head_order), per_attack


def ratio(num, den):
    return num / den if abs(den) > SKIP_EPSILON else np.nan


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    manifest, sha = load_manifest(out)
    heads = load_heads(cfg)
    labels = [h.label for h in heads]
    attacks = list(dict.fromkeys(r["attack_name"] for r in manifest))
    stage = out / "05_aggregate"

    status_files = [out / s / "per_attack" / a / "status.json" for s in STAGES for a in attacks]
    missing = [p for p in status_files if not p.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} upstream units missing, e.g. {missing[0]}")
    fp = upstream_fingerprint(status_files)
    if cheap_stage_is_current(stage, fp, ["global_head_summary.csv"], args.force):
        print("[05] aggregation up to date; skipping")
        return
    stage.mkdir(parents=True, exist_ok=True)

    aq, _ = load_stage(out, "02_all_query", attacks, sha, labels)
    sq, _ = load_stage(out, "03_single_query", attacks, sha, labels)
    wh, _ = load_stage(out, "04_whole_head", attacks, sha, labels)
    check_complete(aq, manifest, labels, "02_all_query")
    check_complete(sq, manifest, labels, "03_single_query", per_token=True)
    check_complete(wh, manifest, labels, "04_whole_head")

    aq.to_csv(stage / "raw_all_query_edges.csv", index=False)
    sq.to_csv(stage / "raw_single_query_edges.csv.gz", index=False)
    wh.to_csv(stage / "raw_whole_head.csv", index=False)

    # ---- baseline consistency (fresh live vs cached Exp 01) ------------------
    ex = aq.drop_duplicates(["attack_name", "example_id"])
    whx = wh.drop_duplicates(["attack_name", "example_id"]).set_index(["attack_name", "example_id"])
    exi = ex.set_index(["attack_name", "example_id"])
    cons = {
        "n_examples": int(len(ex)),
        "max_abs_live_minus_cached_control": float((ex.live_score_control - ex.score_control).abs().max()),
        "max_abs_live_minus_cached_attack": float((ex.live_score_attack - ex.score_attack).abs().max()),
        "mean_abs_live_minus_cached_control": float((ex.live_score_control - ex.score_control).abs().mean()),
        "mean_abs_live_minus_cached_attack": float((ex.live_score_attack - ex.score_attack).abs().mean()),
        "max_abs_edge_vs_wholehead_live_control": float((exi.live_score_control - whx.live_score_control).abs().max()),
        "max_abs_edge_vs_wholehead_live_attack": float((exi.live_score_attack - whx.live_score_attack).abs().max()),
        "max_m_control_norm_all_rows": float(aq.m_control_norm_max.max()),
        "max_decomposition_err_all_rows": float(aq[["decomp_err_control", "decomp_err_attack"]].max().max()),
        "n_alignment_boundary_shift_examples": int(ex.alignment_boundary_shift.sum()),
    }
    with open(stage / "baseline_consistency.json", "w") as fh:
        json.dump(cons, fh, indent=2)

    # ---- (4)/(5) all-query per attack x head, global equal weight ----------
    meta_cols = ["attack_name", "attack_token", "attack_position", "repetitions"]
    meta = aq.drop_duplicates("attack_name")[meta_cols]
    g = aq.groupby(["head_label", "layer", "head", "attack_name"], sort=False)
    pah = g[METRICS].mean().reset_index()
    pah["n_examples"] = g.size().values
    pah["median_e_combined"] = g["e_combined"].median().values
    pah["frac_e_combined_gt0"] = g["e_combined"].apply(lambda s: (s > 0).mean()).values
    pah["mean_m_attack_norm_mean"] = g["m_attack_norm_mean"].mean().values
    pah["mean_attn_mass_A_attack"] = g["attn_mass_A_attack_mean"].mean().values
    pah = pah.merge(meta, on="attack_name")
    pah.to_csv(stage / "per_attack_head_all_query.csv", index=False)

    edge_glob, _ = equal_weight(aq, METRICS, labels)
    wh_glob, wh_pah = equal_weight(wh, WH_METRICS, labels)
    wh_pah.merge(meta, on="attack_name").to_csv(stage / "per_attack_head_whole_head.csv", index=False)

    glob = edge_glob.join(wh_glob.drop(columns="n_attacks"))
    stats_by_head = pah.groupby("head_label")
    glob["n_examples"] = aq.groupby("head_label").size().reindex(labels)
    glob["median_attack_mean_e_combined"] = stats_by_head["e_combined"].median().reindex(labels)
    glob["min_attack_mean_e_combined"] = stats_by_head["e_combined"].min().reindex(labels)
    glob["max_attack_mean_e_combined"] = stats_by_head["e_combined"].max().reindex(labels)
    glob["frac_attacks_mean_e_combined_gt0"] = stats_by_head["e_combined"].apply(lambda s: (s > 0).mean()).reindex(labels)
    lay = {h.label: (h.layer, h.head_idx) for h in heads}
    glob.insert(0, "layer", [lay[l][0] for l in labels])
    glob.insert(1, "head", [lay[l][1] for l in labels])
    glob.index.name = "head_label"
    glob.to_csv(stage / "global_head_summary.csv")

    # ---- (6) edge vs whole-head (SECONDARY) --------------------------------
    thr = float(cfg["comparison"]["exp11_importance_threshold"])
    rows = []
    for lab in labels:
        e, w = glob.loc[lab], glob.loc[lab]
        rows.append({
            "head_label": lab,
            "edge_combined": e.mean_e_combined, "whole_head_combined": w.mean_whole_head_combined,
            "edge_fwd": e.mean_e_fwd, "whole_head_fwd": w.mean_whole_head_fwd,
            "edge_rev": e.mean_e_rev, "whole_head_rev": w.mean_whole_head_rev,
            "ratio_combined": ratio(e.mean_e_combined, w.mean_whole_head_combined),
            "ratio_fwd": ratio(e.mean_e_fwd, w.mean_whole_head_fwd),
            "ratio_rev": ratio(e.mean_e_rev, w.mean_whole_head_rev),
            "ratio_computed": abs(w.mean_whole_head_combined) > SKIP_EPSILON,
            f"whole_head_combined_gt_{thr}": w.mean_whole_head_combined > thr,
        })
    pd.DataFrame(rows).to_csv(stage / "edge_vs_whole_head.csv", index=False)
    cmp_pa = pah[["head_label", "attack_name", "e_combined", "e_fwd", "e_rev"]].merge(
        wh_pah[["head_label", "attack_name", "whole_head_combined", "whole_head_fwd", "whole_head_rev"]],
        on=["head_label", "attack_name"])
    cmp_pa["ratio_combined"] = [ratio(a, b) for a, b in zip(cmp_pa.e_combined, cmp_pa.whole_head_combined)]
    cmp_pa.to_csv(stage / "edge_vs_whole_head_per_attack.csv", index=False)

    # ---- (7) descriptive single-token summaries ----------------------------
    sq["pos_combined"] = sq.e_combined > 0
    per_ex = sq.groupby(["attack_name", "example_id", "head_label"], sort=False)
    top_idx = per_ex["e_combined"].idxmax()
    top = sq.loc[top_idx, ["attack_name", "example_id", "head_label", "query_token_index",
                           "query_token_position", "query_token_string", "e_combined"]].rename(columns={
        "query_token_index": "top_query_token_index", "query_token_position": "top_query_token_position",
        "query_token_string": "top_query_token_string", "e_combined": "top_token_e_combined"})
    tpe = per_ex.agg(n_query_tokens=("e_combined", "size"),
                     frac_tokens_e_combined_gt0=("pos_combined", "mean"),
                     mean_token_e_combined=("e_combined", "mean"),
                     sum_single_raw_fwd=("raw_fwd", "sum"),
                     sum_single_raw_rev=("raw_rev", "sum")).reset_index()
    tpe = tpe.merge(top, on=["attack_name", "example_id", "head_label"])
    tpe = tpe.merge(aq[["attack_name", "example_id", "head_label", "raw_fwd", "raw_rev", "e_combined"]].rename(
        columns={"raw_fwd": "all_query_raw_fwd", "raw_rev": "all_query_raw_rev", "e_combined": "all_query_e_combined"}),
        on=["attack_name", "example_id", "head_label"])
    tpe["top_token_is_first_query_token"] = tpe.top_query_token_index == 0
    tpe["top_token_is_last_query_token"] = tpe.top_query_token_index == tpe.n_query_tokens - 1
    tpe.to_csv(stage / "token_per_example.csv", index=False)

    tsum = []
    for lab in labels:
        s = sq[sq.head_label == lab]
        t = tpe[tpe.head_label == lab]
        att = s.groupby("attack_name")["e_combined"].mean()
        tsum.append({
            "head_label": lab, "n_token_rows": len(s),
            "equal_weight_mean_token_e_combined": att.mean(),
            "equal_weight_mean_token_e_fwd": s.groupby("attack_name")["e_fwd"].mean().mean(),
            "equal_weight_mean_token_e_rev": s.groupby("attack_name")["e_rev"].mean().mean(),
            "pooled_median_token_e_combined": s.e_combined.median(),
            "pooled_q05_token_e_combined": s.e_combined.quantile(0.05),
            "pooled_q95_token_e_combined": s.e_combined.quantile(0.95),
            "pooled_frac_tokens_e_combined_gt0": s.pos_combined.mean(),
            "equal_weight_mean_frac_tokens_gt0_per_example": t.groupby("attack_name")["frac_tokens_e_combined_gt0"].mean().mean(),
            "equal_weight_mean_top_token_e_combined": t.groupby("attack_name")["top_token_e_combined"].mean().mean(),
            "frac_examples_top_token_first": t.top_token_is_first_query_token.mean(),
            "frac_examples_top_token_last": t.top_token_is_last_query_token.mean(),
            "equal_weight_mean_all_query_raw_fwd": t.groupby("attack_name")["all_query_raw_fwd"].mean().mean(),
            "equal_weight_mean_sum_single_raw_fwd": t.groupby("attack_name")["sum_single_raw_fwd"].mean().mean(),
            "equal_weight_mean_all_query_raw_rev": t.groupby("attack_name")["all_query_raw_rev"].mean().mean(),
            "equal_weight_mean_sum_single_raw_rev": t.groupby("attack_name")["sum_single_raw_rev"].mean().mean(),
        })
    pd.DataFrame(tsum).to_csv(stage / "token_summary_by_head.csv", index=False)

    print(glob[["layer", "head", "n_attacks", "n_examples", "mean_e_combined", "mean_e_fwd", "mean_e_rev",
                "mean_whole_head_combined"]].to_string())
    print(json.dumps(cons, indent=2))
    write_status(stage, {"status": "success", "finished": now(), "upstream_fingerprint": fp,
                         "manifest_sha256": sha, "head_labels": labels})


if __name__ == "__main__":
    main()
