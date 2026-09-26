#!/usr/bin/env python3
"""
scripts/01_sanity_checks.py
============================
Real-model sanity checks on a small subset (config `sanity:`), run BEFORE the
expensive stages. Any failure raises -> the pipeline stops.

  baseline   fresh live control/attack scores == cached Exp 01 scores (tol)
  A          z_h(q) == sum_j P[q,j] V[j]                (enforced in prepare_example)
  B          ||m_control(q)|| ~ 0 ; m_attack reported (must be non-zero somewhere)
  C          no-op donor (donor == receiver) leaves the score unchanged
  D/E/F      patched `.o` input differs from the receiver's ONLY at
             (target query positions, selected head slice) — bitwise elsewhere
  G/H        at those entries z_new == z_recv - m_recv + m_don (fwd: control
             receiver / attack donor; rev: attack receiver / control donor)
  M          edge engine with A = Q = ALL positions reproduces Exp 11's
             whole-head patch (exp15lib/whole_head.py) for fwd and rev
  N          optimized batched engine == unoptimized live-recompute reference
             (exp15lib/reference.py) for all-query and single-token rows
  batching   scores independent of max_rows_per_batch (1 vs configured)
  additivity (descriptive) all-query raw effect vs sum of single-token raw
             effects — recorded to show they are distinct quantities

Outputs: outputs/01_sanity/sanity_report.json, status.json
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

import torch  # noqa: E402

from exp11lib.head_hooks import head_geometry  # noqa: E402

from exp15lib.config import load_config, output_dir  # noqa: E402
from exp15lib.engine import (  # noqa: E402
    DIRECTIONS, EdgeRow, all_query_rows, prepare_example, run_edge_rows, single_query_rows,
)
from exp15lib.heads import heads_by_label  # noqa: E402
from exp15lib.reference import reference_patched_score  # noqa: E402
from exp15lib.run_utils import load_manifest, load_model, now, unit_is_done, write_status  # noqa: E402
from exp15lib.stage_runner import rebuild_encodings  # noqa: E402
from exp15lib.whole_head import whole_head_scores  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 15 stage 01: sanity checks")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, passed, **detail):
        self.items.append({"check": name, "passed": bool(passed), **detail})
        if not passed:
            print(f"  FAIL {name}: {detail}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    manifest, sha = load_manifest(out)
    stage = out / "01_sanity"
    s_cfg = cfg["sanity"]
    labels = list(s_cfg["heads"])
    if unit_is_done(stage, ["sanity_report.json"], sha, labels, args.force):
        print("[01] sanity checks already passed for this manifest; skipping (use --force to rerun)")
        return
    write_status(stage, {"status": "running", "started": now()})

    tol = cfg["tolerances"]
    model, tok, tid, fid, device = load_model(cfg)
    _, d_kv, _ = head_geometry(model)
    heads = heads_by_label(cfg, labels)
    hl = [(h.layer, h.head_idx) for h in heads]
    max_rows = int(cfg["runtime"]["max_rows_per_batch"])
    ck = Checks()
    summary = {"examples": []}

    for attack in s_cfg["attacks"]:
        recs = sorted([r for r in manifest if r["attack_name"] == attack], key=lambda r: r["sample_index"])
        if not recs:
            raise RuntimeError(f"sanity attack {attack} not in manifest")
        for rec in recs[: int(s_cfg["n_examples_per_attack"])]:
            tag = f"{attack}/{rec['example_id']}"
            c, a = rebuild_encodings(tok, rec, int(cfg["model"]["max_length"]), device)
            A, Q = rec["attack_source_positions"], rec["query_target_positions"]
            a_tokens = tok.convert_ids_to_tokens(a["input_ids"][0, A].tolist())
            print(f"[01] {tag}: Q={Q} {rec['query_token_strings']}  A={A} {a_tokens}")
            st = prepare_example(model, c, a, A, Q, hl, tid, fid, tol)   # enforces A + B
            ex_sum = {"tag": tag, "attack_tokens_at_A": a_tokens, "query_tokens": rec["query_token_strings"],
                      "diagnostics": {f"L{L}H{h}": d for (L, h), d in st.diagnostics.items()}}

            # baseline: fresh vs cached
            for role, key in (("control", "score_control"), ("attack", "score_attack")):
                d = abs(st.live_scores[role] - rec[key])
                ck.add("baseline_fresh_vs_cached", d <= tol["fresh_vs_cached_score_atol"],
                       tag=tag, role=role, abs_diff=d)
            ck.add("A_decomposition", max(max(v["decomp_err_control"], v["decomp_err_attack"])
                                          for v in st.diagnostics.values()) <= tol["decomposition_atol"], tag=tag)
            ck.add("B_control_message_zero",
                   max(v["m_control_norm_max"] for v in st.diagnostics.values()) <= tol["control_message_max_norm"],
                   tag=tag, max_norm=max(v["m_control_norm_max"] for v in st.diagnostics.values()))
            ck.add("B_attack_message_nonzero",
                   max(v["m_attack_norm_max"] for v in st.diagnostics.values()) > 0.0,
                   tag=tag, max_norm=max(v["m_attack_norm_max"] for v in st.diagnostics.values()))

            # C: no-op donor
            nq = len(Q)
            noop = [EdgeRow(d, L, h, list(range(nq))) for (L, h) in hl for d in ("noop_control", "noop_attack")]
            sc, _ = run_edge_rows(model, st, noop, tid, fid, max_rows)
            for r, s in zip(noop, sc):
                recv = DIRECTIONS[r.direction][0]
                d = abs(s - st.live_scores[recv])
                ck.add("C_noop_donor", d <= tol["noop_score_atol"], tag=tag, row=r.direction,
                       head=f"L{r.layer}H{r.head}", abs_diff=d)

            for (L, h) in hl:
                sl = slice(h * d_kv, (h + 1) * d_kv)
                rows = [EdgeRow("fwd", L, h, list(range(nq))), EdgeRow("rev", L, h, list(range(nq))),
                        EdgeRow("fwd", L, h, [0]), EdgeRow("rev", L, h, [nq - 1])]
                sc, cap = run_edge_rows(model, st, rows, tid, fid, max_rows, capture_layers=[L])
                for i, r in enumerate(rows):
                    recv, don = DIRECTIONS[r.direction]
                    z = cap[L]["pre"][i]            # unpatched .o input from the SAME batched pass
                    got = cap[L]["post"][i]
                    ck.add("DEF_same_pass_matches_receiver_capture",
                           (z - st.caches[recv][L]["z"]).abs().max().item() <= 1e-4, tag=tag,
                           head=f"L{L}H{h}", row=r.direction,
                           max_abs_diff=(z - st.caches[recv][L]["z"]).abs().max().item())
                    tpos = [Q[j] for j in r.target_q_idx]
                    changed = torch.zeros_like(z, dtype=torch.bool)
                    changed[tpos, sl] = True
                    ck.add("DEF_isolation_bitwise", bool(torch.equal(got[~changed], z[~changed])),
                           tag=tag, head=f"L{L}H{h}", row=r.direction, n_targets=len(tpos))
                    idx = torch.tensor(r.target_q_idx)
                    expect = z[tpos][:, sl] - st.msgs[(recv, L, h)][idx] + st.msgs[(don, L, h)][idx]
                    err = (got[tpos][:, sl] - expect).abs().max().item()
                    ck.add("GH_formula", err <= 1e-6, tag=tag, head=f"L{L}H{h}", row=r.direction, max_err=err,
                           receiver=recv, donor=don)
                    # N: reference implementation
                    ref = reference_patched_score(model, st.encs, r.direction, L, h, A, tpos, tid, fid)
                    d = abs(ref - sc[i])
                    ck.add("N_optimized_vs_reference", d <= tol["reference_score_atol"], tag=tag,
                           head=f"L{L}H{h}", row=r.direction, n_targets=len(tpos), abs_diff=d)

                # additivity (descriptive only)
                srows = single_query_rows([(L, h)], nq)
                ss, _ = run_edge_rows(model, st, srows, tid, fid, max_rows)
                sum_fwd = sum(s - rec["score_control"] for r, s in zip(srows, ss) if r.direction == "fwd")
                sum_rev = sum(rec["score_attack"] - s for r, s in zip(srows, ss) if r.direction == "rev")
                ex_sum.setdefault("additivity", {})[f"L{L}H{h}"] = {
                    "all_query_raw_fwd": sc[0] - rec["score_control"], "sum_single_raw_fwd": sum_fwd,
                    "all_query_raw_rev": rec["score_attack"] - sc[1], "sum_single_raw_rev": sum_rev,
                }

            # batching invariance
            aq = all_query_rows(hl, nq)
            s_big, _ = run_edge_rows(model, st, aq, tid, fid, max_rows)
            s_one, _ = run_edge_rows(model, st, aq, tid, fid, 1)
            d = max(abs(x - y) for x, y in zip(s_big, s_one))
            ck.add("batching_invariance", d <= tol["reference_score_atol"], tag=tag, max_abs_diff=d)

            # M: whole-head equivalence (A = Q = all positions)
            T = rec["seq_len"]
            st_all = prepare_example(model, c, a, list(range(T)), list(range(T)), hl, tid, fid, tol,
                                     check_control_zero=False)
            s_all, _ = run_edge_rows(model, st_all, all_query_rows(hl, T), tid, fid, max_rows)
            wh, live_c, live_a = whole_head_scores(model, c, a, hl, device, tid, fid)
            for r, s in zip(all_query_rows(hl, T), s_all):
                ref = wh[(r.layer, r.head)][0 if r.direction == "fwd" else 1]
                d = abs(s - ref)
                ck.add("M_whole_head_equivalence", d <= tol["reference_score_atol"], tag=tag,
                       head=f"L{r.layer}H{r.head}", row=r.direction, abs_diff=d)
            ck.add("M_whole_head_live_baselines",
                   abs(live_c - st.live_scores["control"]) <= 1e-5 and abs(live_a - st.live_scores["attack"]) <= 1e-5,
                   tag=tag)
            summary["examples"].append(ex_sum)

    n_fail = sum(not c["passed"] for c in ck.items)
    by_check = {}
    for c in ck.items:
        b = by_check.setdefault(c["check"], {"n": 0, "n_failed": 0})
        b["n"] += 1
        b["n_failed"] += int(not c["passed"])
    for name, key in (("baseline_fresh_vs_cached", "abs_diff"), ("N_optimized_vs_reference", "abs_diff"),
                      ("M_whole_head_equivalence", "abs_diff"), ("C_noop_donor", "abs_diff"),
                      ("GH_formula", "max_err"), ("DEF_same_pass_matches_receiver_capture", "max_abs_diff"), ("B_control_message_zero", "max_norm"),
                      ("B_attack_message_nonzero", "max_norm")):
        vals = [c[key] for c in ck.items if c["check"] == name]
        if vals:
            by_check[name]["max_" + key] = max(vals)
            by_check[name]["min_" + key] = min(vals)
    report = {"finished": now(), "n_checks": len(ck.items), "n_failed": n_fail, "by_check": by_check,
              "tolerances": tol, "summary": summary, "checks": ck.items}
    stage.mkdir(parents=True, exist_ok=True)
    with open(stage / "sanity_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(by_check, indent=2))
    if n_fail:
        write_status(stage, {"status": "failed", "finished": now(), "n_failed": n_fail})
        raise SystemExit(f"[01] {n_fail} sanity checks FAILED — see {stage / 'sanity_report.json'}")
    write_status(stage, {"status": "success", "finished": now(), "n_checks": len(ck.items),
                         "manifest_sha256": sha, "head_labels": labels})
    print(f"[01] all {len(ck.items)} sanity checks passed")


if __name__ == "__main__":
    main()
