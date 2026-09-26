#!/usr/bin/env python3
"""
scripts/02_run_all_query_edges.py
==================================
PRIMARY condition: for every sampled example and each of the 18 heads,
replace the attack-source message at ALL query-text positions simultaneously
(one explicit intervention — never a sum of single-token runs):

  fwd (sufficiency): receiver = padded control, z <- z_ctrl - m_ctrl + m_atk
  rev (necessity)  : receiver = attacked run,   z <- z_atk  - m_atk  + m_ctrl

Output per attack: outputs/02_all_query/per_attack/{attack}/rows.csv.gz
(one row per example x head; resume-safe, see exp15lib/stage_runner.py).
"""

from __future__ import annotations

import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401  (adds sibling experiment dirs to sys.path)

from exp15lib.engine import all_query_rows, prepare_example, run_edge_rows  # noqa: E402
from exp15lib.metrics import recovery_metrics  # noqa: E402
from exp15lib.stage_runner import run_stage, stage_argparser  # noqa: E402


def per_example(ctx, rec, control_enc, attack_enc):
    cfg, heads = ctx["cfg"], ctx["heads"]
    hl = [(h.layer, h.head_idx) for h in heads]
    st = prepare_example(ctx["model"], control_enc, attack_enc, rec["attack_source_positions"],
                         rec["query_target_positions"], hl, ctx["true_id"], ctx["false_id"], cfg["tolerances"])
    rows = all_query_rows(hl, len(st.Q))
    scores, _ = run_edge_rows(ctx["model"], st, rows, ctx["true_id"], ctx["false_id"],
                              int(cfg["runtime"]["max_rows_per_batch"]))
    by = {(r.layer, r.head, r.direction): s for r, s in zip(rows, scores)}
    out = []
    for h in heads:
        key = (h.layer, h.head_idx)
        s_f, s_r = by[key + ("fwd",)], by[key + ("rev",)]
        m = recovery_metrics(rec["score_control"], rec["score_attack"], s_f, s_r)
        m.pop("delta")
        out.append({
            "query": rec["query"],
            "head_label": h.label, "layer": h.layer, "head": h.head_idx,
            "attack_source_positions": json.dumps(rec["attack_source_positions"]),
            "query_target_positions": json.dumps(rec["query_target_positions"]),
            "score_forward_patched": s_f, "score_reverse_patched": s_r,
            **m,
            "live_score_control": st.live_scores["control"], "live_score_attack": st.live_scores["attack"],
            **st.diagnostics[key],
        })
    return out


if __name__ == "__main__":
    args = stage_argparser("Exp 15 stage 02: all-query edge patching", EXP_DIR).parse_args()
    run_stage(args, "02_all_query", per_example)
