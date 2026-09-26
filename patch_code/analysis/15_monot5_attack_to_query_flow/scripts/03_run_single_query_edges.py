#!/usr/bin/env python3
"""
scripts/03_run_single_query_edges.py
=====================================
DESCRIPTIVE condition: same examples, same 18 heads, forward + reverse, but
the attack-source message is replaced at exactly ONE query model-token
position q (all other query positions keep the receiver's own message).
Every query token position of every example is run separately.

Rows of different (head, q, direction) share one batched pass (exact: rows
are independent runs of same-length inputs, see exp15lib/engine.py).

Output per attack: outputs/03_single_query/per_attack/{attack}/rows.csv.gz
(one row per example x head x query position).
"""

from __future__ import annotations

import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401  (adds sibling experiment dirs to sys.path)

from exp15lib.engine import prepare_example, run_edge_rows, single_query_rows  # noqa: E402
from exp15lib.metrics import recovery_metrics  # noqa: E402
from exp15lib.stage_runner import run_stage, stage_argparser  # noqa: E402


def per_example(ctx, rec, control_enc, attack_enc):
    cfg, heads = ctx["cfg"], ctx["heads"]
    hl = [(h.layer, h.head_idx) for h in heads]
    st = prepare_example(ctx["model"], control_enc, attack_enc, rec["attack_source_positions"],
                         rec["query_target_positions"], hl, ctx["true_id"], ctx["false_id"], cfg["tolerances"])
    rows = single_query_rows(hl, len(st.Q))
    scores, _ = run_edge_rows(ctx["model"], st, rows, ctx["true_id"], ctx["false_id"],
                              int(cfg["runtime"]["max_rows_per_batch"]))
    by = {(r.layer, r.head, r.target_q_idx[0], r.direction): s for r, s in zip(rows, scores)}
    out = []
    for h in heads:
        L, hh = h.layer, h.head_idx
        m_atk = st.msgs[("attack", L, hh)].norm(dim=-1)
        for i, q in enumerate(st.Q):
            s_f, s_r = by[(L, hh, i, "fwd")], by[(L, hh, i, "rev")]
            m = recovery_metrics(rec["score_control"], rec["score_attack"], s_f, s_r)
            m.pop("delta")
            out.append({
                "head_label": h.label, "layer": L, "head": hh,
                "query_token_index": i,
                "query_token_position": q,
                "query_token_id": rec["query_token_ids"][i],
                "query_token_string": rec["query_token_strings"][i],
                "score_forward_patched": s_f, "score_reverse_patched": s_r,
                **m,
                "m_attack_norm_q": m_atk[i].item(),
            })
    return out


if __name__ == "__main__":
    args = stage_argparser("Exp 15 stage 03: single-query-token edge patching", EXP_DIR).parse_args()
    run_stage(args, "03_single_query", per_example)
