#!/usr/bin/env python3
"""
scripts/04_run_whole_head_reference.py
=======================================
Whole-head reference: ordinary Exp 11 encoder head patching (head h's whole
`.o`-input slice at EVERY position swapped between control and attack runs)
for the same 18 heads on EXACTLY the Exp 15 sample manifest, normalised with
the same cached Exp 01 baselines. Implementation: exp11lib.engine via
exp15lib/whole_head.py.

Output per attack: outputs/04_whole_head/per_attack/{attack}/rows.csv.gz
"""

from __future__ import annotations

import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401  (adds sibling experiment dirs to sys.path)

from exp15lib.metrics import recovery_metrics  # noqa: E402
from exp15lib.stage_runner import run_stage, stage_argparser  # noqa: E402
from exp15lib.whole_head import whole_head_scores  # noqa: E402


def per_example(ctx, rec, control_enc, attack_enc):
    heads = ctx["heads"]
    scores, live_c, live_a = whole_head_scores(
        ctx["model"], control_enc, attack_enc, [(h.layer, h.head_idx) for h in heads],
        ctx["device"], ctx["true_id"], ctx["false_id"])
    out = []
    for h in heads:
        s_f, s_r = scores[(h.layer, h.head_idx)]
        m = recovery_metrics(rec["score_control"], rec["score_attack"], s_f, s_r)
        out.append({
            "head_label": h.label, "layer": h.layer, "head": h.head_idx,
            "score_whole_head_fwd_patched": s_f, "score_whole_head_rev_patched": s_r,
            "whole_head_raw_fwd": m["raw_fwd"], "whole_head_raw_rev": m["raw_rev"],
            "whole_head_fwd": m["e_fwd"], "whole_head_rev": m["e_rev"],
            "whole_head_combined": m["e_combined"],
            "live_score_control": live_c, "live_score_attack": live_a,
        })
    return out


if __name__ == "__main__":
    args = stage_argparser("Exp 15 stage 04: whole-head reference patching", EXP_DIR).parse_args()
    run_stage(args, "04_whole_head", per_example)
