"""
exp15lib/metrics.py
====================
Forward / reverse / combined recovery — the project's definitions
(Exp 01 src/patching.py docstring Eq. 4-6; Exp 3/11 engines):

    delta      = score_attack - score_control            (cached Exp 01 scores)
    raw_fwd    = score_control_patched - score_control   (sufficiency, unnormalised)
    raw_rev    = score_attack - score_attack_patched     (necessity, unnormalised)
    e_fwd      = raw_fwd / delta
    e_rev      = raw_rev / delta
    e_combined = min(e_fwd, e_rev)

NEVER clipped: negative values and values > 1 are kept as-is.
"""

from __future__ import annotations

from typing import Dict

import exp15lib  # noqa: F401
from src.patching import SKIP_EPSILON


def recovery_metrics(
    score_control: float, score_attack: float,
    score_fwd_patched: float, score_rev_patched: float,
) -> Dict[str, float]:
    delta = score_attack - score_control
    if not delta > SKIP_EPSILON:
        raise ValueError(
            f"delta={delta} <= SKIP_EPSILON={SKIP_EPSILON}: example is not a successful instance "
            "and should never have entered the sample."
        )
    raw_fwd = score_fwd_patched - score_control
    raw_rev = score_attack - score_rev_patched
    e_fwd = raw_fwd / delta
    e_rev = raw_rev / delta
    return {
        "delta": delta,
        "raw_fwd": raw_fwd,
        "raw_rev": raw_rev,
        "e_fwd": e_fwd,
        "e_rev": e_rev,
        "e_combined": min(e_fwd, e_rev),
    }
