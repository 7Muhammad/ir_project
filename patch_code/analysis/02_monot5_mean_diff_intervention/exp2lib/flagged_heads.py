"""
exp2lib/flagged_heads.py
==========================
Derive the Part-2 intervention scope from Experiment 3's grid_a aggregate.

No pre-existing "flagged heads" artifact exists in Experiment 3 (its README
explicitly defers this). Per user decision (2026-07-12), the scope is:
heads with combined_effect_mean > threshold in
03_monot5_head_patching_ablation/outputs/grid_a/aggregated/head_summary.csv
(pooled over all 105 attacks) — 22 heads at threshold 0.02, of which 21 are
decoder_cross_attn and 1 is decoder_self_attn (layer 11, head 3).
"""

from __future__ import annotations

import csv
import pathlib
from typing import Dict, List


def load_flagged_heads(head_summary_csv: pathlib.Path, threshold: float) -> List[Dict]:
    """
    Read Experiment 3's per-head aggregate and return heads whose
    combined_effect_mean exceeds `threshold`, sorted descending.

    Each returned dict has: layer (int), component (str), head_idx (int),
    combined_effect_mean (float).
    """
    flagged = []
    with open(head_summary_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            effect = float(row["combined_effect_mean"])
            if effect > threshold:
                flagged.append({
                    "layer": int(row["layer"]),
                    "component": row["component"],
                    "head_idx": int(row["head_idx"]),
                    "combined_effect_mean": effect,
                })
    flagged.sort(key=lambda r: -r["combined_effect_mean"])
    return flagged
