"""
exp9lib/targets.py
=====================
Load Experiment 1's per-attack score pool (up to 500 pairs, already scored —
zero new forward passes needed). This is the target set for score-vs-rank
comparison: NO filtering, every available example is used (per task spec,
to avoid survivorship bias).
"""

from __future__ import annotations

import csv
import pathlib
from typing import Dict, List


def load_targets(attack_name: str, exp1_outputs_dir: pathlib.Path) -> List[Dict]:
    """
    Read `{exp1_outputs_dir}/{attack_name}/scores/all_scores.csv`.

    Returns a list of dicts with (at least) qid, docid, control_score,
    attack_score — the full up-to-500-pair pool for that attack, unfiltered.
    """
    path = exp1_outputs_dir / attack_name / "scores" / "all_scores.csv"
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["control_score"] = float(r["control_score"])
        r["attack_score"] = float(r["attack_score"])
    return rows
