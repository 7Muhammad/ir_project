"""
exp12lib/important_heads.py
=============================
Loads the already-identified causally important encoder heads from
Experiment 11 (Section 4.5), read-only. Does NOT perform any new head
selection -- attention is descriptive evidence only in this experiment
(see the experiment prompt's "IMPORTANT-HEAD LIST" section).

Source: ../11_monot5_encoder_head_patching/outputs/top10_heads.csv
    (written by Experiment 11's scripts/05_aggregate_and_plot.py: the top 10
    of the 144 encoder head-slots by combined_effect_mean, canonical attack
    relevant_start_5, n=100 — see that experiment's report Section 4.5.)
"""

from __future__ import annotations

import csv
import pathlib
import re
from dataclasses import dataclass
from typing import List

_HEAD_RE = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")


@dataclass(frozen=True)
class ImportantHead:
    layer: int
    head: int
    combined_effect_mean: float

    @property
    def label(self) -> str:
        return f"L{self.layer}H{self.head}"


def load_important_heads(csv_path: pathlib.Path, top_n: int = None) -> List[ImportantHead]:
    """
    Parse Experiment 11's top10_heads.csv ("head" column formatted "L{layer}H{head}").

    Parameters
    ----------
    csv_path : path to top10_heads.csv (read-only; never modified).
    top_n : if given, keep only the first `top_n` rows (file is already
        sorted by combined_effect_mean descending, per Experiment 11).

    Raises
    ------
    FileNotFoundError if csv_path does not exist (Experiment 11 must have
        been run first -- this experiment does not fall back to re-deriving
        head importance).
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Experiment 11's important-head list not found at {csv_path}. "
            "Run Experiment 11 (scripts/05_aggregate_and_plot.py) first -- "
            "Experiment 12 reuses its head selection and does not re-derive it."
        )
    heads: List[ImportantHead] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            m = _HEAD_RE.match(row["head"].strip())
            if not m:
                raise ValueError(f"Could not parse head label {row['head']!r} in {csv_path}.")
            heads.append(ImportantHead(
                layer=int(m.group("layer")),
                head=int(m.group("head")),
                combined_effect_mean=float(row["combined_effect_mean"]),
            ))
    if top_n is not None:
        heads = heads[:top_n]
    return heads
