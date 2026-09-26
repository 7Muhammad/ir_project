"""
exp16lib/aggregate.py
======================
Two-level equal-weight aggregation and checkpoint-to-checkpoint increments.

  attacks : mean over examples WITHIN each attack, then equal weight per attack
  qrels   : mean over documents WITHIN each (query, group), then equal weight
            per query

delta_step(c) = x(c) - x(c-1) for c >= 1 (undefined at the embedding
checkpoint, c = 0). For per-unit trajectories the step of the mean equals the
mean of the steps, so steps can be taken on either level.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

KEY = ["checkpoint_index", "checkpoint_name", "layer", "sublayer"]


def add_steps(df: pd.DataFrame, value_cols: List[str], unit_cols: List[str]) -> pd.DataFrame:
    """Adds `step_<col>` = col(c) - col(c-1) within each unit (NaN at c=0)."""
    df = df.sort_values(unit_cols + ["checkpoint_index"]).copy()
    for col in value_cols:
        df[f"step_{col}"] = df.groupby(unit_cols, sort=False)[col].diff() if unit_cols else df[col].diff()
    return df


def two_level_mean(df: pd.DataFrame, unit_col: str, value_cols: List[str]):
    """
    Returns (per_unit, global): per_unit = mean over rows within (unit, checkpoint),
    with n_rows; global = equal-weight mean over units + SD / SE across units.
    """
    per_unit = df.groupby([unit_col] + KEY, sort=True)[value_cols].mean().reset_index()
    per_unit["n_rows"] = df.groupby([unit_col] + KEY, sort=True).size().values
    g = per_unit.groupby(KEY, sort=True)[value_cols]
    glob = g.mean().reset_index()
    sd = g.std(ddof=1).reset_index()
    n_units = per_unit.groupby(KEY, sort=True)[unit_col].nunique().values
    for c in value_cols:
        glob[f"{c}_sd_units"] = sd[c].values
        glob[f"{c}_se_units"] = sd[c].values / np.sqrt(n_units)
    glob["n_units"] = n_units
    return per_unit, glob
