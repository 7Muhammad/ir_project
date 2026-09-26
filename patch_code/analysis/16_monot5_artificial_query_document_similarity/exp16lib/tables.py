"""
exp16lib/tables.py
===================
Long-format (example x checkpoint) result tables.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from exp16lib.checkpoints import checkpoint_table


def long_table(meta: List[Dict], values: Dict[str, np.ndarray], n_layers: int = 12) -> pd.DataFrame:
    """
    meta: one dict per example; values: {column: [n_examples, n_checkpoints]}.
    Returns one row per (example, checkpoint) with checkpoint_index/name/layer/sublayer.
    """
    ck = pd.DataFrame(checkpoint_table(n_layers))
    n, C = len(meta), len(ck)
    for k, v in values.items():
        if v.shape != (n, C):
            raise ValueError(f"{k}: shape {v.shape} != ({n}, {C})")
    m = pd.DataFrame(meta).loc[np.repeat(np.arange(n), C)].reset_index(drop=True)
    c = pd.concat([ck] * n, ignore_index=True)
    df = pd.concat([m, c], axis=1)
    for k, v in values.items():
        df[k] = v.reshape(-1)
    return df
