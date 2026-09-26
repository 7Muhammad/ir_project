"""
exp14lib/scale_selection.py
==============================
Shared "pick the best scale per candidate from a validation recovery
dataframe" logic (task spec section 8: highest mean recovery, ties -> the
smaller scale). Used by both the main IID scale-selection stage
(scripts/05_select_scales.py) and every attack-OOD fold
(scripts/07_run_attack_ood.py), so the selection rule exists in one place.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

CANDIDATE_COLS = ["side", "layer", "head_idx", "condition"]
COMPONENT_BY_SIDE = {"decoder": "decoder_cross_attn", "encoder": "encoder_self_attn"}


def select_scales(validation_df: pd.DataFrame) -> Dict[str, dict]:
    """
    validation_df: rows with columns side, layer, head_idx, condition,
    scale, recovery (one row per example x candidate x scale — see
    exp14lib.engine.compute_defense_rows_for_examples).

    Returns {candidate_key: {side, component, layer, head_idx, condition,
    selected_scale, mean_recovery_at_selected_scale, n_scale_candidates_compared}}.
    """
    recov_by_scale = (
        validation_df.groupby(CANDIDATE_COLS + ["scale"])["recovery"]
        .mean()
        .reset_index()
        .rename(columns={"recovery": "mean_recovery"})
    )

    selected: Dict[str, dict] = {}
    for key_cols, group in recov_by_scale.groupby(CANDIDATE_COLS):
        side, layer, head_idx, condition = key_cols
        best = group.sort_values(["mean_recovery", "scale"], ascending=[False, True]).iloc[0]
        key = f"{side}:{int(layer)}:{int(head_idx)}:{condition}"
        selected[key] = {
            "side": side, "component": COMPONENT_BY_SIDE[side], "layer": int(layer),
            "head_idx": int(head_idx), "condition": condition,
            "selected_scale": float(best["scale"]), "mean_recovery_at_selected_scale": float(best["mean_recovery"]),
            "n_scale_candidates_compared": len(group),
        }
    return selected
