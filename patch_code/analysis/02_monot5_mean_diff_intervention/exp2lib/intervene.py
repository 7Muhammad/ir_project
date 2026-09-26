"""
exp2lib/intervene.py
======================
Part 2: additive scale*direction shift interventions at Experiment-3-flagged
decoder heads (decoder_self_attn / decoder_cross_attn only — Part 2 never
touches encoder heads, since Experiment 3 flagged none).

Reuses Experiment 3's encoder-output-reuse speed trick
(headlib.engine.compute_encoder_states / decoder_pass_scores): the encoder
forward runs once per example and is reused across every flagged head and
every scale via decoder-only passes.

New hook (doesn't exist in headlib): make_head_shift_pre_hook is ADDITIVE
(new_hidden = hidden + sign * scale * direction on one head's slice) instead
of headlib.head_hooks.make_per_row_head_pre_hook's REPLACEMENT semantics.
Batches the 3 scales into one decoder pass per (head, example) — row i gets
shifted by scales[i], cutting decoder-only forward passes 3x versus one call
per scale.

2a — Defense (priority): base = attacked input, sign = -1 (subtract).
2b — Sufficiency: base = clean input, sign = +1 (add).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from headlib.engine import compute_encoder_states, decoder_pass_scores
from headlib.head_hooks import get_o_proj, head_geometry

from exp2lib.metrics import delta_toward_attack as _delta_toward_attack
from exp2lib.metrics import delta_toward_control as _delta_toward_control

DirKey = Tuple[str, str, int, Optional[int]]


def make_head_shift_pre_hook(
    direction: torch.Tensor,
    scales: List[float],
    sign: float,
    head_idx: int,
    d_kv: int,
) -> Callable:
    """
    Forward PRE-hook for a batch of len(scales) identical rows that adds
    ``sign * scales[i] * direction`` to row i's slice for `head_idx`, leaving
    every other head's slice (and every other row) untouched.

    Parameters
    ----------
    direction : Tensor (d_kv,), on the model device.
    scales : list of per-row scale multipliers (batch size = len(scales)).
    sign : +1.0 (add / sufficiency test) or -1.0 (subtract / defense test).
    head_idx : which head's slice to shift.
    d_kv : per-head dimension.
    """
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]  # (n_rows, 1, inner_dim)
        start = head_idx * d_kv
        end = start + d_kv
        shift = torch.zeros_like(hidden)
        for i, scale in enumerate(scales):
            shift[i, :, start:end] = sign * scale * direction.to(hidden.dtype)
        return (hidden + shift,) + args[1:]
    return hook


def head_scores_for_shift(
    model: nn.Module,
    enc_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    component: str,
    layer_idx: int,
    direction: torch.Tensor,
    scales: List[float],
    sign: float,
    head_idx: int,
    d_kv: int,
    true_id: int,
    false_id: int,
) -> List[float]:
    """Batched-over-scales decoder pass with head_idx shifted at every row."""
    handle = get_o_proj(model, component, layer_idx).register_forward_pre_hook(
        make_head_shift_pre_hook(direction, scales, sign, head_idx, d_kv)
    )
    try:
        scores = decoder_pass_scores(
            model, enc_hidden, enc_mask, true_id, false_id, len(scales)
        )
    finally:
        handle.remove()
    return scores.tolist()


def run_defense_example(
    model: nn.Module,
    attack_enc: Dict[str, torch.Tensor],
    flagged_heads: List[Dict],
    directions: Dict[DirKey, torch.Tensor],
    direction_norms: Dict[DirKey, float],
    scales: List[float],
    true_id: int,
    false_id: int,
    device: torch.device,
    meta: Dict,
) -> List[Dict]:
    """
    2a — Defense test. Subtract scale*direction from the ATTACKED activation
    at each flagged head; compare the resulting score to the control/attack
    baselines already known from `meta` (score_control, score_attack).
    """
    _, d_kv, _ = head_geometry(model)
    atk_hidden, atk_mask = compute_encoder_states(model, attack_enc, device)

    rows: List[Dict] = []
    for fh in flagged_heads:
        key = ("per_head", fh["component"], fh["layer"], fh["head_idx"])
        if key not in directions:
            continue
        direction = directions[key].to(device)
        mod_scores = head_scores_for_shift(
            model, atk_hidden, atk_mask, fh["component"], fh["layer"],
            direction, scales, sign=-1.0, head_idx=fh["head_idx"], d_kv=d_kv,
            true_id=true_id, false_id=false_id,
        )
        for scale, score_modified in zip(scales, mod_scores):
            delta_toward_control = _delta_toward_control(
                meta["score_attack"], meta["score_control"], score_modified
            )
            rows.append({
                "qid": meta["qid"], "docid": meta["docid"],
                "attack_name": meta["attack_name"],
                "layer": fh["layer"], "component": fh["component"],
                "head_idx": fh["head_idx"], "granularity": "per_head",
                "scale": scale, "intervention_type": "subtract",
                "score_clean": meta.get("score_clean"),
                "score_control": meta["score_control"],
                "score_attack": meta["score_attack"],
                "score_modified": score_modified,
                "delta_toward_control": delta_toward_control,
                "delta_toward_attack": "",
                "direction_norm": direction_norms.get(key),
            })
    return rows


def run_sufficiency_example(
    model: nn.Module,
    clean_enc: Dict[str, torch.Tensor],
    flagged_heads: List[Dict],
    directions: Dict[DirKey, torch.Tensor],
    direction_norms: Dict[DirKey, float],
    scales: List[float],
    true_id: int,
    false_id: int,
    device: torch.device,
    meta: Dict,
) -> List[Dict]:
    """
    2b — Sufficiency test. Add scale*direction to the CLEAN activation at
    each flagged head; compare the resulting score to the clean/attack
    baselines already known from `meta` (score_clean, score_attack).
    """
    _, d_kv, _ = head_geometry(model)
    clean_hidden, clean_mask = compute_encoder_states(model, clean_enc, device)

    rows: List[Dict] = []
    for fh in flagged_heads:
        key = ("per_head", fh["component"], fh["layer"], fh["head_idx"])
        if key not in directions:
            continue
        direction = directions[key].to(device)
        mod_scores = head_scores_for_shift(
            model, clean_hidden, clean_mask, fh["component"], fh["layer"],
            direction, scales, sign=1.0, head_idx=fh["head_idx"], d_kv=d_kv,
            true_id=true_id, false_id=false_id,
        )
        for scale, score_modified in zip(scales, mod_scores):
            delta_toward_attack = _delta_toward_attack(
                meta["score_clean"], meta["score_attack"], score_modified
            )
            rows.append({
                "qid": meta["qid"], "docid": meta["docid"],
                "attack_name": meta["attack_name"],
                "layer": fh["layer"], "component": fh["component"],
                "head_idx": fh["head_idx"], "granularity": "per_head",
                "scale": scale, "intervention_type": "add",
                "score_clean": meta["score_clean"],
                "score_control": meta.get("score_control"),
                "score_attack": meta["score_attack"],
                "score_modified": score_modified,
                "delta_toward_control": "",
                "delta_toward_attack": delta_toward_attack,
                "direction_norm": direction_norms.get(key),
            })
    return rows
