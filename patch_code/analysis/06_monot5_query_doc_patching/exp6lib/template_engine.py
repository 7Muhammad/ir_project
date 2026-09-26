"""
exp6lib/template_engine.py
============================
Experiment 6 extension, Step 1 -- causal single-position patching for the 7
template positions (exp6lib/template_positions.py) that §4.4's query/
document positional patching always excludes.

Reused, unchanged, from exp6lib.engine (not duplicated):
  - make_positional_patch_hook -- the forward hook is already generic over
    "which positions get patched": it only needs a (1, seq_len, 1) mask, not
    a condition name. build_position_mask (engine.py) builds that mask from
    query_span/doc_span; build_template_position_mask below builds the same
    shape of mask from an arbitrary list of token indices instead.
  - score_from_encoding, cache_all_activations_from_enc, get_module_for_component
    (via src.activation_hooks / src.model_utils, same as engine.py)
  - SKIP_EPSILON (src.patching) -- same degenerate-example skip guard.

What is NEW here
------------------
engine.py's run_encoder_self_attn_example loops over the 3 query/doc
CONDITIONS. This module loops over the 7 TEMPLATE_POSITION_NAMES instead,
one at a time (never combined), at encoder self-attention only -- decoder
cross-attention is out of scope for this extension (the experiment prompt's
Step 1 only asks about encoder layers, and exp6's own report already found
decoder cross-attention immune to the early-layer instability this
extension is diagnosing).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.activation_hooks import cache_all_activations_from_enc, get_module_for_component
from src.model_utils import score_from_encoding
from src.patching import SKIP_EPSILON

from exp6lib.engine import make_positional_patch_hook
from exp6lib.template_positions import TEMPLATE_POSITION_NAMES


def build_template_position_mask(seq_len: int, indices: List[int], device: torch.device) -> torch.Tensor:
    """Build a (1, seq_len, 1) float mask, 1.0 at `indices`, 0.0 elsewhere."""
    mask = torch.zeros(1, seq_len, 1, device=device)
    for i in indices:
        mask[0, i, 0] = 1.0
    return mask


def run_template_position_causal_patch_example(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    layers: List[int],
    template_positions: Dict[str, List[int]],
    true_id: int,
    false_id: int,
    device: torch.device,
) -> Optional[List[Dict]]:
    """
    Single-position causal patching (encoder self-attention, whole-block
    granularity) for one example, across all requested layers and all 7
    template positions, one position at a time.

    Same forward/reverse/combined definitions as exp6lib.engine's query/doc
    conditions (Eq. 4-6 in the report): forward patches control->attack at
    the target position and measures sufficiency; reverse patches
    attack->control and measures necessity; combined = min(forward, reverse).
    """
    control_score = score_from_encoding(model, control_enc, true_id, false_id, device)
    attack_score = score_from_encoding(model, attack_enc, true_id, false_id, device)
    delta = attack_score - control_score
    if abs(delta) < SKIP_EPSILON:
        return None

    control_cache = cache_all_activations_from_enc(model, control_enc, device)
    attack_cache = cache_all_activations_from_enc(model, attack_enc, device)

    seq_len = control_enc["input_ids"].shape[1]
    rows: List[Dict] = []

    for layer_idx in layers:
        key = f"encoder_self_attn_layer{layer_idx}"
        module = get_module_for_component(model, "encoder_self_attn", layer_idx)
        ctrl_contrib = control_cache[key].to(device)
        atk_contrib = attack_cache[key].to(device)

        for position_name in TEMPLATE_POSITION_NAMES:
            indices = template_positions[position_name]
            mask = build_template_position_mask(seq_len, indices, device)

            handle = module.register_forward_hook(make_positional_patch_hook(atk_contrib, mask))
            try:
                fwd_patched = score_from_encoding(model, control_enc, true_id, false_id, device)
            finally:
                handle.remove()
            fwd_effect = (fwd_patched - control_score) / delta

            handle = module.register_forward_hook(make_positional_patch_hook(ctrl_contrib, mask))
            try:
                rev_patched = score_from_encoding(model, attack_enc, true_id, false_id, device)
            finally:
                handle.remove()
            rev_effect = (attack_score - rev_patched) / delta

            rows.append({
                "region": "encoder_self_attn",
                "layer": layer_idx,
                "template_position": position_name,
                "n_tokens": len(indices),
                "score_control": control_score,
                "score_attack": attack_score,
                "score_patched_fwd": fwd_patched,
                "score_patched_rev": rev_patched,
                "fwd_effect": fwd_effect,
                "rev_effect": rev_effect,
                "combined_effect": min(fwd_effect, rev_effect),
            })
    return rows
