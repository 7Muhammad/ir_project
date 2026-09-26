"""
exp12lib/causal_engine.py
===========================
Part B (causal): whole-encoder-self-attention-output patching at an
arbitrary named set of token-index spans (one query word, or one structural
control group), for one example, across requested layers.

Reused, unchanged (not duplicated)
------------------------------------
  exp6lib.engine.make_positional_patch_hook -- the forward hook that blends,
      at masked positions, the OTHER run's cached contribution with, at
      unmasked positions, the CURRENT run's own contribution. Already
      generic over "which positions get patched" (any (1, seq_len, 1) mask).
  exp6lib.template_engine.build_template_position_mask -- builds that mask
      from an arbitrary list of token indices (already used by Experiment 6's
      template-position extension for single/grouped positions).
  src.activation_hooks.cache_all_activations_from_enc, get_module_for_component
  src.model_utils.score_from_encoding
  src.patching.SKIP_EPSILON

What is new here
------------------
Experiment 6's engine.py loops over 3 named query/doc CONDITIONS;
template_engine.py loops over the 7 fixed TEMPLATE_POSITION_NAMES. This
module loops over an arbitrary caller-supplied `units` dict (query words,
one at a time -- or the 9 structural control groups), so the same function
serves both Part B's main causal result and the structural-control sanity
checks, without a second copy of the patch/measure loop.

Every intervention here is at ENCODER SELF-ATTENTION only (whole-block, not
per-head -- per-head is Experiment 11's job; per-position-but-single-head is
out of scope per the experiment prompt: "Do NOT patch individual heads
here").
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from src.activation_hooks import cache_all_activations_from_enc, get_module_for_component
from src.model_utils import score_from_encoding
from src.patching import SKIP_EPSILON

from exp6lib.engine import make_positional_patch_hook
from exp6lib.template_engine import build_template_position_mask


def run_causal_patch_units_example(
    model: nn.Module,
    control_enc: Dict,
    attack_enc: Dict,
    layers: List[int],
    units: Dict[str, List[int]],
    true_id: int,
    false_id: int,
    device,
) -> Optional[Tuple[List[Dict], float, float]]:
    """
    Single-span causal patching (encoder self-attention, whole-block
    granularity) for one example, across all requested layers and all
    named units in `units`, one unit at a time (never combined).

    Forward: control base, inject attack's contribution at the unit's
        positions -> e_fwd = (patched - control) / (attack - control).
    Reverse: attack base, inject control's contribution at the unit's
        positions -> e_rev = (attack - patched) / (attack - control).
    combined = min(e_fwd, e_rev)  -- same convention as the report throughout.

    Returns None if |attack_score - control_score| < SKIP_EPSILON (the
    example is not a "successful" attack instance -- same skip guard as
    every other patching experiment in this project).

    Returns
    -------
    (rows, control_score, attack_score) on success.
    rows : one dict per (layer, unit_name):
        {"layer", "unit_name", "n_tokens",
         "score_control", "score_attack",
         "score_patched_fwd", "score_patched_rev",
         "fwd_effect", "rev_effect", "combined_effect"}
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

        for unit_name, indices in units.items():
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
                "layer": layer_idx,
                "unit_name": unit_name,
                "n_tokens": len(indices),
                "score_control": control_score,
                "score_attack": attack_score,
                "score_patched_fwd": fwd_patched,
                "score_patched_rev": rev_patched,
                "fwd_effect": fwd_effect,
                "rev_effect": rev_effect,
                "combined_effect": min(fwd_effect, rev_effect),
            })

    return rows, control_score, attack_score
