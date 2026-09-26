"""
src/patching.py
===============
Layer-level activation patching for the monoT5 adversarial ranking experiment.

What is activation patching?
------------------------------
Activation patching (causal tracing) answers:

    "If I swap the hidden state at layer L, component C from the ATTACK run
     into the PADDED-CONTROL run, how much of the attack's score increase
     do I recover?"

The two directions
-------------------
FORWARD patching
~~~~~~~~~~~~~~~~
  Base run    : padded control input (B)
  Patch source: attack activation at (layer L, component C)

  We run the padded-control input but at layer L / component C we replace
  the hidden state with the one computed during the attack run.

  patched_score = score(control_input WITH attacked activation at (L, C))

  forward_effect = (patched_score - control_score)
                 / (attack_score  - control_score)

  Interpretation: ≈1 means injecting the attack activation here alone is
  sufficient to recover the full score increase → this component CARRIES
  the attack signal.

REVERSE patching
~~~~~~~~~~~~~~~~
  Base run    : attacked input (C)
  Patch source: padded-control activation at (layer L, component C)

  We run the attacked input but replace one component's hidden state with
  the corresponding padded-control activation.

  patched_score = score(attack_input WITH padded-control activation at (L, C))

  reverse_effect = (attack_score - patched_score)
                 / (attack_score - control_score)

  Interpretation: ≈1 means removing the attack activation from that component
  alone cancels the full score increase → this component is NECESSARY.

Combined importance
--------------------
  combined = min(forward_effect, reverse_effect)

  High combined score → component is both sufficient AND necessary.

Normalisation denominator
--------------------------
  attack_delta_vs_control = attack_score - control_score

  This is the quantity activation patching tries to explain.  Both control
  and attack have the same sequence length and passage alignment, so the
  delta is purely due to what occupies the attack-prefix token positions:
  real "relevant:" tokens (attack) vs empty pad slots (control).

  If |attack_delta_vs_control| < SKIP_EPSILON, the attack has negligible
  effect on this example and normalisation would be unstable — we skip it.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model_utils import score_from_encoding
from src.activation_hooks import (
    COMPONENTS,
    get_module_for_component,
    make_patch_hook,
    n_decoder_layers,
    n_encoder_layers,
)

# Examples with |attack_delta| below this are skipped to avoid ÷0
SKIP_EPSILON = 1e-4


# ---------------------------------------------------------------------------
# Single (layer, component) patch — accepts pre-built encodings
# ---------------------------------------------------------------------------

def patch_and_score(
    model: nn.Module,
    base_enc: Dict[str, torch.Tensor],
    replacement_activation: torch.Tensor,
    component: str,
    layer_idx: int,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> float:
    """
    Run one forward pass with a patched activation and return the score.

    Procedure:
      1. Register a patch hook on (component, layer_idx) that will intercept
         the module's output and replace the hidden state with
         `replacement_activation`.
      2. Run the forward pass with `base_enc` (either control or attack enc).
      3. Extract logit("true") - logit("false") from decoder position 0.
      4. Remove the hook.

    WHY WE USE A HOOK
    -----------------
    Registering a temporary forward hook is cleaner than modifying model weights
    or writing a custom forward method:
      - No model parameters are changed.
      - The hook is removed after use — subsequent forward passes are unaffected.
      - Works with any T5 sub-module without subclassing.

    WHY base_enc MUST CARRY THE CORRECT attention_mask
    ---------------------------------------------------
    For the padded-control base run, attention_mask=0 for the pad slots.
    Even though we replace those positions' hidden states with attack
    activations (which were computed with mask=1 for real tokens), the
    base run's mask still governs which keys/queries are valid in downstream
    attention layers.  This is intentional: we are asking "what if the control
    input had the attacked representation at this one component" while keeping
    everything else (including the mask) from the control run.

    Parameters
    ----------
    model : T5ForConditionalGeneration (eval mode)
    base_enc : dict {"input_ids": Tensor(1, L), "attention_mask": Tensor(1, L)}
        The base input (padded control for forward patch; attack for reverse patch).
    replacement_activation : Tensor (1, seq_len, d_model), on CPU.
        Moved to `device` inside this function.
    component : str (one of COMPONENTS)
    layer_idx : int
    true_id, false_id : int
    device : torch.device

    Returns
    -------
    float : patched relevance score
    """
    replacement = replacement_activation.to(device)
    module = get_module_for_component(model, component, layer_idx)

    # Register patch hook — replaces this module's hidden state output
    handle = module.register_forward_hook(make_patch_hook(replacement))
    try:
        patched_score = score_from_encoding(model, base_enc, true_id, false_id, device)
    finally:
        # ALWAYS remove the hook even if score_from_encoding raises.
        # A leaked hook would corrupt all subsequent forward passes.
        handle.remove()

    return patched_score


# ---------------------------------------------------------------------------
# Full sweep over all (layer, component) pairs for one example
# ---------------------------------------------------------------------------

def run_layer_patching_for_example(
    example: Dict,
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc:  Dict[str, torch.Tensor],
    control_cache: Dict[str, torch.Tensor],
    attack_cache:  Dict[str, torch.Tensor],
    true_id: int,
    false_id: int,
    device: torch.device,
) -> Optional[List[Dict]]:
    """
    Run the full patching sweep for one (query, passage) example.

    For every (layer, component) pair we compute forward and reverse effects.

    WHY WE ACCEPT PRE-BUILT ENCODINGS
    ------------------------------------
    The padded-control encoding carries attention_mask=0 for the pad slots.
    We cannot reconstruct that mask from the text string — it must be built
    at the token level (see model_utils.build_padded_control_and_attack_encodings).
    The encodings are built in Stage 02 (cache_activations) and saved to
    disk alongside the activations, then loaded here.

    Parameters
    ----------
    example : scored pair dict (qid, docid, query, passage, control_score, attack_score)
    model : T5ForConditionalGeneration (eval mode)
    control_enc : padded-control encoding dict, tensors on CPU (moved to device inside)
    attack_enc  : attacked encoding dict, tensors on CPU
    control_cache : activations from the padded-control forward pass
    attack_cache  : activations from the attacked forward pass
    true_id, false_id : int
    device : torch.device

    Returns
    -------
    List of result dicts (one per layer × component × direction),
    or None if the example is skipped (negligible attack_delta_vs_control).
    """
    control_score = example["control_score"]
    attack_score  = example["attack_score"]
    # attack_delta_vs_control is what activation patching tries to explain:
    # the score increase caused by the attack tokens in those prefix positions.
    delta = attack_score - control_score

    if abs(delta) < SKIP_EPSILON:
        print(
            f"[patching] Skipping qid={example['qid']} docid={example['docid']}: "
            f"attack_delta_vs_control={delta:.6f} < epsilon={SKIP_EPSILON}"
        )
        return None

    n_enc = n_encoder_layers(model)
    n_dec = n_decoder_layers(model)
    results: List[Dict] = []

    # Helper to add one row to results
    def _record(component, layer_idx, direction, patched_score, effect):
        results.append({
            "qid":           example["qid"],
            "docid":         example["docid"],
            "layer":         layer_idx,
            "component":     component,
            "direction":     direction,
            "control_score": control_score,
            "attack_score":  attack_score,
            "patched_score": patched_score,
            "effect":        effect,
        })

    # ----- Encoder components -----
    for layer_idx in range(n_enc):
        for component in ["encoder_self_attn", "encoder_mlp"]:
            key = f"{component}_layer{layer_idx}"
            ctrl_act = control_cache[key]
            atk_act  = attack_cache[key]

            # Shape check: padded-control and attacked must have identical
            # activation shapes since they have the same sequence length.
            if ctrl_act.shape != atk_act.shape:
                raise ValueError(
                    f"Activation shape mismatch at {key}:\n"
                    f"  control shape: {ctrl_act.shape}\n"
                    f"  attack  shape: {atk_act.shape}\n"
                    "This should not happen if build_padded_control_and_attack_encodings "
                    "succeeded — both sequences have the same length by construction."
                )

            # FORWARD: run control base, inject attack activation
            fwd_patched = patch_and_score(
                model, control_enc, atk_act,
                component, layer_idx, true_id, false_id, device,
            )
            # How much of the attack's score increase does this component explain?
            # 0 = nothing, 1 = everything, >1 = over-recovery
            fwd_effect = (fwd_patched - control_score) / delta
            _record(component, layer_idx, "forward", fwd_patched, fwd_effect)

            # REVERSE: run attack base, inject control activation
            rev_patched = patch_and_score(
                model, attack_enc, ctrl_act,
                component, layer_idx, true_id, false_id, device,
            )
            # How much of the score increase is cancelled by removing this component?
            rev_effect = (attack_score - rev_patched) / delta
            _record(component, layer_idx, "reverse", rev_patched, rev_effect)

    # ----- Decoder components -----
    for layer_idx in range(n_dec):
        for component in ["decoder_self_attn", "decoder_cross_attn", "decoder_mlp"]:
            key = f"{component}_layer{layer_idx}"
            ctrl_act = control_cache[key]
            atk_act  = attack_cache[key]

            if ctrl_act.shape != atk_act.shape:
                raise ValueError(
                    f"Activation shape mismatch at {key}:\n"
                    f"  control shape: {ctrl_act.shape}\n"
                    f"  attack  shape: {atk_act.shape}\n"
                    "Both sequences must have the same length for patching."
                )

            fwd_patched = patch_and_score(
                model, control_enc, atk_act,
                component, layer_idx, true_id, false_id, device,
            )
            fwd_effect = (fwd_patched - control_score) / delta
            _record(component, layer_idx, "forward", fwd_patched, fwd_effect)

            rev_patched = patch_and_score(
                model, attack_enc, ctrl_act,
                component, layer_idx, true_id, false_id, device,
            )
            rev_effect = (attack_score - rev_patched) / delta
            _record(component, layer_idx, "reverse", rev_patched, rev_effect)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(detailed: List[Dict]) -> List[Dict]:
    """
    Aggregate per-example patching results into per-(layer, component, direction)
    statistics.

    Parameters
    ----------
    detailed : list of dicts from run_layer_patching_for_example

    Returns
    -------
    List of dicts: layer, component, direction, mean_effect, std_effect, n_examples
    """
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for row in detailed:
        key = (row["layer"], row["component"], row["direction"])
        groups[key].append(row["effect"])

    agg = []
    for (layer, component, direction), effects in sorted(groups.items()):
        mean_e = sum(effects) / len(effects)
        std_e  = statistics.stdev(effects) if len(effects) > 1 else 0.0
        agg.append({
            "layer":       layer,
            "component":   component,
            "direction":   direction,
            "mean_effect": mean_e,
            "std_effect":  std_e,
            "n_examples":  len(effects),
        })

    return agg
