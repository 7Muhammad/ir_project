"""
exp6lib/engine.py
==================
Positional (query-span / document-span) activation patching for Experiment 6.

Reused from Experiment 1 (imported, not duplicated):
  - src.model_utils.score_from_encoding, build_padded_control_and_attack_encodings_general
  - src.activation_hooks: cache_all_activations_from_enc (whole-network contribution
    cache in one forward pass), get_module_for_component,
    get_hidden_state_from_module_input/output, replace_hidden_state_in_module_output
  - src.patching.SKIP_EPSILON (same degenerate-example skip guard)

What is NEW here (not in Experiment 1 or Experiment 3)
--------------------------------------------------------
Both existing experiments patch a component's FULL contribution (Exp 1) or a
per-HEAD slice (Exp 3). Experiment 6 patches a component's contribution at a
subset of SEQUENCE POSITIONS (query-span, document-span, or their union),
leaving all other positions as the base run's own value. This needs two new
hook types:

  1. Encoder self-attention (whole-block, position-masked):
     a forward hook that blends, at masked positions, the OTHER run's cached
     contribution with, at unmasked positions, the CURRENT run's own
     contribution (computed inside the hook from its own input/output,
     exactly as Exp1's make_cache_hook does — see make_positional_patch_hook).

  2. Decoder cross-attention (encoder-row swap, position-masked):
     monoT5's decoder reads ONE fixed encoder_hidden_states tensor at every
     decoder layer's cross-attention (T5Block passes
     key_value_states=encoder_hidden_states as a KEYWORD argument — verified
     empirically, see make_cross_attn_positional_prehook's docstring). To
     patch "one decoder layer's cross-attention" we register a forward
     PRE-hook (with_kwargs=True, needed because key_value_states is a kwarg,
     not positional) on THAT layer's T5LayerCrossAttention module only; every
     other decoder layer keeps reading the base run's own encoder output
     unchanged.

Encoder-output reuse (exact, same trick as Experiment 3)
----------------------------------------------------------
monoT5 scoring is a single decoder step, so decoder cross-attention patching
never needs to re-run the encoder: the control and attack encoder hidden
states are each computed ONCE per example and reused for every
(decoder layer x condition x direction) patched decoder pass via
`encoder_outputs=BaseModelOutput(...)`. Encoder self-attention patching
CANNOT use this shortcut — the patch happens INSIDE the encoder's own
forward computation, so the encoder must actually re-execute for every
patch (same cost structure as Experiment 1's existing encoder patching;
Experiment 6 only adds position-masking on top of it). This asymmetry is
exactly why the task calls for a timing pilot before committing to a grid
size.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from src.activation_hooks import (
    cache_all_activations_from_enc,
    get_hidden_state_from_module_input,
    get_hidden_state_from_module_output,
    get_module_for_component,
    replace_hidden_state_in_module_output,
)
from src.model_utils import score_from_encoding
from src.patching import SKIP_EPSILON

from exp6lib.attention import compute_encoder_attentions, normalized_attention_masses

CONDITIONS = ["doc_only", "query_only", "both"]


# ---------------------------------------------------------------------------
# Position masks
# ---------------------------------------------------------------------------

def build_position_mask(
    seq_len: int,
    condition: str,
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a (1, seq_len, 1) float mask, 1.0 at positions to patch, 0.0 elsewhere.

    condition: "doc_only" | "query_only" | "both"
    query_span, doc_span : (start, end) exclusive, in the shared control/attack
        (Type B/C) indexing.
    """
    mask = torch.zeros(1, seq_len, 1, device=device)
    if condition in ("doc_only", "both"):
        mask[0, doc_span[0]:doc_span[1], 0] = 1.0
    if condition in ("query_only", "both"):
        mask[0, query_span[0]:query_span[1], 0] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Encoder self-attention: whole-block, position-masked patch hook
# ---------------------------------------------------------------------------

def make_positional_patch_hook(replacement: torch.Tensor, position_mask: torch.Tensor) -> Callable:
    """
    Forward hook: at masked positions, replace this module's residual-stream
    contribution with `replacement`'s contribution at those same positions;
    at unmasked positions, keep the CURRENT run's own contribution (computed
    here from input/output, exactly as src.activation_hooks.make_cache_hook
    does for caching).

    Parameters
    ----------
    replacement : Tensor (1, seq_len, d_model) — the OTHER run's cached
        contribution at this (layer, component), full sequence.
    position_mask : Tensor (1, seq_len, 1) — 1.0 at positions to patch.
    """
    def hook(module: nn.Module, input, output):
        in_hidden = get_hidden_state_from_module_input(input)
        out_hidden = get_hidden_state_from_module_output(output)
        own_contribution = out_hidden - in_hidden
        new_contribution = position_mask * replacement.to(out_hidden.dtype) + (1.0 - position_mask) * own_contribution
        new_hidden = in_hidden + new_contribution
        return replace_hidden_state_in_module_output(output, new_hidden)
    return hook


# ---------------------------------------------------------------------------
# Decoder cross-attention: encoder-row swap, position-masked pre-hook
# ---------------------------------------------------------------------------

def make_cross_attn_positional_prehook(
    base_own_kv: torch.Tensor,
    replacement_kv: torch.Tensor,
    position_mask: torch.Tensor,
) -> Callable:
    """
    Forward PRE-hook (with_kwargs=True) for ONE decoder layer's
    T5LayerCrossAttention module.

    WHY with_kwargs=True IS REQUIRED
    ----------------------------------
    T5Block calls `self.layer[1](hidden_states, key_value_states=encoder_hidden_states,
    ...)` — key_value_states is passed as a KEYWORD argument (verified against
    the installed transformers' T5Block.forward source), so a hook that only
    sees positional args (register_forward_pre_hook's default) cannot access
    it. `with_kwargs=True` (torch>=2.0) gives the hook `(module, args, kwargs)`
    and lets it return a replacement `(args, kwargs)` pair.

    Parameters
    ----------
    base_own_kv : Tensor (1, seq_len, d_model) — the BASE run's own encoder
        hidden states (what key_value_states would be if unpatched).
    replacement_kv : Tensor (1, seq_len, d_model) — the OTHER run's encoder
        hidden states, to blend in at masked positions.
    position_mask : Tensor (1, seq_len, 1) — 1.0 at positions to patch.
    """
    def hook(module: nn.Module, args, kwargs):
        blended = position_mask * replacement_kv.to(base_own_kv.dtype) + (1.0 - position_mask) * base_own_kv
        new_kwargs = dict(kwargs)
        new_kwargs["key_value_states"] = blended
        return args, new_kwargs
    return hook


# ---------------------------------------------------------------------------
# Encoder hidden states (final layer) + single decoder-step scoring
# ---------------------------------------------------------------------------

def compute_encoder_hidden_states(
    model: nn.Module, enc: Dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Run the encoder once and return its final hidden states, (1, L, d_model)."""
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.encoder(input_ids=enc_dev["input_ids"], attention_mask=enc_dev["attention_mask"])
    return out.last_hidden_state


def decoder_step_score(
    model: nn.Module,
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    true_id: int,
    false_id: int,
) -> float:
    """Single monoT5 decoder step reusing precomputed encoder hidden states."""
    dec = torch.tensor([[model.config.decoder_start_token_id]], device=encoder_hidden.device)
    with torch.no_grad():
        out = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=attention_mask,
            decoder_input_ids=dec,
            use_cache=False,
        )
    logits = out.logits[0, 0]
    return (logits[true_id] - logits[false_id]).item()


# ---------------------------------------------------------------------------
# Part 2 driver: one example, one region (encoder_self_attn / decoder_cross_attn)
# ---------------------------------------------------------------------------

def run_encoder_self_attn_example(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    layers: List[int],
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
    true_id: int,
    false_id: int,
    device: torch.device,
) -> Optional[List[Dict]]:
    """
    Positional patching of encoder self-attention, whole-block granularity,
    for one example, across all requested layers and all three conditions.

    The encoder must actually re-execute for every patch (the intervention
    happens mid-encoder), so this calls score_from_encoding once per
    (layer, condition, direction) — the expensive path the timing pilot
    measures.
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

        for condition in CONDITIONS:
            mask = build_position_mask(seq_len, condition, query_span, doc_span, device)

            # Forward: control base, inject attack contribution at masked positions.
            handle = module.register_forward_hook(make_positional_patch_hook(atk_contrib, mask))
            try:
                fwd_patched = score_from_encoding(model, control_enc, true_id, false_id, device)
            finally:
                handle.remove()
            fwd_effect = (fwd_patched - control_score) / delta

            # Reverse: attack base, inject control contribution at masked positions.
            handle = module.register_forward_hook(make_positional_patch_hook(ctrl_contrib, mask))
            try:
                rev_patched = score_from_encoding(model, attack_enc, true_id, false_id, device)
            finally:
                handle.remove()
            rev_effect = (attack_score - rev_patched) / delta

            rows.append({
                "region": "encoder_self_attn",
                "layer": layer_idx,
                "condition": condition,
                "score_control": control_score,
                "score_attack": attack_score,
                "score_patched_fwd": fwd_patched,
                "score_patched_rev": rev_patched,
                "fwd_effect": fwd_effect,
                "rev_effect": rev_effect,
                "combined_effect": min(fwd_effect, rev_effect),
            })
    return rows


def run_decoder_cross_attn_example(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    layers: List[int],
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
    true_id: int,
    false_id: int,
    device: torch.device,
) -> Optional[List[Dict]]:
    """
    Positional patching of decoder cross-attention (encoder-row swap),
    for one example, across all requested decoder layers and all three
    conditions. Reuses one encoder forward pass per input for every patch
    (exact speedup — decoder cross-attention never needs the encoder to
    re-execute).
    """
    ctrl_hidden = compute_encoder_hidden_states(model, control_enc, device)
    atk_hidden = compute_encoder_hidden_states(model, attack_enc, device)
    ctrl_mask = control_enc["attention_mask"].to(device)
    atk_mask = attack_enc["attention_mask"].to(device)

    control_score = decoder_step_score(model, ctrl_hidden, ctrl_mask, true_id, false_id)
    attack_score = decoder_step_score(model, atk_hidden, atk_mask, true_id, false_id)
    delta = attack_score - control_score
    if abs(delta) < SKIP_EPSILON:
        return None

    seq_len = ctrl_hidden.shape[1]
    rows: List[Dict] = []

    for layer_idx in layers:
        cross_attn_module = model.decoder.block[layer_idx].layer[1]

        for condition in CONDITIONS:
            mask = build_position_mask(seq_len, condition, query_span, doc_span, device)

            # Forward: control base, inject attack's encoder rows at masked positions.
            handle = cross_attn_module.register_forward_pre_hook(
                make_cross_attn_positional_prehook(ctrl_hidden, atk_hidden, mask), with_kwargs=True
            )
            try:
                fwd_patched = decoder_step_score(model, ctrl_hidden, ctrl_mask, true_id, false_id)
            finally:
                handle.remove()
            fwd_effect = (fwd_patched - control_score) / delta

            # Reverse: attack base, inject control's encoder rows at masked positions.
            handle = cross_attn_module.register_forward_pre_hook(
                make_cross_attn_positional_prehook(atk_hidden, ctrl_hidden, mask), with_kwargs=True
            )
            try:
                rev_patched = decoder_step_score(model, atk_hidden, atk_mask, true_id, false_id)
            finally:
                handle.remove()
            rev_effect = (attack_score - rev_patched) / delta

            rows.append({
                "region": "decoder_cross_attn",
                "layer": layer_idx,
                "condition": condition,
                "score_control": control_score,
                "score_attack": attack_score,
                "score_patched_fwd": fwd_patched,
                "score_patched_rev": rev_patched,
                "fwd_effect": fwd_effect,
                "rev_effect": rev_effect,
                "combined_effect": min(fwd_effect, rev_effect),
            })
    return rows


# ---------------------------------------------------------------------------
# Unified per-example row schema: Part 1 (attention) + Part 2 (patching)
# ---------------------------------------------------------------------------

def run_example_all_regions(
    model: nn.Module,
    inputs,  # exp6lib.run_utils.ExampleInputs, status == "ok"
    encoder_layers: List[int],
    decoder_layers: List[int],
    true_id: int,
    false_id: int,
    device: torch.device,
) -> List[Dict]:
    """
    Produce the unified per-(layer, region, condition) row schema for one
    example: Part 2's patching metrics (region-specific) plus Part 1's
    attention masses merged into the encoder_self_attn rows only (attention
    is descriptive and encoder-only; decoder_cross_attn rows carry None for
    the attention_* fields).

    A skip (negligible delta) in one region does not block the other —
    each region's rows are included independently when available.
    """
    clean_attn = compute_encoder_attentions(model, inputs.clean_enc, device)
    atk_attn = compute_encoder_attentions(model, inputs.attack_enc, device)
    clean_masses = {r["layer"]: r for r in normalized_attention_masses(clean_attn, inputs.query_span, inputs.doc_span_clean)}
    atk_masses = {r["layer"]: r for r in normalized_attention_masses(atk_attn, inputs.query_span, inputs.doc_span_control_attack)}

    all_rows: List[Dict] = []

    enc_rows = run_encoder_self_attn_example(
        model, inputs.control_enc, inputs.attack_enc, encoder_layers,
        inputs.query_span, inputs.doc_span_control_attack, true_id, false_id, device,
    )
    if enc_rows is not None:
        for row in enc_rows:
            layer = row["layer"]
            row["attention_q_to_d_clean"] = clean_masses[layer]["attention_q_to_d"]
            row["attention_d_to_q_clean"] = clean_masses[layer]["attention_d_to_q"]
            row["attention_q_to_d_attack"] = atk_masses[layer]["attention_q_to_d"]
            row["attention_d_to_q_attack"] = atk_masses[layer]["attention_d_to_q"]
        all_rows.extend(enc_rows)

    dec_rows = run_decoder_cross_attn_example(
        model, inputs.control_enc, inputs.attack_enc, decoder_layers,
        inputs.query_span, inputs.doc_span_control_attack, true_id, false_id, device,
    )
    if dec_rows is not None:
        for row in dec_rows:
            row["attention_q_to_d_clean"] = None
            row["attention_d_to_q_clean"] = None
            row["attention_q_to_d_attack"] = None
            row["attention_d_to_q_attack"] = None
        all_rows.extend(dec_rows)

    return all_rows
