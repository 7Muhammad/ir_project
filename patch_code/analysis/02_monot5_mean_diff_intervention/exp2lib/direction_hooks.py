"""
exp2lib/direction_hooks.py
===========================
Encoder self-attention per-head hooks (new — Experiment 3 only covers the
decoder) plus a single "cache everything Part 1 needs" driver that combines
Experiment 1's whole-vector residual-contribution hooks
(src.activation_hooks.make_cache_hook) with Experiment 3's per-head
``.o``-input pre-hooks (headlib.head_hooks.make_cache_pre_hook) across three
components: encoder_self_attn, decoder_self_attn, decoder_cross_attn.

Why encoder self-attention needs its own module lookup
--------------------------------------------------------
Experiment 3's ``headlib/head_hooks.py`` only knows about decoder attention
modules (``model.decoder.block[i].layer[0|1]``). Encoder self-attention lives
at ``model.encoder.block[i].layer[0]`` and has the identical per-head
decomposition (``inner = concat(head_0, ..., head_{H-1})``, ``out = W_o @
inner``), so the same cache-pre-hook mechanics apply — only the module
lookup differs. Everything else (``make_cache_pre_hook``,
``block_diag_head_mask``, ``head_geometry``) is reused as-is from
``headlib.head_hooks``.

Encoder activations are NOT single-vector-per-example the way decoder head
activations are (monoT5 scoring runs one decoder step, so decoder ``.o``
inputs are always (1, 1, inner_dim); encoder ``.o`` inputs are
(1, seq_len, inner_dim), one row per token). Pooling those down to one
direction vector per example is handled downstream in
``exp2lib/direction_fit.py``, not here — this module only caches raw
activations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.activation_hooks import get_module_for_component, make_cache_hook
from headlib.head_hooks import get_o_proj, make_cache_pre_hook

# The three components Part 1 computes directions for. encoder_self_attn is
# genuinely new (see module docstring); decoder_self_attn is included beyond
# the two locations named in the task spec because Experiment 3's flagged-head
# list (grid_a, combined_effect_mean > 0.02) contains one decoder_self_attn
# head (layer 11, head 3) — see DECISIONS.md.
DIRECTION_COMPONENTS = ["encoder_self_attn", "decoder_self_attn", "decoder_cross_attn"]


def get_encoder_o_proj(model: nn.Module, layer_idx: int) -> nn.Linear:
    """Return the encoder self-attention output-projection Linear (W_o)."""
    return model.encoder.block[layer_idx].layer[0].SelfAttention.o


def get_o_proj_for_component(model: nn.Module, component: str, layer_idx: int) -> nn.Linear:
    """Dispatch to the encoder or decoder ``.o`` projection lookup."""
    if component == "encoder_self_attn":
        return get_encoder_o_proj(model, layer_idx)
    return get_o_proj(model, component, layer_idx)


def whole_vector_key(component: str, layer_idx: int) -> Tuple[str, int]:
    """Cache key for one (component, layer) whole-vector slot: (component, layer)."""
    return (component, layer_idx)


def per_head_key(component: str, layer_idx: int) -> Tuple[str, int]:
    """Cache key for one (component, layer) per-head slot: (component, layer)."""
    return (component, layer_idx)


def cache_activations_for_direction(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    true_id: int,
    false_id: int,
    device: torch.device,
    n_encoder_layers: int,
    n_decoder_layers: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """
    Run one forward pass and cache both whole-vector and per-head activations
    for all three DIRECTION_COMPONENTS, at every layer, in a single pass.

    Mirrors the "register many hooks, one forward pass" style already used
    by ``src.activation_hooks.cache_all_activations_from_enc`` (whole-vector)
    and ``headlib.engine.cache_head_inputs`` (per-head) — this function does
    both at once so Part 1 needs exactly one control pass and one attack pass
    per example, not four.

    Returns
    -------
    whole_vector_cache : Dict[(component, layer_idx), Tensor]
        residual-stream contribution (output - input), shape
        (1, seq_len_or_1, d_model). seq_len for encoder components,
        always 1 for decoder components (single decode step).
    per_head_cache : Dict[(component, layer_idx), Tensor]
        raw ``.o``-projection input, shape (1, seq_len_or_1, inner_dim).
    score : float
        logit("true") - logit("false") from the same pass.
    """
    whole_vector_cache: Dict[Tuple[str, int], torch.Tensor] = {}
    per_head_cache: Dict[Tuple[str, int], torch.Tensor] = {}
    handles = []

    layer_counts = {
        "encoder_self_attn": n_encoder_layers,
        "decoder_self_attn": n_decoder_layers,
        "decoder_cross_attn": n_decoder_layers,
    }

    for component, n_layers in layer_counts.items():
        for layer_idx in range(n_layers):
            module = get_module_for_component(model, component, layer_idx)
            handles.append(
                module.register_forward_hook(
                    make_cache_hook(whole_vector_cache, whole_vector_key(component, layer_idx))
                )
            )
            o_proj = get_o_proj_for_component(model, component, layer_idx)
            handles.append(
                o_proj.register_forward_pre_hook(
                    make_cache_pre_hook(per_head_cache, per_head_key(component, layer_idx))
                )
            )

    enc_on_device = {k: v.to(device) for k, v in enc.items()}
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )
    try:
        with torch.no_grad():
            outputs = model(**enc_on_device, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, 0]
        score = (logits[true_id] - logits[false_id]).item()
    finally:
        for h in handles:
            h.remove()

    whole_vector_cache = {k: v.cpu() for k, v in whole_vector_cache.items()}
    per_head_cache = {k: v.cpu() for k, v in per_head_cache.items()}
    return whole_vector_cache, per_head_cache, score
