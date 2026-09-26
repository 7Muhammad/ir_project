"""
src/activation_hooks.py
=======================
PyTorch forward-hook utilities for caching and replacing activations.

Background: PyTorch forward hooks
-----------------------------------
A forward hook is a function PyTorch calls AFTER a module's forward() method
finishes.  The hook receives (module, input, output) and can optionally
return a new output to replace the original.

We use hooks in two modes:

  1. CACHE mode: record the module's hidden-state output (Stage 02).
  2. PATCH mode: replace the module's hidden-state output with a pre-stored
     tensor from a different forward pass (Stage 03).

T5 module output types
------------------------
Some T5 sub-modules return a TUPLE: (hidden_state, ..., optional_past_kv).
Others (the MLP feed-forward layers) return a plain TENSOR.
The helpers below handle both cases uniformly.

Layer naming convention
------------------------
Encoder layer i:
  model.encoder.block[i].layer[0]  → T5LayerSelfAttention   ("encoder_self_attn")
  model.encoder.block[i].layer[1]  → T5LayerFF              ("encoder_mlp")

Decoder layer i:
  model.decoder.block[i].layer[0]  → T5LayerSelfAttention   ("decoder_self_attn")
  model.decoder.block[i].layer[1]  → T5LayerCrossAttention  ("decoder_cross_attn")
  model.decoder.block[i].layer[2]  → T5LayerFF              ("decoder_mlp")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers: extract / replace hidden states from module outputs
# ---------------------------------------------------------------------------

def get_hidden_state_from_module_output(
    output: Union[torch.Tensor, Tuple],
) -> torch.Tensor:
    """
    Extract the primary hidden-state tensor from a module's forward output.

    WHY THIS IS NEEDED
    -------------------
    T5 sub-modules have heterogeneous return types:
      - T5LayerFF (MLP) returns a plain Tensor.
      - T5LayerSelfAttention and T5LayerCrossAttention return a Tuple whose
        first element is the hidden state.

    Parameters
    ----------
    output : Tensor or Tuple

    Returns
    -------
    Tensor : hidden state, shape (batch, seq_len, d_model).
    """
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, tuple):
        return output[0]
    else:
        raise TypeError(
            f"Unexpected module output type: {type(output)}. Expected Tensor or Tuple."
        )


def replace_hidden_state_in_module_output(
    output: Union[torch.Tensor, Tuple],
    new_hidden: torch.Tensor,
) -> Union[torch.Tensor, Tuple]:
    """
    Return a new module output with the primary hidden state replaced.

    Preserves all other tuple elements (e.g. key-value cache tensors).

    Parameters
    ----------
    output : original module output (Tensor or Tuple)
    new_hidden : replacement hidden-state tensor

    Returns
    -------
    Same type as `output`, with position-0 replaced by new_hidden.
    """
    if isinstance(output, torch.Tensor):
        return new_hidden
    elif isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    else:
        raise TypeError(
            f"Unexpected module output type: {type(output)}. Expected Tensor or Tuple."
        )


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def get_hidden_state_from_module_input(input: Any) -> torch.Tensor:
    """
    Extract the primary hidden-state tensor fed INTO a T5 sub-module.

    Every T5 sub-module (T5LayerSelfAttention, T5LayerCrossAttention,
    T5LayerFF) receives the residual-stream hidden state as its first
    positional argument, so the hook's `input[0]` is always that tensor.

    Parameters
    ----------
    input : the tuple of positional arguments passed to module.forward()

    Returns
    -------
    Tensor : input hidden state, shape (batch, seq_len, d_model).
    """
    if not isinstance(input, tuple) or len(input) == 0:
        raise TypeError(
            f"Unexpected module input type: {type(input)}. Expected a non-empty tuple."
        )
    return input[0]


def make_cache_hook(cache: Dict[str, torch.Tensor], key: str) -> Callable:
    """
    Create a forward hook that stores the module's RESIDUAL-STREAM CONTRIBUTION
    in `cache`, i.e. the tensor the sub-layer ADDS to the residual stream:

        contribution = output_hidden_state - input_hidden_state

    WHY THE CONTRIBUTION AND NOT THE FULL OUTPUT
    --------------------------------------------
    Every T5 sub-layer is a residual block: ``output = input + sublayer(input)``.
    If we cached/patched the *full output* of each sub-layer, patching an
    earlier sub-layer's output and letting the next sub-layer recompute would
    be mathematically identical to patching the later sub-layer's output
    directly (the model is deterministic in eval mode).  That makes the
    per-component effects collapse onto one another — e.g. self-attention and
    the following MLP get the exact same score, which is not informative.

    Caching the *contribution* (the sub-layer's own additive write to the
    residual stream) lets us swap one component's write while leaving the
    incoming residual stream untouched, giving a clean per-component
    (direct-effect) attribution.

    The tensor is detached (no grad) and cloned (our own copy, not a view
    that future passes would overwrite).

    Parameters
    ----------
    cache : dict where the tensor is stored under `key`
    key : string key, e.g. "encoder_self_attn_layer3"
    """
    def hook(module: nn.Module, input: Any, output: Any) -> None:
        out_hidden = get_hidden_state_from_module_output(output)
        in_hidden  = get_hidden_state_from_module_input(input)
        cache[key] = (out_hidden - in_hidden).detach().clone()
    return hook


def make_patch_hook(replacement: torch.Tensor) -> Callable:
    """
    Create a forward hook that replaces the module's RESIDUAL-STREAM
    CONTRIBUTION with `replacement`, i.e. it sets

        new_output = input_hidden_state + replacement

    where `replacement` is the cached contribution from a different forward
    pass (see make_cache_hook).

    This is the core activation-patching mechanism (direct-effect variant):
      - We run a forward pass with one input (e.g. padded control).
      - At a specific layer/component, we keep the incoming residual stream
        from this run but swap in the additive contribution that the SAME
        component produced in a different run (e.g. the attacked input).
      - Downstream layers see the patched residual stream and produce a new
        final score.

    Patching the contribution (rather than the full output) keeps the
    incoming residual stream from the base run, so each component's effect is
    isolated and self-attention / MLP / cross-attention no longer collapse to
    identical scores.

    WHY SHAPES MUST MATCH
    ----------------------
    The replacement contribution is added to the incoming hidden state.
    Both have shape (batch, seq_len, d_model).  If the padded control and
    attacked input have different seq_len, this crashes.  The padded-control
    construction in model_utils.py guarantees equal lengths, so this should
    never happen when used correctly.

    Parameters
    ----------
    replacement : Tensor, shape (1, seq_len, d_model), on model device.
        The cached residual-stream contribution to inject.
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        in_hidden = get_hidden_state_from_module_input(input)
        new_hidden = in_hidden + replacement
        return replace_hidden_state_in_module_output(output, new_hidden)
    return hook


# ---------------------------------------------------------------------------
# Layer/component enumeration
# ---------------------------------------------------------------------------

COMPONENTS = [
    "encoder_self_attn",
    "encoder_mlp",
    "decoder_self_attn",
    "decoder_cross_attn",
    "decoder_mlp",
]


def get_module_for_component(
    model: nn.Module,
    component: str,
    layer_idx: int,
) -> nn.Module:
    """Return the T5 sub-module for a (component, layer) pair."""
    if component == "encoder_self_attn":
        return model.encoder.block[layer_idx].layer[0]
    elif component == "encoder_mlp":
        return model.encoder.block[layer_idx].layer[1]
    elif component == "decoder_self_attn":
        return model.decoder.block[layer_idx].layer[0]
    elif component == "decoder_cross_attn":
        return model.decoder.block[layer_idx].layer[1]
    elif component == "decoder_mlp":
        return model.decoder.block[layer_idx].layer[2]
    else:
        raise ValueError(
            f"Unknown component: '{component}'. Must be one of {COMPONENTS}."
        )


def n_encoder_layers(model: nn.Module) -> int:
    """Return the number of encoder layers."""
    return len(model.encoder.block)


def n_decoder_layers(model: nn.Module) -> int:
    """Return the number of decoder layers."""
    return len(model.decoder.block)


# ---------------------------------------------------------------------------
# High-level: cache all layer activations for one forward pass
# (accepts a pre-built encoding dict instead of raw text)
# ---------------------------------------------------------------------------

def cache_all_activations_from_enc(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Run one forward pass with a pre-built encoder encoding and cache all
    layer-component activations.

    WHY WE ACCEPT AN ENCODING DICT
    --------------------------------
    The padded-control input carries a custom attention_mask (0 for pad slots).
    This mask must be passed explicitly to the model — we cannot reconstruct
    it from text alone.  By accepting the pre-built encoding dict (produced by
    build_padded_control_and_attack_encodings), we ensure the correct mask is
    used.

    The same function works for the attacked input encoding (all masks = 1).

    MEMORY NOTE
    -----------
    For monoT5-base (12 enc + 12 dec layers) with a 512-token sequence,
    the cache contains 60 tensors × (1, seq_len, 768) float32 ≈ 90 MB.
    Tensors are moved to CPU after caching to free GPU memory.
    The config option max_cached_examples limits total disk usage.

    Parameters
    ----------
    model : T5ForConditionalGeneration (eval mode)
    enc : dict with "input_ids" and "attention_mask", each (1, seq_len)
    device : torch.device

    Returns
    -------
    Dict[str, Tensor]  — keys like "encoder_self_attn_layer0", values on CPU.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles = []

    n_enc = n_encoder_layers(model)
    n_dec = n_decoder_layers(model)

    # Register cache hooks for every encoder component at every layer
    for layer_idx in range(n_enc):
        for comp in ["encoder_self_attn", "encoder_mlp"]:
            key = f"{comp}_layer{layer_idx}"
            module = get_module_for_component(model, comp, layer_idx)
            handles.append(module.register_forward_hook(make_cache_hook(cache, key)))

    # Register cache hooks for every decoder component at every layer
    for layer_idx in range(n_dec):
        for comp in ["decoder_self_attn", "decoder_cross_attn", "decoder_mlp"]:
            key = f"{comp}_layer{layer_idx}"
            module = get_module_for_component(model, comp, layer_idx)
            handles.append(module.register_forward_hook(make_cache_hook(cache, key)))

    # Run the forward pass with the pre-built encoding
    enc_on_device = {k: v.to(device) for k, v in enc.items()}
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )
    with torch.no_grad():
        model(**enc_on_device, decoder_input_ids=decoder_input_ids)

    # Remove all hooks — must happen even if an exception was raised
    for h in handles:
        h.remove()

    # Move all cached tensors to CPU to free GPU memory between examples
    cache = {k: v.cpu() for k, v in cache.items()}
    return cache
