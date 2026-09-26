"""
headlib/head_hooks.py
=====================
Per-head hook utilities for Experiment 3 (per-head activation patching +
ablation on the monoT5 decoder).

Where "a head" lives
--------------------
T5Attention computes, per query position:

    inner = concat(head_0, ..., head_{H-1})        # (batch, q_len, H * d_kv)
    out   = W_o @ inner                            # (batch, q_len, d_model)

We intervene on ``inner`` — the input of the output projection ``.o`` —
where head h occupies the contiguous slice [h*d_kv : (h+1)*d_kv].  Because
W_o is a plain linear map with no bias, replacing head h's slice replaces
exactly that head's additive contribution to the attention output (and hence
to the residual stream); zeroing the slice zeroes the head's contribution.

Why this is simpler than the layer-level machinery in src/activation_hooks.py
------------------------------------------------------------------------------
Experiment 1 patches whole sub-layer residual contributions and therefore had
to cache the (output - input) difference.  Here we patch BEFORE W_o, where the
per-head decomposition is exact by construction, so we cache/replace the raw
``.o`` input via forward PRE-hooks on the ``.o`` Linear module.

Decoder-only scope, one decoder step
-------------------------------------
monoT5 scoring runs a single decoder step (decoder_input_ids = [start token]),
so the ``.o`` input of every decoder attention module has query length 1:
shape (batch, 1, inner_dim).  Control, attack, and clean runs therefore all
produce IDENTICALLY-SHAPED decoder head activations — no sequence-length
alignment is needed at patch time (alignment only matters when building the
padded-control encoder input, which we reuse from src/model_utils.py).

Batched-over-heads patching
----------------------------
To score all H heads of one (layer, component) in a single decoder pass, we
run a batch of H identical rows and patch, in row h, ONLY head h's slice.
``make_per_row_head_pre_hook`` implements this with a block-diagonal mask so
row h = base activations everywhere except head h's slice, which comes from
the replacement (or zero).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Decoder attention components in scope for Experiment 3.
# Decoder MLP is excluded (no heads); encoder is out of scope.
HEAD_COMPONENTS = ["decoder_self_attn", "decoder_cross_attn"]

# Short labels used in plots/CSV-derived figures: L{layer}{S|X}H{head}
COMPONENT_SHORT = {"decoder_self_attn": "S", "decoder_cross_attn": "X"}


def get_attention_module(model: nn.Module, component: str, layer_idx: int) -> nn.Module:
    """Return the T5Attention module for a decoder (component, layer) pair."""
    block = model.decoder.block[layer_idx]
    if component == "decoder_self_attn":
        return block.layer[0].SelfAttention
    elif component == "decoder_cross_attn":
        return block.layer[1].EncDecAttention
    raise ValueError(
        f"Unknown head component: '{component}'. Must be one of {HEAD_COMPONENTS}."
    )


def get_o_proj(model: nn.Module, component: str, layer_idx: int) -> nn.Linear:
    """Return the output-projection Linear (W_o) whose INPUT we intervene on."""
    return get_attention_module(model, component, layer_idx).o


def head_geometry(model: nn.Module) -> Tuple[int, int, int]:
    """Return (n_heads, d_kv, inner_dim) from the model config."""
    n_heads = model.config.num_heads
    d_kv = model.config.d_kv
    return n_heads, d_kv, n_heads * d_kv


def enumerate_slots(
    model: nn.Module,
    components: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
) -> List[Tuple[str, int]]:
    """
    Enumerate the (component, layer) slots in scope.

    Each slot holds n_heads heads, so for monoT5-base the default scope is
    2 components x 12 layers = 24 slots = 288 head-slots.
    """
    components = components or HEAD_COMPONENTS
    if layers is None:
        layers = list(range(len(model.decoder.block)))
    return [(comp, layer) for layer in layers for comp in components]


def slot_key(component: str, layer_idx: int) -> str:
    """Cache key for one (component, layer) slot, e.g. 'decoder_cross_attn_layer3'."""
    return f"{component}_layer{layer_idx}"


# ---------------------------------------------------------------------------
# Pre-hooks on the .o projection
# ---------------------------------------------------------------------------

def make_cache_pre_hook(cache: Dict[str, torch.Tensor], key: str) -> Callable:
    """
    Forward PRE-hook that records the ``.o`` input (concatenated head outputs).

    The tensor is detached and cloned; for a single decoder step it is tiny
    ((1, 1, inner_dim)), so it stays on the model device for cheap reuse.
    """
    def hook(module: nn.Module, args: tuple) -> None:
        cache[key] = args[0].detach().clone()
    return hook


def block_diag_head_mask(n_heads: int, d_kv: int, device: torch.device) -> torch.Tensor:
    """
    Build the (n_heads, 1, n_heads*d_kv) mask where row h is 1 exactly on
    head h's slice.  Used to patch head h only in batch row h.
    """
    mask = torch.zeros(n_heads, 1, n_heads * d_kv, device=device)
    for h in range(n_heads):
        mask[h, :, h * d_kv:(h + 1) * d_kv] = 1.0
    return mask


def make_per_row_head_pre_hook(
    replacement: Optional[torch.Tensor],
    head_mask: torch.Tensor,
) -> Callable:
    """
    Forward PRE-hook for a batch of n_heads rows that, in row h, replaces
    ONLY head h's slice of the ``.o`` input.

    Parameters
    ----------
    replacement : Tensor (1, 1, inner_dim) or None
        Cached ``.o`` input from the source run (patching / mean ablation),
        or None for zero ablation (head output set to the zero vector).
    head_mask : Tensor (n_heads, 1, inner_dim)
        Block-diagonal mask from block_diag_head_mask, on the model device.
    """
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]  # (n_heads, 1, inner_dim) — base run, one row per head
        if replacement is None:
            new_hidden = hidden * (1.0 - head_mask)
        else:
            new_hidden = hidden * (1.0 - head_mask) + replacement.to(hidden.dtype) * head_mask
        return (new_hidden,) + args[1:]
    return hook


def make_full_replace_pre_hook(replacement: torch.Tensor) -> Callable:
    """
    Forward PRE-hook that replaces the ENTIRE ``.o`` input (all heads at once).

    Only used by the tests: replacing all head slices must reproduce the
    layer-level contribution patch of Experiment 1 (src/activation_hooks.py).
    """
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]
        return (replacement.to(hidden.dtype).expand_as(hidden),) + args[1:]
    return hook
