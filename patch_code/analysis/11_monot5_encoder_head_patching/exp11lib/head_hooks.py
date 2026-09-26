"""
exp11lib/head_hooks.py
=======================
Per-head hook utilities for Experiment 11 (per-head activation patching +
ablation on the monoT5 ENCODER self-attention) — the direct encoder-side
analog of Experiment 3's decoder head_hooks.py.

Where "a head" lives (identical mechanism to Experiment 3)
------------------------------------------------------------
T5Attention computes, per position:

    inner = concat(head_0, ..., head_{H-1})        # (batch, seq_len, H * d_kv)
    out   = W_o @ inner                             # (batch, seq_len, d_model)

We intervene on ``inner`` — the input of the output projection ``.o`` —
where head h occupies the contiguous slice [h*d_kv : (h+1)*d_kv]. Because
W_o is a plain linear map with no bias, replacing head h's slice replaces
exactly that head's additive contribution to the residual stream.

What's different from Experiment 3's decoder case
-----------------------------------------------------
The decoder's ``.o`` input has shape (batch, 1, inner_dim) because monoT5
scoring is a single decoder step. The ENCODER's ``.o`` input has shape
(batch, seq_len, inner_dim), seq_len ~100-200. Step 1/2 of this experiment
patch WHOLE-SEQUENCE (every position at once, not per-position — that would
be a much larger head x position sweep, explicitly out of scope for this
pass), so the hook logic below is UNCHANGED from Experiment 3's: the
block-diagonal head mask has shape (n_heads, 1, inner_dim) and broadcasts
against a (n_heads, seq_len, inner_dim) hidden tensor exactly the same way
it broadcasts against (n_heads, 1, inner_dim) for the decoder. Same for the
`replacement` tensor: Experiment 3's replacement is (1, 1, inner_dim) and
broadcasts to (n_heads, 1, inner_dim); ours is (1, seq_len, inner_dim) and
broadcasts to (n_heads, seq_len, inner_dim) — same code, no special-casing
needed.

Encoder-only, single component
--------------------------------
Unlike the decoder (self-attn + cross-attn), the encoder has exactly one
attention sublayer per block (T5Block.layer[0].SelfAttention); layer[1] is
the feed-forward (DenseReluDense), which has no heads. So the "component"
axis Experiment 3 needed does not exist here — scope is (layer, head) only,
12 layers x 12 heads = 144 head-slots for monoT5-base.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

COMPONENT = "encoder_self_attn"


def get_attention_module(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the T5Attention module for an encoder layer."""
    return model.encoder.block[layer_idx].layer[0].SelfAttention


def get_o_proj(model: nn.Module, layer_idx: int) -> nn.Linear:
    """Return the output-projection Linear (W_o) whose INPUT we intervene on."""
    return get_attention_module(model, layer_idx).o


def head_geometry(model: nn.Module) -> Tuple[int, int, int]:
    """Return (n_heads, d_kv, inner_dim) from the model config."""
    n_heads = model.config.num_heads
    d_kv = model.config.d_kv
    return n_heads, d_kv, n_heads * d_kv


def enumerate_layers(model: nn.Module, layers: Optional[List[int]] = None) -> List[int]:
    """Encoder layers in scope; None means all (12 for monoT5-base)."""
    if layers is not None:
        return list(layers)
    return list(range(len(model.encoder.block)))


def slot_key(layer_idx: int) -> str:
    """Cache key for one encoder layer's `.o` input, e.g. 'encoder_self_attn_layer3'."""
    return f"{COMPONENT}_layer{layer_idx}"


# ---------------------------------------------------------------------------
# Pre-hooks on the .o projection (identical mechanism to Experiment 3)
# ---------------------------------------------------------------------------

def make_cache_pre_hook(cache: Dict[str, torch.Tensor], key: str) -> Callable:
    """Forward PRE-hook that records the ``.o`` input (concatenated head outputs)."""
    def hook(module: nn.Module, args: tuple) -> None:
        cache[key] = args[0].detach().clone()
    return hook


def block_diag_head_mask(n_heads: int, d_kv: int, device: torch.device) -> torch.Tensor:
    """(n_heads, 1, n_heads*d_kv) mask where row h is 1 exactly on head h's slice."""
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
    ONLY head h's slice of the ``.o`` input, at EVERY sequence position
    (whole-sequence patching — Step 1/2 scope, not per-position Step 4-5
    granularity, which never touches this hook).

    Parameters
    ----------
    replacement : Tensor (1, seq_len, inner_dim) or None
        Cached ``.o`` input from the source run, or None for zero ablation.
    head_mask : Tensor (n_heads, 1, inner_dim)
        Broadcasts against (n_heads, seq_len, inner_dim) automatically.
    """
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]  # (n_heads, seq_len, inner_dim) — one row per head
        if replacement is None:
            new_hidden = hidden * (1.0 - head_mask)
        else:
            new_hidden = hidden * (1.0 - head_mask) + replacement.to(hidden.dtype) * head_mask
        return (new_hidden,) + args[1:]
    return hook


def make_full_replace_pre_hook(replacement: torch.Tensor) -> Callable:
    """
    Forward PRE-hook that replaces the ENTIRE ``.o`` input (all heads at
    once). Used only by the correctness check: replacing all 12 head slices
    at a layer must reproduce Experiment 1/6's whole-layer encoder
    self-attention contribution patch exactly.
    """
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]
        return (replacement.to(hidden.dtype).expand_as(hidden),) + args[1:]
    return hook
