"""
exp14lib/hooks.py
===================
Defense (subtraction) intervention: a' = a - scale * direction, at ONE head,
batched over a list of scales in a single forward pass — the same
batching-for-kernel-efficiency trick Experiment 3 uses for batched-over-
heads decoder passes (see DECISIONS.md item 9 for the scope boundary: we
batch over scales, not over heads/position-masks/examples).

Decoder heads
-------------
Reuses Experiment 2's additive-shift machinery as-is: the pre-`.o` head
slice math (headlib.head_hooks.head_geometry) and the batched-decoder-pass
shift hook (exp2lib.intervene.make_head_shift_pre_hook /
head_scores_for_shift), restricted to sign=-1.0 (defense only — Experiment
2's sufficiency/addition test, sign=+1.0, is explicitly out of scope; task
spec section 7). Encoder-output reuse (headlib.engine.compute_encoder_states)
still applies: the encoder forward for the attacked/clean input runs once
and is reused across every decoder head, condition, and scale batch.

Encoder heads
-------------
NEW — Experiment 2 never touched encoder heads; Experiment 6 patches whole-
layer contributions, not per-head slices. Combines Experiment 2's per-head
shift math with Experiment 6's position-masked patch shape
(exp14lib.position_masks): the shift lands ONLY inside the requested
position mask. Encoder-output reuse does NOT apply (the intervention is
INSIDE the encoder's own forward computation, same limitation Experiment
6/11's encoder-side patching already document) — the encoder must actually
re-execute for every (head, position mask) unit, but the len(scales) scale
values for that unit are batched into one encoder forward pass.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from headlib.engine import compute_encoder_states, decoder_pass_scores
from headlib.head_hooks import get_o_proj, head_geometry

from exp2lib.direction_hooks import get_encoder_o_proj

DEFENSE_SIGN = -1.0  # subtract only — Phase 1 scope (task spec section 7)


# ---------------------------------------------------------------------------
# Decoder heads (reuse Experiment 2's shift hook, sign fixed to -1.0)
# ---------------------------------------------------------------------------

def make_decoder_shift_pre_hook(direction: torch.Tensor, scales: List[float], head_idx: int, d_kv: int) -> Callable:
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]  # (n_scales, 1, inner_dim)
        start = head_idx * d_kv
        end = start + d_kv
        shift = torch.zeros_like(hidden)
        for i, scale in enumerate(scales):
            shift[i, :, start:end] = DEFENSE_SIGN * scale * direction.to(hidden.dtype)
        return (hidden + shift,) + args[1:]
    return hook


def decoder_defense_scores_for_scales(
    model: nn.Module,
    enc_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    direction: torch.Tensor,
    scales: List[float],
    true_id: int,
    false_id: int,
) -> List[float]:
    """Batched-over-scales decoder pass with `head_idx`'s slice shifted by -scale*direction in every row."""
    _, d_kv, _ = head_geometry(model)
    handle = get_o_proj(model, "decoder_cross_attn", layer_idx).register_forward_pre_hook(
        make_decoder_shift_pre_hook(direction, scales, head_idx, d_kv)
    )
    try:
        scores = decoder_pass_scores(model, enc_hidden, enc_mask, true_id, false_id, len(scales))
    finally:
        handle.remove()
    return scores.tolist()


# ---------------------------------------------------------------------------
# Encoder heads (new: per-head slice + position mask, batched over scales)
# ---------------------------------------------------------------------------

def make_encoder_masked_shift_pre_hook(
    direction: torch.Tensor,
    scales: List[float],
    head_idx: int,
    d_kv: int,
    position_mask: torch.Tensor,
) -> Callable:
    """
    Forward PRE-hook on the encoder `.o` projection of ONE layer. Batch row
    i (of len(scales) identical input copies) gets head `head_idx`'s slice
    shifted by -scales[i]*direction, but ONLY at positions where
    `position_mask` (shape (1, seq_len, 1)) is 1 — every other position, and
    every other head's slice, is left as the base run's own value.
    """
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]  # (n_scales, seq_len, inner_dim)
        start = head_idx * d_kv
        end = start + d_kv
        shift = torch.zeros_like(hidden)
        for i, scale in enumerate(scales):
            shift[i, :, start:end] = DEFENSE_SIGN * scale * direction.to(hidden.dtype)
        shift = shift * position_mask.to(hidden.dtype)
        return (hidden + shift,) + args[1:]
    return hook


def encoder_defense_scores_for_scales(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    direction: torch.Tensor,
    scales: List[float],
    position_mask: torch.Tensor,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> List[float]:
    """
    One batched-over-scales encoder forward (batch=len(scales) identical
    copies of the input, one head-slice shifted per row at the masked
    positions only) followed by one batched single-step decoder pass over
    the resulting (per-row DIFFERENT) encoder states.
    """
    _, d_kv, _ = head_geometry(model)
    n_scales = len(scales)
    input_ids_b = input_ids.to(device).expand(n_scales, -1)
    attention_mask_b = attention_mask.to(device).expand(n_scales, -1)
    pm = position_mask.to(device)

    handle = get_encoder_o_proj(model, layer_idx).register_forward_pre_hook(
        make_encoder_masked_shift_pre_hook(direction, scales, head_idx, d_kv, pm)
    )
    try:
        with torch.no_grad():
            enc_out = model.encoder(input_ids=input_ids_b, attention_mask=attention_mask_b)
    finally:
        handle.remove()

    dec = torch.full((n_scales, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_out.last_hidden_state),
            attention_mask=attention_mask_b,
            decoder_input_ids=dec,
            use_cache=False,
        )
    logits = out.logits[:, 0, :]
    scores = logits[:, true_id] - logits[:, false_id]
    return scores.tolist()
