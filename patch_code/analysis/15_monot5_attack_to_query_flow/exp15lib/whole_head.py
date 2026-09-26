"""
exp15lib/whole_head.py
=======================
Whole-head reference patching, recomputed on Exp 15's own sample manifest
(DECISIONS.md item 22) by calling Experiment 11's implementation directly:

  exp11lib.engine.cache_head_inputs   — unpatched pass caching every layer's
                                        `.o` input (+ live score)
  exp11lib.engine.head_scores_for_layer — one batched pass per (layer,
                                        direction) where row h replaces head
                                        h's `.o`-input slice at EVERY position

Exp 11 scores all 12 heads of a layer per pass by building a block-diagonal
mask with one batch row per head. `head_scores_for_layer` takes that mask as
an argument and sizes the batch from it, so we pass only the canonical heads'
rows of Exp 11's own mask (same code path, row h still patches exactly head
h's slice) — skipping the 42 non-canonical heads' rows. The normalisation uses the SAME cached Exp 01 baselines as the
edge stages (not Exp 11's live delta), so edge and whole-head metrics share
one denominator per example.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

import exp15lib  # noqa: F401
from exp11lib.engine import cache_head_inputs, head_scores_for_layer
from exp11lib.head_hooks import block_diag_head_mask, head_geometry, slot_key


def whole_head_scores(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    heads: Sequence[Tuple[int, int]],
    device: torch.device,
    true_id: int,
    false_id: int,
) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], float, float]:
    """Returns ({(L,h): (score_fwd_patched, score_rev_patched)}, live_control, live_attack)."""
    n_heads, d_kv, _ = head_geometry(model)
    layers = sorted({L for L, _ in heads})
    ctrl_cache, live_c = cache_head_inputs(model, control_enc, device, layers, true_id, false_id)
    atk_cache, live_a = cache_head_inputs(model, attack_enc, device, layers, true_id, false_id)
    full_mask = block_diag_head_mask(n_heads, d_kv, device)
    out = {}
    for L in layers:
        hs = [h for (LL, h) in heads if LL == L]
        mask = full_mask[hs]          # (len(hs), 1, inner): row i patches head hs[i]
        fwd = head_scores_for_layer(model, control_enc, device, L, atk_cache[slot_key(L)], mask, true_id, false_id)
        rev = head_scores_for_layer(model, attack_enc, device, L, ctrl_cache[slot_key(L)], mask, true_id, false_id)
        for i, h in enumerate(hs):
            out[(L, h)] = (fwd[i], rev[i])
    return out, live_c, live_a
