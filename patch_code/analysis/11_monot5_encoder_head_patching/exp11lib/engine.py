"""
exp11lib/engine.py
====================
Per-head patching/ablation engine for Experiment 11 (encoder self-attention).

Reuses from Experiment 1 (imported, not duplicated):
  - model loading / scoring formula      src/model_utils.py
  - padded-control (Type B) construction src/model_utils.py + src/alignment.py
  - fwd / rev / combined effect formulas src/patching.py (Eq. 4-6), applied
    here per (layer, head) instead of per layer-component
  - SKIP_EPSILON degenerate-example guard src/patching.py

One performance trick, adapted from Experiment 3 for the encoder
--------------------------------------------------------------------
Experiment 3 gets TWO tricks: encoder-output reuse (patches are decoder-only,
so the encoder runs once and is cached) AND batched-over-heads decoder
passes. Encoder-output reuse does NOT apply here — the patch happens INSIDE
the encoder's own forward computation (same limitation exp6lib/engine.py's
whole-block encoder self-attention patching has), so the encoder must
actually re-execute for every (layer, direction) unit.

Batched-over-heads STILL applies, translated to the encoder: to score all
n_heads heads of one layer, we run ONE encoder forward pass with a batch of
n_heads IDENTICAL copies of the same input sequence, where row h has only
head h's `.o`-input slice patched (see exp11lib/head_hooks.py). This is
exactly Experiment 3's trick, just batching over "hypothetical parallel
encoder runs" instead of "hypothetical parallel decoder runs" — a full
network re-execution per (layer, direction) instead of per (layer, head,
direction), i.e. a 12x reduction in forward-pass COUNT (not FLOPs) versus
scoring each head with its own encoder call.

Score bookkeeping per (head, example) — identical definitions to Experiment 3
--------------------------------------------------------------------------------
  fwd patch : base = padded control, replacement = attack head activation
              fwd_effect = (score_patched_fwd - score_control) / delta
  rev patch : base = attack, replacement = control head activation
              rev_effect = (score_attack - score_patched_rev) / delta
  combined  = min(fwd_effect, rev_effect)
  delta     = score_attack - score_control

  zero ablation : head output set to the zero vector.
  mean ablation  : head output replaced by the padded-control (Type B)
                   activation for that head — SECONDARY metric only, per
                   Experiment 3's finding that zero ablation over-attributes
                   importance; mean/control is primary throughout.

  NOTE (same identity Experiment 3 relies on): on the ATTACK base input,
  "mean ablation" and "reverse patching" are the same computation (attack
  run, control activation at that head), so score_ablated_mean ==
  score_patched_rev by construction. Computed once, written to both columns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from src.patching import SKIP_EPSILON

from exp11lib.head_hooks import (
    block_diag_head_mask,
    get_o_proj,
    head_geometry,
    make_cache_pre_hook,
    make_per_row_head_pre_hook,
    slot_key,
)


# ---------------------------------------------------------------------------
# Decoder scoring over an (already batched) encoder hidden state
# ---------------------------------------------------------------------------

def decoder_pass_scores_batched(
    model: nn.Module,
    enc_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    true_id: int,
    false_id: int,
) -> torch.Tensor:
    """
    Single-step decoder scoring over an encoder hidden state that is ALREADY
    batched (batch = n_heads, one row per hypothetical head-patch), unlike
    Experiment 3's decoder_pass_scores which expands a single (1, L, d) row.

    enc_hidden : (B, L, d)   enc_mask : (B, L)   Returns : (B,) scores.
    """
    batch = enc_hidden.shape[0]
    dec = torch.full(
        (batch, 1), model.config.decoder_start_token_id,
        dtype=torch.long, device=enc_hidden.device,
    )
    with torch.no_grad():
        out = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_mask,
            decoder_input_ids=dec,
            use_cache=False,
        )
    logits = out.logits[:, 0, :]
    return logits[:, true_id] - logits[:, false_id]


# ---------------------------------------------------------------------------
# Head-activation caching (one unpatched encoder+decoder pass per input)
# ---------------------------------------------------------------------------

def cache_head_inputs(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
    layers: List[int],
    true_id: int,
    false_id: int,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Run one unpatched (batch=1) encoder forward + decoder step, caching the
    ``.o`` input (shape (1, seq_len, inner_dim)) at every requested layer.

    Returns (cache, base_score) — the base score comes for free from the
    same pass.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles = []
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    try:
        for layer in layers:
            handles.append(
                get_o_proj(model, layer).register_forward_pre_hook(
                    make_cache_pre_hook(cache, slot_key(layer))
                )
            )
        with torch.no_grad():
            enc_out = model.encoder(
                input_ids=enc_dev["input_ids"], attention_mask=enc_dev["attention_mask"],
            )
        score = decoder_pass_scores_batched(
            model, enc_out.last_hidden_state, enc_dev["attention_mask"], true_id, false_id,
        )[0].item()
    finally:
        for h in handles:
            h.remove()
    return cache, score


def head_scores_for_layer(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
    layer_idx: int,
    replacement: Optional[torch.Tensor],
    head_mask: torch.Tensor,
    true_id: int,
    false_id: int,
) -> List[float]:
    """
    Score every head of one encoder layer in a SINGLE batched
    encoder-then-decoder pass. Row h has only head h's slice replaced (or
    zeroed when `replacement` is None).

    Returns a list of n_heads floats (index = head_idx).
    """
    n_heads = head_mask.shape[0]
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    batched_ids = enc_dev["input_ids"].expand(n_heads, -1)
    batched_mask = enc_dev["attention_mask"].expand(n_heads, -1)

    handle = get_o_proj(model, layer_idx).register_forward_pre_hook(
        make_per_row_head_pre_hook(replacement, head_mask)
    )
    try:
        with torch.no_grad():
            enc_out = model.encoder(input_ids=batched_ids, attention_mask=batched_mask)
        scores = decoder_pass_scores_batched(
            model, enc_out.last_hidden_state, batched_mask, true_id, false_id,
        )
    finally:
        handle.remove()
    return scores.tolist()


# ---------------------------------------------------------------------------
# Per-example driver
# ---------------------------------------------------------------------------

def run_grid_example(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    layers: List[int],
    true_id: int,
    false_id: int,
    device: torch.device,
    meta: Dict,
    methods: List[str],
) -> Optional[List[Dict]]:
    """
    Grid row computation for one example: fwd/rev patching + zero/mean
    ablation (attack base) for every head at every requested encoder layer.

    `meta` must carry qid, docid, attack_name.

    Returns one row dict per (layer, head), or None when
    |score_attack - score_control| < SKIP_EPSILON.
    """
    n_heads, d_kv, _ = head_geometry(model)
    head_mask = block_diag_head_mask(n_heads, d_kv, device)

    ctrl_cache, control_score = cache_head_inputs(model, control_enc, device, layers, true_id, false_id)
    atk_cache, attack_score = cache_head_inputs(model, attack_enc, device, layers, true_id, false_id)

    delta = attack_score - control_score
    if abs(delta) < SKIP_EPSILON:
        return None

    rows: List[Dict] = []
    for layer in layers:
        key = slot_key(layer)

        fwd_scores = head_scores_for_layer(
            model, control_enc, device, layer, atk_cache[key], head_mask, true_id, false_id,
        )
        # Reverse patch == mean ablation on the attack base (see module docstring).
        rev_scores = head_scores_for_layer(
            model, attack_enc, device, layer, ctrl_cache[key], head_mask, true_id, false_id,
        )
        if "zero" in methods:
            zero_scores = head_scores_for_layer(
                model, attack_enc, device, layer, None, head_mask, true_id, false_id,
            )
        else:
            zero_scores = [None] * n_heads

        for h in range(n_heads):
            fwd_effect = (fwd_scores[h] - control_score) / delta
            rev_effect = (attack_score - rev_scores[h]) / delta
            mean_score = rev_scores[h] if "mean" in methods else None
            rows.append({
                "qid": meta["qid"], "docid": meta["docid"], "attack_name": meta["attack_name"],
                "layer": layer, "head_idx": h,
                "score_control": control_score, "score_attack": attack_score,
                "score_patched_fwd": fwd_scores[h], "score_patched_rev": rev_scores[h],
                "score_ablated_zero": zero_scores[h], "score_ablated_mean": mean_score,
                "fwd_effect": fwd_effect, "rev_effect": rev_effect,
                "combined_effect": min(fwd_effect, rev_effect),
                "score_drop_zero": (attack_score - zero_scores[h]) if zero_scores[h] is not None else None,
                "score_drop_mean": (attack_score - mean_score) if mean_score is not None else None,
            })
    return rows
