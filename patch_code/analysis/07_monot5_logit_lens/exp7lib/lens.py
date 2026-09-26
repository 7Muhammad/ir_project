"""
exp7lib/lens.py
================
Logit-lens projection for monoT5 decoder cross-attention. See DECISIONS.md
for why "post" means the sublayer's CONTRIBUTION (out - in), why the
unembedding is `lm_head` alone (no final_layer_norm re-application), and
why per-head contributions go through W_o rather than being projected as
raw d_kv-wide slices.

Reused from Experiment 1 (imported, not duplicated):
  - src.activation_hooks.cache_all_activations_from_enc (whole-network
    contribution cache in one forward pass; filtered here to the
    decoder_cross_attn keys this experiment needs).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.activation_hooks import cache_all_activations_from_enc


# ---------------------------------------------------------------------------
# Unembedding projection
# ---------------------------------------------------------------------------

def project_to_logits(model: nn.Module, contribution: torch.Tensor) -> torch.Tensor:
    """
    Apply the model's OWN unembedding to a (1, 1, d_model) contribution.

    Reuses `model.lm_head` directly (already handles tied embeddings) and
    T5's `scale_decoder_outputs` convention — the same scaling
    `T5ForConditionalGeneration.forward` applies to its real final hidden
    state before `lm_head`. Deliberately does NOT re-apply
    `final_layer_norm` (see DECISIONS.md — direct logit attribution, not
    classical full-residual logit lens).

    Returns
    -------
    Tensor, shape (vocab_size,).
    """
    scaled = contribution
    if getattr(model.config, "scale_decoder_outputs", False):
        scaled = contribution * (model.config.d_model ** -0.5)
    with torch.no_grad():
        logits = model.lm_head(scaled)
    return logits[0, 0]


def top_k_tokens(tokenizer, logits: torch.Tensor, k: int) -> Tuple[List[str], List[float]]:
    """Top-k SentencePiece token strings and their raw logit values."""
    values, indices = torch.topk(logits, k)
    tokens = [tokenizer.convert_ids_to_tokens(int(idx)) for idx in indices]
    return tokens, [float(v) for v in values.tolist()]


def token_rank_and_logit(logits: torch.Tensor, token_id: int) -> Tuple[int, float]:
    """1-indexed rank (1 = highest logit) and raw logit of one specific token id."""
    logit_value = float(logits[token_id].item())
    rank = int((logits > logits[token_id]).sum().item()) + 1
    return rank, logit_value


# ---------------------------------------------------------------------------
# Whole-block contributions (all 12 decoder layers, one forward pass)
# ---------------------------------------------------------------------------

def cache_whole_block_contributions(
    model: nn.Module, enc: Dict[str, torch.Tensor], device: torch.device
) -> Dict[int, torch.Tensor]:
    """
    All 12 decoder cross-attention contributions from ONE forward pass,
    reusing Experiment 1's cache_all_activations_from_enc.
    """
    cache = cache_all_activations_from_enc(model, enc, device)
    n_dec = model.config.num_decoder_layers
    return {i: cache[f"decoder_cross_attn_layer{i}"].to(device) for i in range(n_dec)}


# ---------------------------------------------------------------------------
# Per-head contributions (flagged heads only, exact via W_o linearity)
# ---------------------------------------------------------------------------

def head_geometry(model: nn.Module) -> Tuple[int, int]:
    return model.config.num_heads, model.config.d_kv


def cache_per_head_contributions(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
    layer_head_pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Per-head contribution to the residual stream, W_o(zero_pad(head_h_slice)),
    for each requested (layer, head) pair. Because W_o has no bias (verified
    empirically for castorini/monot5-base-msmarco), this is an EXACT
    decomposition of the layer's full contribution, not an approximation:
    summing every head's W_o(zero_pad(...)) reproduces the whole-block
    contribution exactly (tested).

    Each (layer, head) pair triggers one full forward pass (a pre-hook on
    that layer's `.o` projection captures the pre-projection input; the
    model must actually run to produce it). Only called for the small
    Experiment-3-flagged set, so this cost is bounded regardless of grid size.
    """
    n_heads, d_kv = head_geometry(model)
    results: Dict[Tuple[int, int], torch.Tensor] = {}
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    dec = torch.tensor([[model.config.decoder_start_token_id]], device=device)

    for layer_idx, head_idx in layer_head_pairs:
        o_proj = model.decoder.block[layer_idx].layer[1].EncDecAttention.o
        captured: Dict[str, torch.Tensor] = {}

        def prehook(module, args, captured=captured):
            captured["input"] = args[0]
            return args

        handle = o_proj.register_forward_pre_hook(prehook)
        try:
            with torch.no_grad():
                model(**enc_dev, decoder_input_ids=dec, use_cache=False)
        finally:
            handle.remove()

        full_input = captured["input"]
        masked = torch.zeros_like(full_input)
        sl = slice(head_idx * d_kv, (head_idx + 1) * d_kv)
        masked[:, :, sl] = full_input[:, :, sl]
        with torch.no_grad():
            head_contribution = o_proj(masked)
        results[(layer_idx, head_idx)] = head_contribution

    return results
