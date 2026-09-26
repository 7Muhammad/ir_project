"""
Correctness tests for exp7lib/lens.py:
  1. project_to_logits applies the model's own scaling+lm_head correctly
     (matches a manual reference computation).
  2. Whole-block contribution caching matches Experiment 1's own
     make_cache_hook (same quantity, different entry point).
  3. Per-head contributions sum EXACTLY to the whole-block contribution
     (W_o has no bias for the real checkpoint's cross-attention — verified;
     the tiny random model's o-projection may or may not have a bias
     depending on T5's default, checked explicitly here rather than assumed).
  4. top_k_tokens / token_rank_and_logit basic correctness.
"""

from __future__ import annotations

import pytest
import torch

from conftest import DEVICE

from src.activation_hooks import cache_all_activations_from_enc

from exp7lib.lens import (
    cache_per_head_contributions,
    cache_whole_block_contributions,
    head_geometry,
    project_to_logits,
    token_rank_and_logit,
    top_k_tokens,
)


def test_project_to_logits_matches_manual_scaling(tiny_model):
    torch.manual_seed(0)
    contribution = torch.randn(1, 1, tiny_model.config.d_model)
    logits = project_to_logits(tiny_model, contribution)

    expected = tiny_model.lm_head(contribution * (tiny_model.config.d_model ** -0.5))[0, 0]
    assert torch.allclose(logits, expected, atol=1e-6)
    assert logits.shape == (tiny_model.config.vocab_size,)


def test_whole_block_contributions_match_experiment1_cache(tiny_model, control_enc):
    contributions = cache_whole_block_contributions(tiny_model, control_enc, DEVICE)
    reference_cache = cache_all_activations_from_enc(tiny_model, control_enc, DEVICE)

    n_dec = tiny_model.config.num_decoder_layers
    assert set(contributions.keys()) == set(range(n_dec))
    for i in range(n_dec):
        assert torch.allclose(contributions[i], reference_cache[f"decoder_cross_attn_layer{i}"].to(DEVICE), atol=1e-6)


def test_per_head_contributions_sum_to_whole_block(tiny_model, control_enc):
    """W_o linearity: summing every head's zero-padded projection must exactly
    reproduce the whole-block contribution — this is only exact if W_o has no
    bias, so we check that explicitly rather than assume it."""
    layer_idx = 1
    o_proj = tiny_model.decoder.block[layer_idx].layer[1].EncDecAttention.o
    assert o_proj.bias is None, "This test's exactness assumption requires W_o to have no bias."

    n_heads, _ = head_geometry(tiny_model)
    pairs = [(layer_idx, h) for h in range(n_heads)]
    per_head = cache_per_head_contributions(tiny_model, control_enc, DEVICE, pairs)

    summed = sum(per_head[(layer_idx, h)] for h in range(n_heads))
    whole_block = cache_whole_block_contributions(tiny_model, control_enc, DEVICE)[layer_idx]
    assert torch.allclose(summed, whole_block, atol=1e-5)


def test_per_head_contribution_isolates_one_head(tiny_model, control_enc):
    """A single head's contribution must differ from another head's (not all zero/identical)."""
    layer_idx = 0
    n_heads, _ = head_geometry(tiny_model)
    pairs = [(layer_idx, 0), (layer_idx, 1)]
    per_head = cache_per_head_contributions(tiny_model, control_enc, DEVICE, pairs)
    assert not torch.allclose(per_head[(layer_idx, 0)], per_head[(layer_idx, 1)], atol=1e-8)


def test_top_k_tokens_and_rank(tiny_model):
    logits = torch.arange(100, dtype=torch.float32)  # token 99 has the highest logit
    toks, vals = top_k_tokens(_FakeTokenizer(), logits, k=3)
    assert vals == [99.0, 98.0, 97.0]

    rank, logit_value = token_rank_and_logit(logits, token_id=97)
    assert rank == 3
    assert logit_value == 97.0

    rank_top, _ = token_rank_and_logit(logits, token_id=99)
    assert rank_top == 1


class _FakeTokenizer:
    def convert_ids_to_tokens(self, idx):
        return f"tok{idx}"
