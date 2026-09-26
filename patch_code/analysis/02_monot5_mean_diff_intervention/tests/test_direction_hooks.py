"""Tests for exp2lib/direction_hooks.py: cache-only hooks must not alter the
forward pass, and must produce the expected keys/shapes for all three
DIRECTION_COMPONENTS at every layer."""

from __future__ import annotations

import torch

from conftest import DEVICE, FALSE_ID, TRUE_ID, D_KV, N_HEADS

from src.model_utils import score_from_encoding
from exp2lib.direction_hooks import DIRECTION_COMPONENTS, cache_activations_for_direction


def test_cache_keys_and_shapes(tiny_model, control_enc):
    n_enc = tiny_model.config.num_layers
    n_dec = tiny_model.config.num_decoder_layers
    whole, per_head, score = cache_activations_for_direction(
        tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE, n_enc, n_dec,
    )

    expected_keys = set()
    for comp in DIRECTION_COMPONENTS:
        n_layers = n_enc if comp == "encoder_self_attn" else n_dec
        for layer in range(n_layers):
            expected_keys.add((comp, layer))

    assert set(whole.keys()) == expected_keys
    assert set(per_head.keys()) == expected_keys

    seq_len = control_enc["input_ids"].shape[1]
    d_model = tiny_model.config.d_model
    inner_dim = N_HEADS * D_KV

    for (comp, _layer), tensor in whole.items():
        expected_seq = seq_len if comp == "encoder_self_attn" else 1
        assert tensor.shape == (1, expected_seq, d_model)

    for (comp, _layer), tensor in per_head.items():
        expected_seq = seq_len if comp == "encoder_self_attn" else 1
        assert tensor.shape == (1, expected_seq, inner_dim)


def test_hooks_do_not_alter_score(tiny_model, control_enc):
    """Cache hooks record activations but must not change the forward pass:
    the score with hooks registered must exactly match a plain forward."""
    n_enc = tiny_model.config.num_layers
    n_dec = tiny_model.config.num_decoder_layers
    _whole, _per_head, hooked_score = cache_activations_for_direction(
        tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE, n_enc, n_dec,
    )
    plain_score = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    assert hooked_score == plain_score


def test_no_hooks_leaked(tiny_model, control_enc):
    """After caching, no forward hooks/pre-hooks should remain registered."""
    n_enc = tiny_model.config.num_layers
    n_dec = tiny_model.config.num_decoder_layers
    cache_activations_for_direction(
        tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE, n_enc, n_dec,
    )
    for block in tiny_model.encoder.block:
        assert len(block.layer[0].SelfAttention.o._forward_pre_hooks) == 0
        assert len(block.layer[0]._forward_hooks) == 0
    for block in tiny_model.decoder.block:
        assert len(block.layer[0].SelfAttention.o._forward_pre_hooks) == 0
        assert len(block.layer[1].EncDecAttention.o._forward_pre_hooks) == 0
