"""Tests for exp2lib/direction_fit.py: pooling rule and mean-diff arithmetic
on synthetic (no-model) data."""

from __future__ import annotations

import torch

from exp2lib.direction_fit import DirectionAccumulator, pool_activation


def test_pool_decoder_squeezes_single_position():
    tensor = torch.arange(6.0).reshape(1, 1, 6)
    mask = torch.ones(1, 1, dtype=torch.long)
    pooled = pool_activation("decoder_cross_attn", tensor, mask)
    assert torch.equal(pooled, tensor.squeeze(0).squeeze(0))


def test_pool_encoder_masks_padding():
    # 3 positions, last one padded (mask=0); pooled value must ignore it.
    tensor = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])  # (1,3,2)
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_activation("encoder_self_attn", tensor, mask)
    assert torch.allclose(pooled, torch.tensor([2.0, 2.0]))  # mean(1,3), ignoring the pad


def test_accumulator_direction_is_mean_diff():
    acc = DirectionAccumulator(n_heads=2, d_kv=2)
    mask = torch.ones(1, 1, dtype=torch.long)

    # Two "control" examples for one whole-vector slot, values 0 and 2 -> mean 1.
    for val in (0.0, 2.0):
        whole = {("decoder_cross_attn", 0): torch.full((1, 1, 4), val)}
        per_head = {("decoder_cross_attn", 0): torch.full((1, 1, 4), val)}
        acc.add_example("control", whole, per_head, mask)

    # Two "attack" examples, values 4 and 8 -> mean 6.
    for val in (4.0, 8.0):
        whole = {("decoder_cross_attn", 0): torch.full((1, 1, 4), val)}
        per_head = {("decoder_cross_attn", 0): torch.full((1, 1, 4), val)}
        acc.add_example("attack", whole, per_head, mask)

    directions = acc.compute_directions()

    whole_key = ("whole_vector", "decoder_cross_attn", 0, None)
    assert torch.allclose(directions[whole_key], torch.full((4,), 5.0))  # 6 - 1

    # Per-head: head 0 occupies [0:2], head 1 occupies [2:4]; same uniform
    # values, so both heads' directions should also be 5.0 everywhere.
    for h in (0, 1):
        head_key = ("per_head", "decoder_cross_attn", 0, h)
        assert torch.allclose(directions[head_key], torch.full((2,), 5.0))


def test_accumulator_skips_keys_missing_from_one_group():
    """A key only ever seen in 'control' (never 'attack') must not appear
    in the computed directions — direction needs both means."""
    acc = DirectionAccumulator(n_heads=1, d_kv=2)
    mask = torch.ones(1, 1, dtype=torch.long)
    whole = {("decoder_cross_attn", 0): torch.zeros(1, 1, 2)}
    per_head = {("decoder_cross_attn", 0): torch.zeros(1, 1, 2)}
    acc.add_example("control", whole, per_head, mask)

    directions = acc.compute_directions()
    assert directions == {}
