"""Tests 11-12: masked mean and cosine on synthetic data."""

from __future__ import annotations

import math

import pytest
import torch

from exp16lib.pooling import cosine, masked_mean, pooled_cosine


def test_11_masked_mean_exact():
    h = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [100.0, 100.0], [5.0, 4.0]]])
    m = torch.tensor([[1, 1, 0, 1]])
    assert torch.allclose(masked_mean(h, m), torch.tensor([[3.0, 2.0]]))
    with pytest.raises(ValueError):
        masked_mean(h, torch.zeros(1, 4, dtype=torch.long))


def test_12_cosine_known_values():
    a = torch.tensor([[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[0.0, 3.0], [2.0, 4.0], [-2.0, 0.0], [1.0, 0.0]])
    c = cosine(a, b)
    assert c.dtype == torch.float64
    assert abs(c[0]) < 1e-12          # orthogonal
    assert abs(c[1] - 1) < 1e-12      # identical direction
    assert abs(c[2] + 1) < 1e-12      # opposite
    assert abs(c[3] - 1 / math.sqrt(2)) < 1e-12


def test_pooled_cosine_rejects_overlapping_masks():
    h = torch.randn(1, 4, 3)
    with pytest.raises(ValueError):
        pooled_cosine(h, torch.tensor([[1, 1, 0, 0]]), torch.tensor([[0, 1, 1, 0]]))
    q = torch.tensor([[1, 0, 0, 0]])
    d = torch.tensor([[0, 0, 1, 1]])
    exp = cosine(h[:, 0], h[:, 2:].mean(1))
    assert torch.allclose(pooled_cosine(h, q, d), exp)
