"""Tests for exp9lib/metrics.py."""

from __future__ import annotations

import pytest

from exp9lib.metrics import spearman_correlation, success


def test_success():
    assert success(delta_rank=1) == 1
    assert success(delta_rank=0) == 0
    assert success(delta_rank=-1) == 0


def test_spearman_perfect_positive():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    x = [1, 2, 3, 4, 5]
    y = [50, 40, 30, 20, 10]
    assert spearman_correlation(x, y) == pytest.approx(-1.0)


def test_spearman_none_for_degenerate_input():
    assert spearman_correlation([1.0], [2.0]) is None       # too few points
    assert spearman_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # constant x
