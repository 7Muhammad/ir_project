"""Normalized attention-mass math, against hand-computed reference values."""

from __future__ import annotations

import torch

from exp6lib.attention import normalized_attention_masses


def _fake_attentions(matrix: torch.Tensor, n_layers: int = 1, n_heads: int = 2):
    """Wrap one (seq_len, seq_len) matrix as if it were attentions from n_layers identical layers/heads."""
    seq_len = matrix.shape[0]
    per_layer = matrix.view(1, 1, seq_len, seq_len).expand(1, n_heads, seq_len, seq_len).clone()
    return tuple(per_layer for _ in range(n_layers))


def test_uniform_attention_gives_normalized_mass_one():
    seq_len = 10
    uniform = torch.full((seq_len, seq_len), 1.0 / seq_len)
    attentions = _fake_attentions(uniform)
    query_span, doc_span = (0, 3), (3, 8)

    rows = normalized_attention_masses(attentions, query_span, doc_span)
    assert len(rows) == 1
    assert abs(rows[0]["attention_q_to_d"] - 1.0) < 1e-5
    assert abs(rows[0]["attention_d_to_q"] - 1.0) < 1e-5


def test_intra_region_only_attention_gives_zero_cross_mass():
    """Each position only attends within its own region -> cross-region mass ~ 0, self-region > 1."""
    seq_len = 10
    query_span, doc_span = (0, 3), (3, 8)
    other_span_len = seq_len - (doc_span[1] - doc_span[0]) - (query_span[1] - query_span[0])

    attn = torch.zeros(seq_len, seq_len)
    attn[query_span[0]:query_span[1], query_span[0]:query_span[1]] = 1.0 / (query_span[1] - query_span[0])
    attn[doc_span[0]:doc_span[1], doc_span[0]:doc_span[1]] = 1.0 / (doc_span[1] - doc_span[0])
    # remaining rows (positions outside query/doc) attend uniformly to themselves only; irrelevant to the test

    attentions = _fake_attentions(attn)
    rows = normalized_attention_masses(attentions, query_span, doc_span)
    assert rows[0]["attention_q_to_d"] < 1e-6
    assert rows[0]["attention_d_to_q"] < 1e-6


def test_multi_layer_shape(tiny_model):
    """Sanity: a real (tiny) model's encoder attentions plug into normalized_attention_masses cleanly."""
    from exp6lib.attention import compute_encoder_attentions
    enc = {
        "input_ids": torch.randint(2, 100, (1, 12)),
        "attention_mask": torch.ones(1, 12, dtype=torch.long),
    }
    attentions = compute_encoder_attentions(tiny_model, enc, torch.device("cpu"))
    assert len(attentions) == tiny_model.config.num_layers
    rows = normalized_attention_masses(attentions, query_span=(0, 3), doc_span=(3, 10))
    assert len(rows) == tiny_model.config.num_layers
    for r in rows:
        assert r["attention_q_to_d"] >= 0
        assert r["attention_d_to_q"] >= 0
