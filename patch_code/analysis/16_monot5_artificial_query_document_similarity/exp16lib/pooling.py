"""
exp16lib/pooling.py
====================
Masked mean pooling and cosine similarity (the locked primary metric):

    Q = sum_s M_q[s] H[s] / sum_s M_q[s]
    D = sum_s M_d[s] H[s] / sum_s M_d[s]
    sim = cos(Q, D)

No extra LayerNorm, no token-pair matrices. Pooling sums are accumulated in
the state's dtype (fp32); the cosine of the two pooled vectors is taken in
float64.
"""

from __future__ import annotations

import torch


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h [B, S, d], mask [B, S] in {0,1} -> [B, d]. Every row must select >= 1 token."""
    m = mask.to(h.dtype)
    counts = m.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("masked_mean: a row has an empty mask")
    return (h * m.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise cosine of [B, d] tensors, computed in float64."""
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)


def pooled_cosine(h: torch.Tensor, query_mask: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
    if bool((query_mask.bool() & doc_mask.bool()).any()):
        raise ValueError("query and document masks overlap")
    return cosine(masked_mean(h, query_mask), masked_mean(h, doc_mask))
