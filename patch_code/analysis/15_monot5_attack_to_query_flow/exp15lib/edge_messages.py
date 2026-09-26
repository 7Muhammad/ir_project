"""
exp15lib/edge_messages.py
==========================
Source-specific edge messages for one encoder self-attention head.

Tensor layout (verified against the installed transformers 5.9
T5Attention.forward, modeling_t5.py lines ~264-325):

    value_states = self.v(h).view(B, T, H, d_kv).transpose(1, 2)   # V: (B, H, S, d_kv)
    attn_weights = softmax(q k^T + position_bias [+ mask])          # P: (B, H, T, S)  (post-softmax)
    attn_output  = (attn_weights @ value_states)                    #    (B, H, T, d_kv)
                   .transpose(1, 2).reshape(B, T, H * d_kv)         # z : `.o` input, head h = slice [h*d_kv:(h+1)*d_kv]
    out          = self.o(attn_output)                              # W_o has no bias

Hence for head h, target position q:

    z_h(q) = sum_j P[h, q, j] V[h, j]                       (full head output, d_kv)
    m_h(q) = sum_{a in A} P[h, q, a] V[h, a]                (attack -> q message, d_kv)

Everything here operates on ONE example (batch index already removed):
    P_h : (T, T)      V_h : (T, d_kv)
"""

from __future__ import annotations

from typing import Sequence

import torch


def split_heads_values(v_out: torch.Tensor, n_heads: int, d_kv: int) -> torch.Tensor:
    """`.v` Linear output (1, T, H*d_kv) -> V (H, T, d_kv), same view/transpose as T5Attention."""
    if v_out.dim() != 3 or v_out.shape[0] != 1 or v_out.shape[2] != n_heads * d_kv:
        raise ValueError(f"unexpected .v output shape {tuple(v_out.shape)}")
    T = v_out.shape[1]
    return v_out.view(1, T, n_heads, d_kv).transpose(1, 2)[0]


def full_head_output(P_h: torch.Tensor, V_h: torch.Tensor, targets: Sequence[int]) -> torch.Tensor:
    """sum over ALL sources j of P[q, j] V[j], for q in targets -> (n_targets, d_kv)."""
    return P_h[list(targets)] @ V_h


def edge_message(
    P_h: torch.Tensor, V_h: torch.Tensor, sources: Sequence[int], targets: Sequence[int],
) -> torch.Tensor:
    """m(q) = sum_{a in sources} P[q, a] V[a], for q in targets -> (n_targets, d_kv)."""
    src = list(sources)
    return P_h[list(targets)][:, src] @ V_h[src]
