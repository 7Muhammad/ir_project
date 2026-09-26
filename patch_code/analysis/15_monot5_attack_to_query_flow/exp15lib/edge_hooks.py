"""
exp15lib/edge_hooks.py
=======================
Hooks for capturing P / V / z at encoder self-attention layers and for the
edge-message intervention on the `.o` input (pre-`o_proj` per-head
representation — Exp 03/11 convention, DECISIONS.md item 13).

Capture
-------
* P : `model.encoder(..., output_attentions=True).attentions[L]`, shape
      (1, H, T, T), the post-softmax probabilities T5Attention returns
      (masked keys get exactly 0 in fp32 — additive finfo.min before softmax).
* V : forward hook on `SelfAttention.v` (output (1, T, H*d_kv)), split into
      heads exactly as T5Attention does.
* z : forward PRE-hook on `SelfAttention.o` (input (1, T, H*d_kv)).

Intervention
------------
`make_edge_patch_pre_hook` receives flat index tensors describing, for every
batch row that intervenes at this layer, the (row, target position, head)
triples plus the receiver's and donor's messages for each triple, and sets

    z[row, q, head] = z[row, q, head] - m_receiver + m_donor

on exactly those triples. Every other (row, position, head) entry of the
`.o` input is returned bit-identical. Triples are unique by construction
(each row patches one head, each target position at most once), so the
advanced-index assignment has no duplicate-write ambiguity.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn

import exp15lib  # noqa: F401
from exp11lib.head_hooks import get_attention_module, get_o_proj  # noqa: F401  (reused)


def make_v_capture_hook(store: Dict, key) -> Callable:
    def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        store[key] = output.detach().clone()
    return hook


def make_o_input_capture_pre_hook(store: Dict, key) -> Callable:
    def hook(module: nn.Module, args: tuple) -> None:
        store[key] = args[0].detach().clone()
    return hook


def make_edge_patch_pre_hook(
    row_idx: torch.Tensor,   # (N,) long — batch row
    pos_idx: torch.Tensor,   # (N,) long — target (query) token position
    head_idx: torch.Tensor,  # (N,) long — head within this layer
    m_recv: torch.Tensor,    # (N, d_kv) receiver's own attack-source message
    m_don: torch.Tensor,     # (N, d_kv) donor's attack-source message
    n_heads: int,
    d_kv: int,
) -> Callable:
    def hook(module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]                                   # (B, T, H*d_kv)
        B, T, inner = hidden.shape
        if inner != n_heads * d_kv:
            raise RuntimeError(f"unexpected .o input width {inner}")
        x = hidden.clone().view(B, T, n_heads, d_kv)
        cur = x[row_idx, pos_idx, head_idx]                # (N, d_kv)
        x[row_idx, pos_idx, head_idx] = cur - m_recv.to(cur.dtype) + m_don.to(cur.dtype)
        return (x.view(B, T, inner),) + args[1:]
    return hook


def remove_all(handles: List) -> None:
    for h in handles:
        h.remove()
