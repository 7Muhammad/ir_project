"""
exp15lib/reference.py
======================
Straightforward, UNOPTIMIZED reference implementation of one edge-message
intervention (check N). Deliberately shares none of the optimized path's
shortcuts:

* no `output_attentions`, no `.v` capture hook, no precomputed messages:
  inside the actual forward pass, a pre-hook on the SelfAttention module
  records its live inputs (hidden_states, mask, position_bias) and the `.o`
  pre-hook RE-DERIVES P and V from the module's own q/k/v weights, mirroring
  T5Attention.forward line by line (transformers 5.9);
* messages are summed with an explicit Python loop over a in A;
* the edit is applied with an explicit loop over target positions;
* batch size 1, one intervention per pass.

    donor pass     : unpatched donor run -> m_donor(q) (live)
    receiver pass  : receiver run; at layer L head h, for q in targets:
                     z[q, h] = z[q, h] - m_receiver_live(q) + m_donor(q)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

import exp15lib  # noqa: F401
from exp11lib.engine import decoder_pass_scores_batched
from exp11lib.head_hooks import get_attention_module, head_geometry

from exp15lib.engine import DIRECTIONS


def _live_P_V(module: nn.Module, hidden_states, mask, position_bias):
    """Mirror of T5Attention.forward's self-attention probability computation."""
    B, T, _ = hidden_states.shape
    H, dkv = module.n_heads, module.key_value_proj_dim
    q = module.q(hidden_states).view(B, T, H, dkv).transpose(1, 2)
    k = module.k(hidden_states).view(B, T, H, dkv).transpose(1, 2)
    v = module.v(hidden_states).view(B, T, H, dkv).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(3, 2))
    if position_bias is None:
        if not module.has_relative_attention_bias:
            position_bias = torch.zeros((1, H, T, T), device=scores.device, dtype=scores.dtype)
        else:
            position_bias = module.compute_bias(T, T, device=scores.device)
        if mask is not None:
            position_bias = position_bias + mask[:, :, :, :T]
    scores = scores + position_bias
    P = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    return P[0], v[0]          # (H, T, T), (H, T, d_kv)


def _loop_message(P_h, V_h, A: Sequence[int], q: int) -> torch.Tensor:
    m = torch.zeros(V_h.shape[-1], dtype=V_h.dtype, device=V_h.device)
    for a in A:
        m = m + P_h[q, a] * V_h[a]
    return m


def _run(model, enc, layer, head, A, targets, true_id, false_id, donor_msgs=None):
    """If donor_msgs is None: record live messages. Else: apply the edit."""
    attn = get_attention_module(model, layer)
    _, d_kv, _ = head_geometry(model)
    live: Dict = {}
    recorded: Dict[int, torch.Tensor] = {}

    def attn_pre(module, args, kwargs):
        live["hs"] = args[0] if args else kwargs["hidden_states"]
        live["mask"] = kwargs.get("mask")
        live["pb"] = kwargs.get("position_bias")

    def o_pre(module, args):
        P, V = _live_P_V(attn, live["hs"], live["mask"], live["pb"])
        z = args[0].clone()
        for q in targets:
            m_recv = _loop_message(P[head], V[head], A, q)
            recorded[q] = m_recv
            if donor_msgs is not None:
                sl = slice(head * d_kv, (head + 1) * d_kv)
                z[0, q, sl] = z[0, q, sl] - m_recv + donor_msgs[q]
        return (z,) + args[1:]

    h1 = attn.register_forward_pre_hook(attn_pre, with_kwargs=True)
    h2 = attn.o.register_forward_pre_hook(o_pre)
    try:
        with torch.no_grad():
            out = model.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        score = decoder_pass_scores_batched(
            model, out.last_hidden_state, enc["attention_mask"], true_id, false_id)[0].item()
    finally:
        h1.remove()
        h2.remove()
    return score, recorded


def reference_patched_score(
    model: nn.Module,
    encs: Dict[str, Dict[str, torch.Tensor]],
    direction: str,
    layer: int,
    head: int,
    A: Sequence[int],
    targets: List[int],
    true_id: int,
    false_id: int,
) -> float:
    """`targets` are absolute token positions (not indices into Q)."""
    recv, don = DIRECTIONS[direction]
    _, donor_msgs = _run(model, encs[don], layer, head, A, targets, true_id, false_id)
    score, _ = _run(model, encs[recv], layer, head, A, targets, true_id, false_id, donor_msgs)
    return score
