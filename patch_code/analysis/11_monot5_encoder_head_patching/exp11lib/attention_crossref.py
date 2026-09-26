"""
exp11lib/attention_crossref.py
================================
Experiment 11, Step 4 -- per-head document<->query attention mass, and its
cross-reference against Step 1/2's causally-important heads.

Reuses Experiment 6's query/document span-finding (exp6lib.spans, via
exp6lib.run_utils.build_example_inputs) and attention-computation machinery
(exp6lib.attention.compute_encoder_attentions), per the experiment prompt's
explicit instruction. NOT saved anywhere by Experiment 6: its
`normalized_attention_masses` immediately averages over heads before any
CSV write, so no per-head attention tensor survives to disk -- this module
recomputes attention fresh (stated explicitly, per the prompt's instruction
not to silently recompute under different conditions than what was asked).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def per_head_masses(
    attn_layer: torch.Tensor,
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
) -> List[Dict]:
    """
    Per-head normalized_mass(document->query) and normalized_mass(query->document)
    for one encoder layer's raw attention tensor.

    Parameters
    ----------
    attn_layer : (1, n_heads, seq_len, seq_len) -- one entry of
        exp6lib.attention.compute_encoder_attentions' return value, NOT
        averaged over heads (Experiment 6 averages before this point; we
        deliberately skip that step).
    query_span, doc_span : (start, end) exclusive token ranges.

    Returns
    -------
    List[Dict], one per head: {"head", "attention_d_to_q", "attention_q_to_d"}
    """
    n_heads = attn_layer.shape[1]
    total_len = attn_layer.shape[-1]
    q_share = (query_span[1] - query_span[0]) / total_len
    d_share = (doc_span[1] - doc_span[0]) / total_len

    rows = []
    for h in range(n_heads):
        attn_h = attn_layer[0, h]  # (seq_len, seq_len)
        d_to_q_block = attn_h[doc_span[0]:doc_span[1], query_span[0]:query_span[1]]
        q_to_d_block = attn_h[query_span[0]:query_span[1], doc_span[0]:doc_span[1]]
        rows.append({
            "head": h,
            "attention_d_to_q": d_to_q_block.sum(dim=1).mean().item() / q_share,
            "attention_q_to_d": q_to_d_block.sum(dim=1).mean().item() / d_share,
        })
    return rows
