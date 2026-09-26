"""
exp6lib/attention.py
=====================
Part 1 (descriptive): normalized query<->document attention mass per encoder
layer, on the CLEAN (Type A) and ATTACKED (Type C) inputs.

Normalized attention mass definition
--------------------------------------
For a source region A and target region B (token-index ranges), and a
per-layer, head-averaged attention matrix attn[i, j] (rows sum to 1, since
each head's softmax is per-query-position and we average over heads):

    raw_mass(A -> B)  = mean_{i in A} [ sum_{j in B} attn[i, j] ]

    normalized_mass(A -> B) = raw_mass(A -> B) / (|B| / total_seq_len)

The denominator is the fraction of the sequence that B occupies — i.e. the
mass region A WOULD send to B if attention were spread uniformly over all
positions. normalized_mass > 1 means A attends to B more than a uniform
baseline; < 1 means less.

Why average over heads (not per-head) and why Type A / Type C (not Type B)
----------------------------------------------------------------------------
Part 1 is deliberately whole-block / descriptive (per-head is out of scope
this pass, matching Part 2's own head-granularity restriction). It also
uses the CLEAN and ATTACKED inputs specifically (not the padded control):
this is a descriptive question about how attention actually behaves on real
documents vs real attacked documents, not a causal-patching baseline (Part 2
is the causal half and uses Type B by explicit design choice).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def compute_encoder_attentions(
    model: nn.Module, enc: Dict[str, torch.Tensor], device: torch.device
):
    """
    Run the encoder once with output_attentions=True.

    Returns
    -------
    Tuple[Tensor, ...] : one (1, num_heads, seq_len, seq_len) tensor per
        encoder layer (model.config.num_layers entries).
    """
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.encoder(
            input_ids=enc_dev["input_ids"],
            attention_mask=enc_dev["attention_mask"],
            output_attentions=True,
        )
    return out.attentions


def _mass(attn_avg: torch.Tensor, region_a: Tuple[int, int], region_b: Tuple[int, int]) -> float:
    """mean_{i in A} sum_{j in B} attn_avg[i, j]."""
    block = attn_avg[region_a[0]:region_a[1], region_b[0]:region_b[1]]
    return block.sum(dim=1).mean().item()


def normalized_mass_indices(
    attn_avg: torch.Tensor, indices_a: List[int], indices_b: List[int], total_len: int,
) -> float:
    """
    normalized_mass(A -> B) with A, B given as explicit token-index lists
    (rather than the contiguous (start, end) ranges `normalized_attention_masses`
    assumes) -- needed for single/grouped template-token positions and for
    non-contiguous regions like an attack's inserted-token set.

    Same definition as Eq. in the module docstring: raw mean_{i in A} sum_{j
    in B} attn_avg[i,j], normalized by B's share of the sequence.
    """
    block = attn_avg[indices_a][:, indices_b]
    raw = block.sum(dim=1).mean().item()
    b_share = len(indices_b) / total_len
    return raw / b_share


def normalized_attention_masses(
    attentions,
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
) -> list:
    """
    Per-layer normalized q->d and d->q attention mass.

    Parameters
    ----------
    attentions : output of compute_encoder_attentions (one tensor per layer)
    query_span, doc_span : (start, end) exclusive token ranges, in the
        SAME indexing as the encoding used to produce `attentions`.

    Returns
    -------
    List[Dict] — one dict per layer: {"layer", "attention_q_to_d", "attention_d_to_q"}
    """
    total_len = attentions[0].shape[-1]
    q_share = (query_span[1] - query_span[0]) / total_len
    d_share = (doc_span[1] - doc_span[0]) / total_len

    rows = []
    for layer_idx, attn in enumerate(attentions):
        attn_avg = attn[0].mean(dim=0)  # (seq_len, seq_len), averaged over heads
        q_to_d = _mass(attn_avg, query_span, doc_span) / d_share
        d_to_q = _mass(attn_avg, doc_span, query_span) / q_share
        rows.append({
            "layer": layer_idx,
            "attention_q_to_d": q_to_d,
            "attention_d_to_q": d_to_q,
        })
    return rows
