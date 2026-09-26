"""
exp11lib/hub_analysis.py
==========================
Experiment 11, Step 5 -- attack-token-as-hub analysis.

Tests whether the attack token itself acts as an active aggregator (reaches
out and pulls query/document context into itself) rather than a passive
object that gets corrupted by others' attention, at the heads Step 1/2
flagged as causally important.

Attack-token positions are reused directly from Experiment 6's alignment
exposure (`exp6lib.run_utils.ExampleInputs.attack_span_indices`, itself
`AlignmentResult.inserted_positions` from Experiment 1's src/alignment.py)
-- not recomputed.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch


def per_head_outgoing_mass(
    attn_layer: torch.Tensor,
    source_indices: List[int],
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
) -> List[Dict]:
    """
    Per-head normalized_mass(source -> query) and normalized_mass(source -> document)
    for an arbitrary (possibly non-contiguous) source index set -- the attack
    token's own positions, or a random-control token set.

    Parameters
    ----------
    attn_layer : (1, n_heads, seq_len, seq_len)
    source_indices : token indices to use as the attention SOURCE (rows).
    query_span, doc_span : (start, end) exclusive target ranges.
    """
    n_heads = attn_layer.shape[1]
    total_len = attn_layer.shape[-1]
    q_share = (query_span[1] - query_span[0]) / total_len
    d_share = (doc_span[1] - doc_span[0]) / total_len

    rows = []
    for h in range(n_heads):
        attn_h = attn_layer[0, h]  # (seq_len, seq_len)
        block = attn_h[source_indices]  # (len(source_indices), seq_len)
        to_query = block[:, query_span[0]:query_span[1]].sum(dim=1).mean().item() / q_share
        to_doc = block[:, doc_span[0]:doc_span[1]].sum(dim=1).mean().item() / d_share
        rows.append({"head": h, "outgoing_to_query": to_query, "outgoing_to_doc": to_doc})
    return rows


def pick_random_control_positions(
    doc_span: Tuple[int, int],
    attack_span_indices: List[int],
    n: int,
    seed: int,
    example_key: str,
) -> List[int]:
    """
    Deterministically pick `n` document-span token positions that are NOT
    part of the attack span, as a matched (same count) non-attack-token
    control. Seeded per-(global seed, example) for reproducibility without
    every example picking the identical offset.
    """
    attack_set = set(attack_span_indices)
    candidates = [i for i in range(doc_span[0], doc_span[1]) if i not in attack_set]
    if not candidates:
        return []
    rng = random.Random(f"{seed}:{example_key}")
    n = min(n, len(candidates))
    return sorted(rng.sample(candidates, n))
