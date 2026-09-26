"""
exp14lib/direction_fit.py
============================
Train-only mean-diff direction fitting for the 18 encoder_self_attn heads
and 31 decoder_cross_attn heads (exp14lib.head_lists), restricted to
TRAIN-split successful instances only (see DECISIONS.md item 2). One fixed
64-dim direction per head, pooled across ALL train instances that
contribute to it (mirrors Experiment 2's grid_a pooling — see DECISIONS.md
item 4).

Encoder directions are still pooled over ALL valid encoder positions,
exactly as Experiment 2's pool_activation defines for encoder_self_attn,
regardless of which position mask will later be used to APPLY the frozen
direction (task spec section 5: "position-specific direction fitting is
not part of this experiment" — only the intervention SITE is
position-restricted, not the fitting; see DECISIONS.md item 5).

Reuses Experiment 2's DirectionAccumulator/pool_activation and
cache_activations_for_direction UNCHANGED — those already cache and pool
every layer/component in one control pass + one attack pass per example;
this module only persists the 49 DirKeys matching our candidate heads
(same reuse-cost tradeoff Experiment 2 itself makes: it also caches every
layer/head and only uses a subset downstream).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch

from src.model_utils import build_padded_control_and_attack_encodings_general
from src.patching import SKIP_EPSILON

from exp2lib.direction_fit import new_accumulator
from exp2lib.direction_hooks import cache_activations_for_direction

from exp14lib.head_lists import DecoderHead, EncoderHead

DirKey = Tuple[str, str, int, int]


def candidate_dir_keys(encoder_heads: List[EncoderHead], decoder_heads: List[DecoderHead]) -> Set[DirKey]:
    keys: Set[DirKey] = set()
    for h in encoder_heads:
        keys.add(("per_head", "encoder_self_attn", h.layer, h.head_idx))
    for h in decoder_heads:
        keys.add(("per_head", "decoder_cross_attn", h.layer, h.head_idx))
    return keys


def fit_directions_train_only(
    model,
    tokenizer,
    train_examples: List[Dict],
    encoder_heads: List[EncoderHead],
    decoder_heads: List[DecoderHead],
    max_length: int,
    device,
    true_id: int,
    false_id: int,
) -> Tuple[Dict[DirKey, "torch.Tensor"], Dict[DirKey, Dict[str, int]], Dict]:
    """
    One control + one attack forward pass per TRAIN example, accumulated via
    Experiment 2's DirectionAccumulator, restricted at the end to the 49
    candidate DirKeys.

    Returns (directions, counts, stats). `directions` may have fewer than
    49 keys if some candidate head never had a common control+attack
    example (e.g. every train instance for that attack failed alignment) —
    callers must handle missing keys explicitly, never silently skip.
    """
    acc = new_accumulator(model)
    n_encoder_layers = model.config.num_layers
    n_decoder_layers = model.config.num_decoder_layers
    wanted_keys = candidate_dir_keys(encoder_heads, decoder_heads)

    n_used = 0
    n_align_failed = 0
    n_skipped_epsilon = 0
    for ex in train_examples:
        control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
            tokenizer=tokenizer, query=ex["query"], passage=ex["passage"],
            attacked_passage=ex["attacked_passage"], max_length=max_length, device=device,
        )
        if align_result.status != "ok":
            n_align_failed += 1
            continue

        ctrl_whole, ctrl_head, control_score = cache_activations_for_direction(
            model, control_enc, true_id, false_id, device, n_encoder_layers, n_decoder_layers,
        )
        atk_whole, atk_head, attack_score = cache_activations_for_direction(
            model, attack_enc, true_id, false_id, device, n_encoder_layers, n_decoder_layers,
        )
        if abs(attack_score - control_score) < SKIP_EPSILON:
            n_skipped_epsilon += 1
            continue

        acc.add_example("control", ctrl_whole, ctrl_head, control_enc["attention_mask"])
        acc.add_example("attack", atk_whole, atk_head, attack_enc["attention_mask"])
        n_used += 1

    all_directions = acc.compute_directions()
    directions = {k: v for k, v in all_directions.items() if k in wanted_keys}
    counts = {
        k: {
            "n_control": acc.counts["control"].get(k, 0),
            "n_attack": acc.counts["attack"].get(k, 0),
        }
        for k in wanted_keys
    }
    stats = {
        "n_train_examples_seen": len(train_examples),
        "n_used": n_used,
        "n_align_failed": n_align_failed,
        "n_skipped_epsilon": n_skipped_epsilon,
        "n_directions_found": len(directions),
        "n_directions_expected": len(wanted_keys),
    }
    return directions, counts, stats


def direction_norm_rows(
    directions: Dict[DirKey, "torch.Tensor"], counts: Dict[DirKey, Dict[str, int]], tag: str,
) -> List[dict]:
    """Row schema matches exp2lib.run_utils.save_norm_rows's fixed fieldnames exactly ('tier', not 'tag')."""
    rows = []
    for (granularity, component, layer, head_idx), vec in directions.items():
        c = counts.get((granularity, component, layer, head_idx), {"n_control": 0, "n_attack": 0})
        rows.append({
            "tier": tag, "granularity": granularity, "component": component,
            "layer": layer, "head_idx": head_idx,
            "direction_norm": float(torch.linalg.norm(vec).item()),
            "n_control": c["n_control"], "n_attack": c["n_attack"],
        })
    return rows
