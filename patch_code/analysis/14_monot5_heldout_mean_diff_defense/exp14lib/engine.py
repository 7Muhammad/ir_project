"""
exp14lib/engine.py
=====================
Per-example drivers tying together encoding construction (Experiment 1),
position masks (Experiment 6), encoder-output reuse (Experiment 3), and the
defense hooks (exp14lib.hooks) for one candidate intervention — a decoder
head, or an (encoder head, position-mask condition) pair. Shared by the
IID (scripts 04/06) and attack-OOD (script 07) evaluation stages so the
scoring logic exists in exactly one place.

Encoder-output reuse (task spec section 12): for DECODER heads, the encoder
forward pass for a given example is identical regardless of which decoder
head/scale is being probed, so `compute_encoder_states` is called EXACTLY
ONCE per example (per base input — attacked or clean) and its
(enc_hidden, enc_mask) is reused across every one of the (up to 31) decoder
heads and every scale — exactly Experiment 3's encoder-output-reuse trick.
Encoder heads cannot use this shortcut (the intervention happens INSIDE the
encoder's own forward computation — same limitation Experiments 6/11
document), so each (encoder head, position condition) genuinely re-executes
the encoder, batched only over scales (see exp14lib/hooks.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from headlib.engine import compute_encoder_states
from src.model_utils import build_padded_control_and_attack_encodings_general

from exp2lib.run_utils import build_clean_encoding

from exp14lib.head_lists import DecoderHead, EncoderHead
from exp14lib.hooks import decoder_defense_scores_for_scales, encoder_defense_scores_for_scales
from exp14lib.metrics import raw_movement_toward_control, recovery
from exp14lib.position_masks import CONDITION_NAMES, masks_for_attacked_or_control, masks_for_clean


def build_example_encodings(tokenizer, query: str, passage: str, attacked_passage: str, max_length: int, device):
    """(control_enc, attack_enc, align_result); control_enc/attack_enc are None on alignment failure."""
    return build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=query, passage=passage,
        attacked_passage=attacked_passage, max_length=max_length, device=device,
    )


def build_clean_enc(tokenizer, query: str, passage: str, max_length: int, device):
    return build_clean_encoding(tokenizer, query, passage, max_length, device)


def defend_encoder_scores(
    model, tokenizer, query: str, passage: str, n_attack_tokens: int,
    attack_enc: Dict[str, torch.Tensor], layer: int, head_idx: int,
    direction: torch.Tensor, scales: List[float], condition: str,
    true_id: int, false_id: int, device,
) -> Optional[List[float]]:
    """Returns None if position-mask construction fails alignment — caller should skip/count it."""
    input_ids_list = attack_enc["input_ids"][0].tolist()
    masks = masks_for_attacked_or_control(tokenizer, query, passage, n_attack_tokens, input_ids_list, device)
    if masks is None:
        return None
    return encoder_defense_scores_for_scales(
        model, attack_enc["input_ids"], attack_enc["attention_mask"], layer, head_idx,
        direction, scales, masks[condition], true_id, false_id, device,
    )


def defend_clean_encoder_scores(
    model, tokenizer, query: str, passage: str, clean_enc: Dict[str, torch.Tensor],
    layer: int, head_idx: int, direction: torch.Tensor, scales: List[float], condition: str,
    true_id: int, false_id: int, device,
) -> Optional[List[float]]:
    input_ids_list = clean_enc["input_ids"][0].tolist()
    masks = masks_for_clean(tokenizer, query, passage, input_ids_list, device)
    if masks is None:
        return None
    return encoder_defense_scores_for_scales(
        model, clean_enc["input_ids"], clean_enc["attention_mask"], layer, head_idx,
        direction, scales, masks[condition], true_id, false_id, device,
    )


def _candidate_scales(get_scales_for, side: str, layer: int, head_idx: int, condition: str) -> List[float]:
    scales = get_scales_for(side, layer, head_idx, condition)
    return scales or []


def _decoder_rows_shared_encoder_state(
    model, enc_hidden, enc_mask, decoder_heads: List[DecoderHead], directions: Dict,
    scales_for_head, true_id: int, false_id: int,
    common_row_fields: Dict, score_key_pairs,
) -> Tuple[List[Dict], int, int]:
    """
    Shared decoder-head loop: computes decoder scores for every decoder
    head using the SAME (already-computed) `enc_hidden`/`enc_mask` — one
    encoder forward pass serves all decoder heads for this example (task
    spec section 12). `scales_for_head(layer, head_idx) -> List[float]`
    lets callers pass either a fixed scale sweep or a per-candidate
    selected-scale lookup. `score_key_pairs` is a list of
    (output_field_name, baseline_value) pairs merged into every row
    alongside "scale" and "score_defended"/"score_clean_defended".
    """
    rows: List[Dict] = []
    n_missing_direction = 0
    n_no_scale = 0
    for h in decoder_heads:
        key = ("per_head", "decoder_cross_attn", h.layer, h.head_idx)
        if key not in directions:
            n_missing_direction += 1
            continue
        scales = scales_for_head(h.layer, h.head_idx)
        if not scales:
            n_no_scale += 1
            continue
        defended = decoder_defense_scores_for_scales(
            model, enc_hidden, enc_mask, h.layer, h.head_idx, directions[key], scales, true_id, false_id,
        )
        for scale, sd in zip(scales, defended):
            row = dict(common_row_fields)
            row.update({
                "side": "decoder", "component": "decoder_cross_attn",
                "layer": h.layer, "head_idx": h.head_idx, "condition": "decoder", "scale": scale,
            })
            for out_key, out_val in score_key_pairs(sd):
                row[out_key] = out_val
            rows.append(row)
    return rows, n_missing_direction, n_no_scale


# ---------------------------------------------------------------------------
# Shared drivers: attacked-instance defense rows + clean-damage rows,
# reused by scripts 04 (validation), 06 (IID test), 07 (attack-OOD).
# ---------------------------------------------------------------------------

def compute_defense_rows_for_examples(
    model, tokenizer,
    attack_examples: List[Tuple[str, Dict]],   # [(attack_name, example_record), ...]
    encoder_heads: List[EncoderHead],
    decoder_heads: List[DecoderHead],
    directions: Dict,
    scales: List[float],
    max_length: int, true_id: int, false_id: int, device,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    For every (attack_name, example) and every candidate (decoder head) or
    (encoder head, position condition), compute defended scores at every
    scale and the per-scale recovery metric. Reuses the example's already-
    cached control_score/attack_score (Experiment 1) rather than rescoring.
    """
    rows: List[Dict] = []
    skip = {"align_failed": 0, "missing_direction": 0, "encoder_mask_failed": 0}

    for attack_name, ex in attack_examples:
        control_enc, attack_enc, align_result = build_example_encodings(
            tokenizer, ex["query"], ex["passage"], ex["attacked_passage"], max_length, device,
        )
        if align_result.status != "ok":
            skip["align_failed"] += 1
            continue
        score_control = ex["control_score"]
        score_attack = ex["attack_score"]
        n_attack_tokens = align_result.n_inserted

        common = {"qid": ex["qid"], "docid": ex["docid"], "attack_name": attack_name,
                  "score_control": score_control, "score_attack": score_attack}

        atk_hidden, atk_mask = compute_encoder_states(model, attack_enc, device)
        dec_rows, n_missing, _ = _decoder_rows_shared_encoder_state(
            model, atk_hidden, atk_mask, decoder_heads, directions,
            lambda layer, head_idx: scales, true_id, false_id, common,
            lambda sd: [("score_defended", sd),
                        ("recovery", recovery(score_attack, score_control, sd)),
                        ("raw_movement", raw_movement_toward_control(score_attack, score_control, sd))],
        )
        skip["missing_direction"] += n_missing
        rows.extend(dec_rows)

        for h in encoder_heads:
            key = ("per_head", "encoder_self_attn", h.layer, h.head_idx)
            if key not in directions:
                skip["missing_direction"] += len(CONDITION_NAMES)
                continue
            for condition in CONDITION_NAMES:
                defended = defend_encoder_scores(
                    model, tokenizer, ex["query"], ex["passage"], n_attack_tokens, attack_enc,
                    h.layer, h.head_idx, directions[key], scales, condition, true_id, false_id, device,
                )
                if defended is None:
                    skip["encoder_mask_failed"] += 1
                    continue
                for scale, sd in zip(scales, defended):
                    rows.append({
                        "qid": ex["qid"], "docid": ex["docid"], "attack_name": attack_name,
                        "side": "encoder", "component": "encoder_self_attn",
                        "layer": h.layer, "head_idx": h.head_idx, "condition": condition,
                        "scale": scale,
                        "score_control": score_control, "score_attack": score_attack, "score_defended": sd,
                        "recovery": recovery(score_attack, score_control, sd),
                        "raw_movement": raw_movement_toward_control(score_attack, score_control, sd),
                    })
    return rows, skip


def compute_defense_rows_selected_scales(
    model, tokenizer,
    attack_examples: List[Tuple[str, Dict]],
    encoder_heads: List[EncoderHead],
    decoder_heads: List[DecoderHead],
    directions: Dict,
    get_scales_for,   # (side, layer, head_idx, condition) -> List[float], e.g. [selected] or [selected, reference]
    max_length: int, true_id: int, false_id: int, device,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Same as compute_defense_rows_for_examples, but each candidate runs only
    its OWN (typically 1-2 element) scale list instead of the full shared
    sweep — used for held-out test/OOD evaluation, where only the
    validation-selected scale (and the fixed reference scale) are needed,
    not the full 6-scale grid (task spec section 12 efficiency: don't
    recompute more than necessary).
    """
    rows: List[Dict] = []
    skip = {"align_failed": 0, "missing_direction": 0, "encoder_mask_failed": 0, "no_scale_selected": 0}

    for attack_name, ex in attack_examples:
        control_enc, attack_enc, align_result = build_example_encodings(
            tokenizer, ex["query"], ex["passage"], ex["attacked_passage"], max_length, device,
        )
        if align_result.status != "ok":
            skip["align_failed"] += 1
            continue
        score_control = ex["control_score"]
        score_attack = ex["attack_score"]
        n_attack_tokens = align_result.n_inserted

        common = {"qid": ex["qid"], "docid": ex["docid"], "attack_name": attack_name,
                  "score_control": score_control, "score_attack": score_attack}

        atk_hidden, atk_mask = compute_encoder_states(model, attack_enc, device)
        dec_rows, n_missing, n_no_scale = _decoder_rows_shared_encoder_state(
            model, atk_hidden, atk_mask, decoder_heads, directions,
            lambda layer, head_idx: _candidate_scales(get_scales_for, "decoder", layer, head_idx, "decoder"),
            true_id, false_id, common,
            lambda sd: [("score_defended", sd),
                        ("recovery", recovery(score_attack, score_control, sd)),
                        ("raw_movement", raw_movement_toward_control(score_attack, score_control, sd))],
        )
        skip["missing_direction"] += n_missing
        skip["no_scale_selected"] += n_no_scale
        rows.extend(dec_rows)

        for h in encoder_heads:
            key = ("per_head", "encoder_self_attn", h.layer, h.head_idx)
            if key not in directions:
                skip["missing_direction"] += len(CONDITION_NAMES)
                continue
            for condition in CONDITION_NAMES:
                scales = _candidate_scales(get_scales_for, "encoder", h.layer, h.head_idx, condition)
                if not scales:
                    skip["no_scale_selected"] += 1
                    continue
                defended = defend_encoder_scores(
                    model, tokenizer, ex["query"], ex["passage"], n_attack_tokens, attack_enc,
                    h.layer, h.head_idx, directions[key], scales, condition, true_id, false_id, device,
                )
                if defended is None:
                    skip["encoder_mask_failed"] += 1
                    continue
                for scale, sd in zip(scales, defended):
                    rows.append({
                        "qid": ex["qid"], "docid": ex["docid"], "attack_name": attack_name,
                        "side": "encoder", "component": "encoder_self_attn",
                        "layer": h.layer, "head_idx": h.head_idx, "condition": condition,
                        "scale": scale,
                        "score_control": score_control, "score_attack": score_attack, "score_defended": sd,
                        "recovery": recovery(score_attack, score_control, sd),
                        "raw_movement": raw_movement_toward_control(score_attack, score_control, sd),
                    })
    return rows, skip


def compute_clean_damage_rows_multiscale(
    model, tokenizer,
    clean_examples: List[Dict],
    encoder_heads: List[EncoderHead],
    decoder_heads: List[DecoderHead],
    directions: Dict,
    scales: List[float],
    max_length: int, true_id: int, false_id: int, device,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Like compute_clean_damage_rows, but sweeps ALL `scales` per candidate
    instead of one selected scale — used only to build the VALIDATION
    recovery-vs-clean-damage trade-off curve (task spec section 8; NOT the
    primary test clean-damage result, which uses the single selected scale
    — see compute_clean_damage_rows / script 06).
    """
    rows: List[Dict] = []
    skip = {"missing_direction": 0, "encoder_mask_failed": 0}

    for ex in clean_examples:
        clean_enc = build_clean_enc(tokenizer, ex["query"], ex["passage"], max_length, device)
        score_clean = ex["original_score"]
        common = {"qid": ex["qid"], "docid": ex["docid"]}

        clean_hidden, clean_mask = compute_encoder_states(model, clean_enc, device)
        dec_rows, n_missing, _ = _decoder_rows_shared_encoder_state(
            model, clean_hidden, clean_mask, decoder_heads, directions,
            lambda layer, head_idx: scales, true_id, false_id, common,
            lambda sd: [("score_clean", score_clean), ("score_clean_defended", sd)],
        )
        skip["missing_direction"] += n_missing
        rows.extend(dec_rows)

        for h in encoder_heads:
            key = ("per_head", "encoder_self_attn", h.layer, h.head_idx)
            if key not in directions:
                skip["missing_direction"] += len(CONDITION_NAMES)
                continue
            for condition in CONDITION_NAMES:
                defended = defend_clean_encoder_scores(
                    model, tokenizer, ex["query"], ex["passage"], clean_enc,
                    h.layer, h.head_idx, directions[key], scales, condition, true_id, false_id, device,
                )
                if defended is None:
                    skip["encoder_mask_failed"] += 1
                    continue
                for scale, sd in zip(scales, defended):
                    rows.append({
                        "qid": ex["qid"], "docid": ex["docid"],
                        "side": "encoder", "layer": h.layer, "head_idx": h.head_idx,
                        "condition": condition, "scale": scale,
                        "score_clean": score_clean, "score_clean_defended": sd,
                    })
    return rows, skip


def compute_clean_damage_rows(
    model, tokenizer,
    clean_examples: List[Dict],   # [{qid, docid, query, passage, original_score}, ...]
    encoder_heads: List[EncoderHead],
    decoder_heads: List[DecoderHead],
    directions: Dict,
    get_scales_for,   # (side, layer, head_idx, condition) -> List[float]
    max_length: int, true_id: int, false_id: int, device,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    For every held-out clean pair and every candidate, apply the SAME
    frozen direction (at the candidate's selected/reference scale(s)) to
    the CLEAN input and report the score change (task spec section 10).
    """
    rows: List[Dict] = []
    skip = {"missing_direction": 0, "encoder_mask_failed": 0, "no_scale_selected": 0}

    for ex in clean_examples:
        clean_enc = build_clean_enc(tokenizer, ex["query"], ex["passage"], max_length, device)
        score_clean = ex["original_score"]
        common = {"qid": ex["qid"], "docid": ex["docid"]}

        clean_hidden, clean_mask = compute_encoder_states(model, clean_enc, device)
        dec_rows, n_missing, n_no_scale = _decoder_rows_shared_encoder_state(
            model, clean_hidden, clean_mask, decoder_heads, directions,
            lambda layer, head_idx: _candidate_scales(get_scales_for, "decoder", layer, head_idx, "decoder"),
            true_id, false_id, common,
            lambda sd: [("score_clean", score_clean), ("score_clean_defended", sd)],
        )
        skip["missing_direction"] += n_missing
        skip["no_scale_selected"] += n_no_scale
        rows.extend(dec_rows)

        for h in encoder_heads:
            key = ("per_head", "encoder_self_attn", h.layer, h.head_idx)
            if key not in directions:
                skip["missing_direction"] += len(CONDITION_NAMES)
                continue
            for condition in CONDITION_NAMES:
                scales = _candidate_scales(get_scales_for, "encoder", h.layer, h.head_idx, condition)
                if not scales:
                    skip["no_scale_selected"] += 1
                    continue
                defended = defend_clean_encoder_scores(
                    model, tokenizer, ex["query"], ex["passage"], clean_enc,
                    h.layer, h.head_idx, directions[key], scales, condition, true_id, false_id, device,
                )
                if defended is None:
                    skip["encoder_mask_failed"] += 1
                    continue
                for scale, sd in zip(scales, defended):
                    rows.append({
                        "qid": ex["qid"], "docid": ex["docid"],
                        "side": "encoder", "component": "encoder_self_attn",
                        "layer": h.layer, "head_idx": h.head_idx, "condition": condition, "scale": scale,
                        "score_clean": score_clean, "score_clean_defended": sd,
                    })
    return rows, skip
