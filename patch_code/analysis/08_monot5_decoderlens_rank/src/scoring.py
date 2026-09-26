"""
src/scoring.py
==============
Layerwise DecoderLens scoring for candidate sets and target variants.

monoT5 score definition (same as the previous experiment):
    score = logit("true") - logit("false")

This module produces:
  * Per-candidate, per-layer CLEAN scores (one encoder pass per candidate).
    These form the fixed scoring table used to rank a target passage.
  * Per-target, per-variant, per-layer scores, where the target passage is
    swapped for one of {original, padded_control, attack}.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.data_loading import Candidate
from src.decoderlens import layerwise_scores_for_input
from src.padded_control import encode_clean

VARIANTS = ("original", "padded_control", "attack")


def score_candidates_layerwise(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    candidates: List[Candidate],
    true_id: int,
    false_id: int,
    max_length: int,
    device: torch.device,
    apply_final_layer_norm: bool = True,
) -> Dict[str, List[float]]:
    """
    Compute clean layerwise scores for every candidate in a query's pool.

    Each candidate is scored on its ORIGINAL clean passage (text_0).

    Returns
    -------
    dict {docid: [score_layer0, score_layer1, ..., score_layer12]}
    """
    scores_by_doc: Dict[str, List[float]] = {}
    for cand in candidates:
        enc = encode_clean(tokenizer, cand.query, cand.passage, max_length, device)
        per_layer = layerwise_scores_for_input(
            model, enc, true_id, false_id, apply_final_layer_norm
        )
        scores_by_doc[cand.docid] = [s for (s, _t, _f) in per_layer]
    return scores_by_doc


def score_target_variant_layerwise(
    model: T5ForConditionalGeneration,
    enc: Dict[str, torch.Tensor],
    true_id: int,
    false_id: int,
    apply_final_layer_norm: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Compute layerwise (score, true_logit, false_logit) for one variant input.

    Thin wrapper over decoderlens.layerwise_scores_for_input kept here so the
    scripts depend on a single scoring entry point.
    """
    return layerwise_scores_for_input(
        model, enc, true_id, false_id, apply_final_layer_norm
    )
