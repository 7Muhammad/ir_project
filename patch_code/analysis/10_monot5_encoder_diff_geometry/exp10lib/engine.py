"""
exp10lib/engine.py
====================
Per-example driver: run the encoder once on the padded-control (Type B) and
attacked (Type C) inputs, pull out raw per-position hidden states at the
requested layers via HuggingFace's ``output_hidden_states=True``, and
compute the position-wise difference vectors + position tags.

Why plain ``output_hidden_states=True`` instead of Experiment 1's
activation-hook cache
------------------------------------------------------------------------
Experiment 1's ``src/activation_hooks.py`` caches each sub-layer's residual
*contribution* (``out - in``), for activation-patching. This experiment
needs the raw, fully-accumulated hidden state at a given encoder depth (the
same quantity DecoderLens reads out, Experiment 8) rather than one
sub-layer's contribution — so it uses the encoder's own standard
``output_hidden_states`` mechanism directly:
``hidden_states[0]`` = post-embedding, ``hidden_states[L]`` = output of
encoder block L (``L=1..12``). No cached ``.pt`` files from Experiment 1
are reused for this reason (see DECISIONS.md and the reuse map): the two
experiments' cached tensors are not the same physical quantity, so
recomputing from encoder hidden states is not a redundant re-derivation, it
is the correct quantity for a different question.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from exp10lib.run_utils import ExampleInputs
from exp10lib.tagging import tag_example_positions


def encoder_hidden_states(
    model: T5ForConditionalGeneration,
    enc: Dict[str, torch.Tensor],
    layers: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Run the encoder once; return {layer_index: (seq_len, d_model) numpy array}."""
    with torch.no_grad():
        out = model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
    return {L: out.hidden_states[L][0].detach().cpu().numpy() for L in layers}


def compute_example_diffs(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    query: str,
    passage: str,
    inputs: ExampleInputs,
    layers: Sequence[int],
) -> Tuple[str, str, Optional[Dict[int, np.ndarray]], Optional[List[str]]]:
    """
    Compute {layer: diff (seq_len, d_model)} = attack_hidden - control_hidden
    and the per-position tags for one example.

    Returns (status, reason, diffs_or_None, tags_or_None). status is "ok",
    or "span_failed" (Experiment 6-style query/document probe mismatch —
    should be rare and is independent of Experiment 1's own alignment
    check, which already passed by construction of ``inputs``).
    """
    tags, spans = tag_example_positions(tokenizer, query, passage, inputs.align_result, inputs.seq_len)
    if tags is None:
        return "span_failed", spans.reason, None, None

    attack_hs = encoder_hidden_states(model, inputs.attack_enc, layers)
    control_hs = encoder_hidden_states(model, inputs.control_enc, layers)
    diffs = {L: attack_hs[L] - control_hs[L] for L in layers}
    return "ok", "", diffs, tags
