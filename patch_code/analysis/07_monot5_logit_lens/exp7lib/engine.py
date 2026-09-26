"""
exp7lib/engine.py
===================
Per-example driver combining Analyses 1, 4, 5 (general logit lens,
query/doc/attack composition, attack-token tracking) — all three need the
same underlying contribution capture + unembedding projection, so they are
computed together in one pass per (run_type, layer) or (run_type, layer,
head) cell rather than three separate passes over the model.

Nested fields (top_k tokens/logits, bucket_composition) are JSON-encoded
strings in the output rows — the simplest robust way to store variable-
length lists / dicts in a flat per-example CSV.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import T5Tokenizer

from exp7lib.lens import (
    cache_per_head_contributions,
    cache_whole_block_contributions,
    project_to_logits,
    token_rank_and_logit,
    top_k_tokens,
)
from exp7lib.run_utils import ExampleInputs
from exp7lib.tagging import TaggingVocab, bucket_composition, build_tagging_vocab

RUN_TYPES = ["clean", "control", "attack"]


def _lens_row(
    model: nn.Module,
    tokenizer: T5Tokenizer,
    contribution: torch.Tensor,
    k_values: List[int],
    composition_k: int,
    vocab: TaggingVocab,
    attack_token_id: int,
) -> Dict:
    logits = project_to_logits(model, contribution)

    row: Dict = {}
    comp_tokens: Optional[List[str]] = None
    for k in k_values:
        toks, vals = top_k_tokens(tokenizer, logits, k)
        row[f"top_k{k}_tokens"] = json.dumps(toks)
        row[f"top_k{k}_logits"] = json.dumps(vals)
        if k == composition_k:
            comp_tokens = toks

    if comp_tokens is None:
        comp_tokens, _ = top_k_tokens(tokenizer, logits, composition_k)
    row["bucket_composition"] = json.dumps(bucket_composition(comp_tokens, vocab))

    rank, logit_value = token_rank_and_logit(logits, attack_token_id)
    row["attack_token_rank"] = rank
    row["attack_token_logit"] = logit_value
    return row


def run_example(
    model: nn.Module,
    tokenizer: T5Tokenizer,
    inputs: ExampleInputs,
    query: str,
    passage: str,
    attack_token_text: str,
    flagged_heads: List[Tuple[int, int]],
    k_values: List[int],
    composition_k: int,
    device: torch.device,
) -> List[Dict]:
    """
    Produce one row per (run_type, layer) [whole-block, head=None] plus one
    row per (run_type, layer, head) for every flagged head — Analyses 1, 4,
    5 combined into the unified row schema.
    """
    vocab = build_tagging_vocab(query, passage, attack_token_text)
    encodings = {"clean": inputs.clean_enc, "control": inputs.control_enc, "attack": inputs.attack_enc}

    rows: List[Dict] = []
    for run_type in RUN_TYPES:
        enc = encodings[run_type]

        whole_block = cache_whole_block_contributions(model, enc, device)
        for layer_idx, contribution in whole_block.items():
            row = _lens_row(model, tokenizer, contribution, k_values, composition_k, vocab, inputs.attack_token_id)
            row.update({"run_type": run_type, "layer": layer_idx, "head": None})
            rows.append(row)

        if flagged_heads:
            per_head = cache_per_head_contributions(model, enc, device, flagged_heads)
            for (layer_idx, head_idx), contribution in per_head.items():
                row = _lens_row(model, tokenizer, contribution, k_values, composition_k, vocab, inputs.attack_token_id)
                row.update({"run_type": run_type, "layer": layer_idx, "head": head_idx})
                rows.append(row)

    return rows
