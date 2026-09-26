"""
exp14lib/position_masks.py
============================
The five encoder intervention-position conditions (task spec section 6):
document, query, template, query_document, all_valid.

Reuses Experiment 6's exact position definitions, unmodified — no new probe
strings, no new "template" definition:
  - exp6lib.spans.find_query_and_doc_spans   -> query_span, doc_span (Type
    B/C indexing, i.e. including the injected attack-token positions —
    "document" therefore includes the attack tokens themselves, matching
    the task spec's explicit instruction in section 6).
  - exp6lib.template_positions.get_template_positions -> the 7 named
    template/prompt sub-regions (Query, ':_after_query', Document,
    ':_after_document', Relevant, ':_after_relevant', '</s>'), whose union
    is exactly the complement of query_span/doc_span within the sequence
    (see that module's docstring). "template" here is that union collapsed
    to one mask, since Experiment 6 only ever consumed them individually —
    see DECISIONS.md item 6.

All masks are (1, seq_len, 1) float tensors, 1.0 at positions to intervene
on, matching exp6lib.engine.build_position_mask's shape convention so the
same hook-application code works for both.

Two entry points, not one, because "document" means different token ranges
in the attacked/control encoding (Type B/C: includes the attack-token
positions, doc_span_control_attack) versus the clean Type-A encoding used
for clean-input damage evaluation (doc_span_original, no attack tokens,
different sequence length) — see task spec section 10 ("apply the same
encoder position mask being evaluated" to the clean input).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import T5Tokenizer

from exp6lib.spans import QueryDocSpans, find_query_and_doc_spans
from exp6lib.template_positions import get_template_positions

CONDITION_NAMES: List[str] = ["document", "query", "template", "query_document", "all_valid"]


def _span_mask(seq_len: int, span: Tuple[int, int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, seq_len, 1, device=device)
    mask[0, span[0]:span[1], 0] = 1.0
    return mask


def _index_mask(seq_len: int, indices: List[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, seq_len, 1, device=device)
    for i in indices:
        mask[0, i, 0] = 1.0
    return mask


def build_position_masks(
    tokenizer: T5Tokenizer,
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
    input_ids: List[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build the 5 named masks given already-resolved query/doc spans and the matching input_ids."""
    seq_len = len(input_ids)
    query_mask = _span_mask(seq_len, query_span, device)
    doc_mask = _span_mask(seq_len, doc_span, device)

    template_positions = get_template_positions(tokenizer, input_ids, query_span, doc_span)
    template_indices = sorted(i for idxs in template_positions.values() for i in idxs)
    template_mask = _index_mask(seq_len, template_indices, device)

    query_document_mask = torch.clamp(query_mask + doc_mask, max=1.0)
    all_valid_mask = torch.clamp(query_document_mask + template_mask, max=1.0)

    return {
        "document": doc_mask,
        "query": query_mask,
        "template": template_mask,
        "query_document": query_document_mask,
        "all_valid": all_valid_mask,
    }


def compute_spans(tokenizer: T5Tokenizer, query: str, passage: str, n_attack_tokens: int) -> QueryDocSpans:
    """Thin re-export of Experiment 6's span finder (n_attack_tokens only affects doc_span_control_attack)."""
    return find_query_and_doc_spans(tokenizer, query, passage, n_attack_tokens)


def masks_for_attacked_or_control(
    tokenizer: T5Tokenizer, query: str, passage: str, n_attack_tokens: int,
    input_ids: List[int], device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """5 masks for the Type-B (control) or Type-C (attacked) encoding — 'document' includes attack tokens."""
    spans = compute_spans(tokenizer, query, passage, n_attack_tokens)
    if spans.status != "ok":
        return None
    return build_position_masks(tokenizer, spans.query_span, spans.doc_span_control_attack, input_ids, device)


def masks_for_clean(
    tokenizer: T5Tokenizer, query: str, passage: str, input_ids: List[int], device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """5 masks for the Type-A clean encoding (no attack tokens) — used for clean-damage evaluation."""
    spans = compute_spans(tokenizer, query, passage, n_attack_tokens=0)
    if spans.status != "ok":
        return None
    return build_position_masks(tokenizer, spans.query_span, spans.doc_span_original, input_ids, device)
