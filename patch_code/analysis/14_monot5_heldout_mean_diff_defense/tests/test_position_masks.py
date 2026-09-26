from __future__ import annotations

import torch

from src.model_utils import build_padded_control_and_attack_encodings_general

from exp6lib.template_positions import get_template_positions

from exp14lib.position_masks import compute_spans, masks_for_attacked_or_control

QUERY = "what is the capital of france"
PASSAGE = "Paris is the capital and most populous city of France."
ATTACKED_PASSAGE = "relevant: relevant: relevant: relevant: relevant: " + PASSAGE


def _build(tokenizer):
    device = torch.device("cpu")
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=QUERY, passage=PASSAGE, attacked_passage=ATTACKED_PASSAGE,
        max_length=512, device=device,
    )
    assert align_result.status == "ok"
    input_ids = attack_enc["input_ids"][0].tolist()
    return attack_enc, align_result, input_ids


def test_document_mask_includes_attack_tokens(tokenizer):
    """Test 5: 'document' condition includes the injected attack-token positions."""
    attack_enc, align_result, input_ids = _build(tokenizer)
    masks = masks_for_attacked_or_control(tokenizer, QUERY, PASSAGE, align_result.n_inserted, input_ids, torch.device("cpu"))
    assert masks is not None
    spans = compute_spans(tokenizer, QUERY, PASSAGE, align_result.n_inserted)
    doc_start, doc_end = spans.doc_span_control_attack
    inserted = set(align_result.inserted_positions)
    doc_positions = set(range(doc_start, doc_end))
    assert inserted.issubset(doc_positions), "attack-token positions must fall inside the document span"
    on_positions = {i for i in range(len(input_ids)) if masks["document"][0, i, 0].item() == 1.0}
    assert on_positions == doc_positions


def test_query_mask_matches_query_span(tokenizer):
    attack_enc, align_result, input_ids = _build(tokenizer)
    masks = masks_for_attacked_or_control(tokenizer, QUERY, PASSAGE, align_result.n_inserted, input_ids, torch.device("cpu"))
    spans = compute_spans(tokenizer, QUERY, PASSAGE, align_result.n_inserted)
    q_start, q_end = spans.query_span
    on_positions = {i for i in range(len(input_ids)) if masks["query"][0, i, 0].item() == 1.0}
    assert on_positions == set(range(q_start, q_end))


def test_template_positions_match_experiment6_implementation(tokenizer):
    """Test 6: 'template' mask is exactly Experiment 6's 7-named-group union — no new definition."""
    attack_enc, align_result, input_ids = _build(tokenizer)
    spans = compute_spans(tokenizer, QUERY, PASSAGE, align_result.n_inserted)
    masks = masks_for_attacked_or_control(tokenizer, QUERY, PASSAGE, align_result.n_inserted, input_ids, torch.device("cpu"))

    reference = get_template_positions(tokenizer, input_ids, spans.query_span, spans.doc_span_control_attack)
    reference_indices = {i for idxs in reference.values() for i in idxs}

    mask_indices = {i for i in range(len(input_ids)) if masks["template"][0, i, 0].item() == 1.0}
    assert mask_indices == reference_indices


def test_query_document_excludes_template(tokenizer):
    """Test 7: query_document and template masks never overlap."""
    attack_enc, align_result, input_ids = _build(tokenizer)
    masks = masks_for_attacked_or_control(tokenizer, QUERY, PASSAGE, align_result.n_inserted, input_ids, torch.device("cpu"))
    qd = masks["query_document"][0, :, 0]
    tmpl = masks["template"][0, :, 0]
    overlap = (qd * tmpl).sum().item()
    assert overlap == 0.0


def test_all_valid_is_expected_union(tokenizer):
    """Test 8: all_valid == query | document | template, and covers every position (no gaps)."""
    attack_enc, align_result, input_ids = _build(tokenizer)
    masks = masks_for_attacked_or_control(tokenizer, QUERY, PASSAGE, align_result.n_inserted, input_ids, torch.device("cpu"))
    seq_len = len(input_ids)

    union = torch.clamp(masks["query"] + masks["document"] + masks["template"], max=1.0)
    assert torch.equal(union, masks["all_valid"])
    assert masks["all_valid"].sum().item() == seq_len  # every position accounted for exactly once
    # exactly-once (no double counting) across query/document/template:
    raw_sum = (masks["query"] + masks["document"] + masks["template"]).sum().item()
    assert raw_sum == seq_len
