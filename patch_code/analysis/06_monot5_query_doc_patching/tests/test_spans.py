"""Span-finding correctness, on real tokenization (not synthetic)."""

from __future__ import annotations

import pytest

from exp6lib.run_utils import build_example_inputs
from exp6lib.spans import find_query_and_doc_spans
from src.model_utils import build_padded_control_and_attack_encodings_general
from conftest import DEVICE


def test_spans_found_ok_and_roundtrip(real_tokenizer, real_example):
    spans = find_query_and_doc_spans(
        real_tokenizer, real_example["query"], real_example["passage"], n_attack_tokens=0
    )
    assert spans.status == "ok", spans.reason

    original_ids = real_tokenizer.encode(
        f"Query: {real_example['query']} Document: {real_example['passage']} Relevant:",
        add_special_tokens=True,
    )

    q0, q1 = spans.query_span
    query_decoded = real_tokenizer.decode(original_ids[q0:q1])
    assert real_example["query"].strip().lower() in query_decoded.strip().lower()

    d0, d1 = spans.doc_span_original
    doc_decoded = real_tokenizer.decode(original_ids[d0:d1])
    # decoded document span should reproduce (most of) the passage text
    assert doc_decoded.strip()[:30].lower() in real_example["passage"].lower()


def test_spans_ordering(real_tokenizer, real_example):
    spans = find_query_and_doc_spans(
        real_tokenizer, real_example["query"], real_example["passage"], n_attack_tokens=5
    )
    assert spans.status == "ok"
    q0, q1 = spans.query_span
    d0, d1 = spans.doc_span_original
    assert 0 <= q0 < q1 <= d0 < d1

    # control/attack indexing shifts doc_span end by n_attack_tokens, leaves start and query untouched
    d0_ca, d1_ca = spans.doc_span_control_attack
    assert d0_ca == d0
    assert d1_ca == d1 + 5


def test_query_span_invariant_across_control_and_attack(real_tokenizer, real_example, tmp_path):
    """query_span must be identical in Type B and Type C encodings (never shifted by the attack)."""
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=real_tokenizer,
        query=real_example["query"],
        passage=real_example["passage"],
        attacked_passage=real_example["attacked_passage"],
        max_length=512,
        device=DEVICE,
    )
    assert align_result.status == "ok"

    spans = find_query_and_doc_spans(
        real_tokenizer, real_example["query"], real_example["passage"],
        n_attack_tokens=align_result.n_inserted,
    )
    assert spans.status == "ok"

    q0, q1 = spans.query_span
    # Same query text tokens must appear at the same indices in both encodings
    ctrl_ids = control_enc["input_ids"][0].tolist()
    atk_ids = attack_enc["input_ids"][0].tolist()
    assert ctrl_ids[q0:q1] == atk_ids[q0:q1]

    # And doc_span_control_attack must exactly cover the true remaining shared+inserted region
    d0, d1 = spans.doc_span_control_attack
    assert d1 - d0 == (spans.doc_span_original[1] - spans.doc_span_original[0]) + align_result.n_inserted


def test_build_example_inputs_end_to_end(real_tokenizer, real_example):
    result = build_example_inputs(real_tokenizer, real_example, max_length=512, device=DEVICE)
    assert result.status == "ok", result.reason
    assert result.clean_enc is not None and result.control_enc is not None and result.attack_enc is not None
    # control and attack must have identical shape (patching precondition)
    assert result.control_enc["input_ids"].shape == result.attack_enc["input_ids"].shape
    q0, q1 = result.query_span
    assert q1 > q0
    d0, d1 = result.doc_span_control_attack
    assert d1 > d0
    assert d1 <= result.control_enc["input_ids"].shape[1]


def test_spans_fail_gracefully_on_garbage_input(real_tokenizer):
    """A query/passage pair that can't plausibly break alignment should still return ok;
    this test just documents the failure-mode contract (status='failed', not an exception)."""
    spans = find_query_and_doc_spans(real_tokenizer, "", "", n_attack_tokens=0)
    assert spans.status in ("ok", "failed")  # must not raise
