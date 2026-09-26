"""Position-tagging correctness, on real tokenization (not synthetic)."""

from __future__ import annotations

from src.model_utils import build_padded_control_and_attack_encodings_general
from exp10lib.tagging import ALL_TAGS, TAG_CONNECTIVE, TAG_INJECTED, TAG_OTHER_DOC, TAG_QUERY, tag_example_positions
from conftest import DEVICE


def test_all_positions_tagged_and_only_known_tags(real_tokenizer, real_example):
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=real_tokenizer, query=real_example["query"], passage=real_example["passage"],
        attacked_passage=real_example["attacked_passage"], max_length=512, device=DEVICE,
    )
    assert align_result.status == "ok"
    seq_len = attack_enc["input_ids"].shape[1]

    tags, spans = tag_example_positions(real_tokenizer, real_example["query"], real_example["passage"], align_result, seq_len)
    assert spans.status == "ok", spans.reason
    assert tags is not None
    assert len(tags) == seq_len
    assert set(tags) <= set(ALL_TAGS)


def test_injected_positions_are_tagged_injected(real_tokenizer, real_example):
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=real_tokenizer, query=real_example["query"], passage=real_example["passage"],
        attacked_passage=real_example["attacked_passage"], max_length=512, device=DEVICE,
    )
    assert align_result.status == "ok"
    seq_len = attack_enc["input_ids"].shape[1]
    tags, spans = tag_example_positions(real_tokenizer, real_example["query"], real_example["passage"], align_result, seq_len)

    for s, e in align_result.inserted_spans:
        assert all(tags[i] == TAG_INJECTED for i in range(s, e))


def test_query_span_tagged_query_and_disjoint_from_document(real_tokenizer, real_example):
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=real_tokenizer, query=real_example["query"], passage=real_example["passage"],
        attacked_passage=real_example["attacked_passage"], max_length=512, device=DEVICE,
    )
    assert align_result.status == "ok"
    seq_len = attack_enc["input_ids"].shape[1]
    tags, spans = tag_example_positions(real_tokenizer, real_example["query"], real_example["passage"], align_result, seq_len)

    q0, q1 = spans.query_span
    assert all(tags[i] == TAG_QUERY for i in range(q0, q1))

    d0, d1 = spans.doc_span_control_attack
    # No overlap between query span and document span.
    assert q1 <= d0

    # Positions strictly before the query span and strictly after the
    # document span are template/connective boilerplate.
    if q0 > 0:
        assert tags[0] == TAG_CONNECTIVE
    if d1 < seq_len:
        assert tags[seq_len - 1] == TAG_CONNECTIVE


def test_non_injected_document_positions_tagged_other_document(real_tokenizer, real_example):
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=real_tokenizer, query=real_example["query"], passage=real_example["passage"],
        attacked_passage=real_example["attacked_passage"], max_length=512, device=DEVICE,
    )
    assert align_result.status == "ok"
    seq_len = attack_enc["input_ids"].shape[1]
    tags, spans = tag_example_positions(real_tokenizer, real_example["query"], real_example["passage"], align_result, seq_len)

    d0, d1 = spans.doc_span_control_attack
    inserted = set()
    for s, e in align_result.inserted_spans:
        inserted.update(range(s, e))

    for i in range(d0, d1):
        if i not in inserted:
            assert tags[i] == TAG_OTHER_DOC
