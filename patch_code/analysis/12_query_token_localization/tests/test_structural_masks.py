from __future__ import annotations

from exp12lib.structural_masks import STRUCTURAL_GROUP_NAMES, build_structural_units


def _fake_template_positions():
    # Mirrors exp6lib.template_positions.get_template_positions's shape:
    # {"Query": [...], ":_after_query": [...], "Document": [...],
    #  ":_after_document": [...], "Relevant": [...], ":_after_relevant": [...], "</s>": [...]}
    return {
        "Query": [0, 1], ":_after_query": [2],
        "Document": [10], ":_after_document": [11],
        "Relevant": [30, 31], ":_after_relevant": [32], "</s>": [33],
    }


def test_structural_units_cover_expected_names():
    query_span = (3, 9)          # query content tokens 3..8
    doc_span = (12, 30)          # document field (incl. attack) tokens 12..29
    attack_span_indices = [12, 13, 14]  # a "start"-position attack
    units = build_structural_units(_fake_template_positions(), query_span, doc_span, attack_span_indices)
    assert set(units.keys()) == set(STRUCTURAL_GROUP_NAMES)


def test_query_content_matches_query_span():
    query_span = (3, 9)
    units = build_structural_units(_fake_template_positions(), query_span, (12, 30), [12, 13, 14])
    assert units["query_content"] == list(range(3, 9))


def test_query_and_colon_disjoint_from_query_content():
    query_span = (3, 9)
    units = build_structural_units(_fake_template_positions(), query_span, (12, 30), [12, 13, 14])
    query_content = set(units["query_content"])
    assert query_content.isdisjoint(units["Query"])
    assert query_content.isdisjoint(units["Query_colon"])


def test_attack_tokens_exact_and_original_document_is_the_complement():
    doc_span = (12, 30)
    attack_span_indices = [12, 13, 14]
    units = build_structural_units(_fake_template_positions(), (3, 9), doc_span, attack_span_indices)
    assert units["attack_tokens"] == sorted(attack_span_indices)
    # original_document ∪ attack_tokens == doc_span, no overlap, no gaps.
    doc_full = set(range(*doc_span))
    assert set(units["original_document"]) | set(units["attack_tokens"]) == doc_full
    assert set(units["original_document"]).isdisjoint(units["attack_tokens"])


def test_random_position_attack_non_contiguous_indices():
    """"random"-position attacks can produce non-contiguous inserted-token
    index sets -- original_document must still be the exact complement."""
    doc_span = (12, 30)
    attack_span_indices = [15, 20, 25]  # non-contiguous
    units = build_structural_units(_fake_template_positions(), (3, 9), doc_span, attack_span_indices)
    assert units["attack_tokens"] == [15, 20, 25]
    assert 15 not in units["original_document"]
    assert 20 not in units["original_document"]
    assert 25 not in units["original_document"]
    assert len(units["original_document"]) == (doc_span[1] - doc_span[0]) - len(attack_span_indices)
