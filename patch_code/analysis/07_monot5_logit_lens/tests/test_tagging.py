"""Tagging/bucket-composition correctness."""

from __future__ import annotations

from exp7lib.tagging import (
    STOPWORDS,
    bucket_composition,
    build_tagging_vocab,
    normalize_sp_token,
    simple_stem,
    tag_token,
)


def test_normalize_sp_token_strips_marker_and_lowercases():
    assert normalize_sp_token("▁Relevant") == "relevant"
    assert normalize_sp_token("s") == "s"


def test_simple_stem_handles_common_inflections():
    assert simple_stem("documents") == "document"
    assert simple_stem("running") == "runn"
    assert simple_stem("relevance") in ("relevan", "relevance")  # not linguistically perfect, just consistent


def test_attack_checked_before_query_and_document():
    """A word that is BOTH the attack token and present in the query/document
    text must be tagged 'attack', not 'query'/'document' — the whole point
    of Analysis 4 (see DECISIONS.md)."""
    vocab = build_tagging_vocab(query="is this important", passage="an important document", attack_token="important")
    assert tag_token("▁important", vocab) == "attack"


def test_query_before_document_when_not_attack():
    vocab = build_tagging_vocab(query="rsa definition key", passage="rsa stands for something else", attack_token="bar")
    assert tag_token("▁rsa", vocab) == "query"
    assert tag_token("▁something", vocab) == "document"


def test_stopword_and_other():
    vocab = build_tagging_vocab(query="rsa key", passage="a document", attack_token="bar")
    assert tag_token("▁the", vocab) == "stopword"
    assert tag_token("▁zebra", vocab) == "other"


def test_stem_fallback_catches_plural_variant():
    vocab = build_tagging_vocab(query="definition", passage="rsa document about keys", attack_token="bar")
    # "keys" isn't an exact match for "key"... wait passage has "keys" itself; test a genuine near-miss instead
    vocab2 = build_tagging_vocab(query="one key", passage="something", attack_token="bar")
    assert tag_token("▁keys", vocab2) == "query"  # "keys" stems to "key", matching query's "key"


def test_bucket_composition_sums_to_100_and_empty_case():
    vocab = build_tagging_vocab(query="rsa key", passage="a document", attack_token="bar")
    comp = bucket_composition(["▁rsa", "▁the", "▁zebra", "▁bar"], vocab)
    assert abs(sum(comp.values()) - 100.0) < 1e-9
    assert comp["attack"] == 25.0
    assert comp["query"] == 25.0
    assert comp["stopword"] == 25.0
    assert comp["other"] == 25.0

    empty = bucket_composition([], vocab)
    assert all(v == 0.0 for v in empty.values())


def test_stopwords_list_is_reasonable_size():
    assert len(STOPWORDS) > 100
    assert "the" in STOPWORDS and "and" in STOPWORDS
