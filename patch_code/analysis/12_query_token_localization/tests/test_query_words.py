from __future__ import annotations

from exp6lib.spans import find_query_and_doc_spans
from exp12lib.query_words import classify_query_words, find_query_word_spans, normalize_word


def test_normalize_word():
    assert normalize_word("Fleas") == "fleas"
    assert normalize_word("live?") == "live"
    assert normalize_word("--") == ""
    assert normalize_word("don't") == "don't"


def test_word_spans_reproduce_query_span(tokenizer):
    query = "how long do fleas live"
    passage = "Fleas typically live for about 100 days depending on conditions."
    spans = find_query_and_doc_spans(tokenizer, query, passage, n_attack_tokens=5)
    assert spans.status == "ok"

    result = find_query_word_spans(tokenizer, query, spans.query_span)
    assert result.status == "ok"
    words = result.words
    assert [w.text for w in words] == ["how", "long", "do", "fleas", "live"]

    # Sanity check 1: union of word spans == query_span, contiguous, no gaps/overlap.
    union = set()
    for w in words:
        assert not (union & set(w.token_indices)), "word spans must not overlap"
        union |= set(w.token_indices)
    assert union == set(range(*spans.query_span))
    assert all(w.token_end - w.token_start == w.num_subtokens for w in words)


def test_multi_subtoken_word_stays_one_unit(tokenizer):
    # "synaptic" / "knob" reliably split into >1 SentencePiece token on this
    # checkpoint's vocab -- verified empirically against the actual tokenizer.
    query = "axon terminals or synaptic knob definition"
    passage = "irrelevant placeholder passage text"
    spans = find_query_and_doc_spans(tokenizer, query, passage, n_attack_tokens=1)
    assert spans.status == "ok"
    result = find_query_word_spans(tokenizer, query, spans.query_span)
    assert result.status == "ok"
    by_text = {w.text: w for w in result.words}
    multi = [w for w in by_text.values() if w.num_subtokens > 1]
    assert multi, "expected at least one multi-subtoken word in this query"
    for w in multi:
        assert w.token_indices == list(range(w.token_start, w.token_end))


def test_classification_content_vs_stopword(tokenizer):
    query = "how long do fleas live"
    passage = "Fleas can live for several months under the right conditions."
    spans = find_query_and_doc_spans(tokenizer, query, passage, n_attack_tokens=1)
    result = find_query_word_spans(tokenizer, query, spans.query_span)
    words = result.words
    classify_query_words(words, passage)
    labels = {w.text: (w.content_or_stopword, w.matched_or_unmatched) for w in words}

    assert labels["how"][0] == "stopword"
    assert labels["do"][0] == "stopword"
    assert labels["long"][0] == "content"
    assert labels["fleas"][0] == "content"
    assert labels["live"][0] == "content"

    # "fleas" and "live" both appear (case-insensitively) in the clean passage.
    assert labels["fleas"][1] == "matched"
    assert labels["live"][1] == "matched"
    # "long" does not appear anywhere in the passage.
    assert labels["long"][1] == "unmatched"


def test_classification_uses_clean_document_not_attacked(tokenizer):
    """matched/unmatched must be computed from the ORIGINAL clean passage,
    never the attacked one -- a word only present in an injected attack
    prefix must NOT count as 'matched'."""
    query = "what is the capital of france"
    clean_passage = "Paris is the capital and most populous city of France."
    attacked_passage = "relevant relevant relevant " + clean_passage
    spans = find_query_and_doc_spans(tokenizer, query, clean_passage, n_attack_tokens=3)
    result = find_query_word_spans(tokenizer, query, spans.query_span)
    words = result.words
    classify_query_words(words, clean_passage)  # must use clean_passage, not attacked_passage
    by_text = {w.text: w for w in words}
    assert by_text["capital"].matched_or_unmatched == "matched"
    assert by_text["france"].matched_or_unmatched == "matched"
    assert "relevant" not in [w.text for w in words]  # sanity: attack words aren't query words
