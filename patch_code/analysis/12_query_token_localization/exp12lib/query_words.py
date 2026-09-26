"""
exp12lib/query_words.py
========================
Human query-word span finding and classification for Experiment 12.

Word-span construction
-----------------------
A "query word" is a whitespace-delimited unit of the ORIGINAL query string
(``query.split()``). Multi-subtoken SentencePiece words are kept as ONE
intervention unit (their complete, contiguous token span), never split.

Spans are located by the exact same alignment-checked prefix-probing style
`exp6lib.spans.find_query_and_doc_spans` already uses (never hardcoded
offsets): each successive probe string ``"Query: " + " ".join(words[:i+1])``
must be an exact token-level prefix of ``"Query: " + query`` — if a probe
mismatches, alignment has failed and the example is skipped, exactly like
Experiment 6's alignment-failure convention.

Because `exp6lib.spans.find_query_and_doc_spans` already validates that
``"Query: {query}"`` is itself a token-level prefix of the full prompt, and
returns ``query_span = (n_q_start, n_q_end)`` in the SAME indexing for the
Type A (clean), Type B (padded control), and Type C (attacked) encodings
(the query always precedes the document field, which is the only field the
attack touches), the token-index spans computed here against the
``"Query: {query}"`` substring alone are valid absolute indices in all three
encodings without any further adjustment.

Classification
---------------
Each query word gets two independent binary labels:
  content vs. stopword   -- via exp12lib.stopwords (fixed NLTK list)
  matched vs. unmatched  -- whether the normalized word string occurs in the
                             ORIGINAL CLEAN document (never the attacked one)

Lexical normalization (documented, minimal, no semantic matching)
--------------------------------------------------------------------
    normalize_word(w) = w.strip(string.punctuation).lower()

Applied identically to query words and to the clean document's
whitespace-split tokens before the matched/unmatched membership test.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers import T5Tokenizer

from exp12lib.stopwords import is_stopword

_Q_PREFIX = "Query: "


def normalize_word(word: str) -> str:
    """Lowercase + strip leading/trailing punctuation. No semantic normalization."""
    return word.strip(string.punctuation).lower()


def _is_prefix(probe_ids: List[int], full_ids: List[int]) -> bool:
    return len(probe_ids) <= len(full_ids) and full_ids[: len(probe_ids)] == probe_ids


@dataclass
class QueryWordSpan:
    index: int
    text: str                       # raw (unnormalized) word as it appears in the query
    token_start: int                # inclusive, absolute encoder index (Type A/B/C-shared)
    token_end: int                  # exclusive
    num_subtokens: int
    normalized: str = ""
    content_or_stopword: str = ""   # "content" | "stopword"
    matched_or_unmatched: str = ""  # "matched" | "unmatched"

    @property
    def token_indices(self) -> List[int]:
        return list(range(self.token_start, self.token_end))

    @property
    def word_group(self) -> str:
        return f"{self.content_or_stopword}_{self.matched_or_unmatched}"


@dataclass
class QueryWordResult:
    status: str            # "ok" | "failed"
    reason: str = ""
    words: Optional[List[QueryWordSpan]] = None


def find_query_word_spans(
    tokenizer: T5Tokenizer,
    query: str,
    query_span: Tuple[int, int],
) -> QueryWordResult:
    """
    Locate one contiguous token span per whitespace-delimited query word.

    Parameters
    ----------
    query : the ORIGINAL query text (not the prompt, not the document).
    query_span : (n_q_start, n_q_end) as returned by
        `exp6lib.spans.find_query_and_doc_spans` for this same query — the
        SAME indexing used in the Type A/B/C encoder input.

    Returns
    -------
    QueryWordResult. On success, `words` covers query_span exactly (union of
    all word spans == query_span, verified below — this is Sanity Check 1
    from the experiment prompt).
    """
    n_q_start, n_q_end = query_span
    full_query_probe = tokenizer.encode(f"{_Q_PREFIX}{query}", add_special_tokens=False)
    if len(full_query_probe) != n_q_end:
        return QueryWordResult(
            status="failed",
            reason=f"'Query: {{query}}' probe length {len(full_query_probe)} != query_span end {n_q_end}.",
        )
    prefix_probe = tokenizer.encode(_Q_PREFIX, add_special_tokens=False)
    if not _is_prefix(prefix_probe, full_query_probe) or len(prefix_probe) != n_q_start:
        return QueryWordResult(
            status="failed",
            reason="'Query: ' is not a token-level prefix of 'Query: {query}' at the expected offset.",
        )

    words = query.split()
    if not words:
        return QueryWordResult(status="failed", reason="Query has no whitespace-delimited words.")

    spans: List[QueryWordSpan] = []
    prev_end = n_q_start
    cumulative = ""
    for i, w in enumerate(words):
        cumulative = w if i == 0 else f"{cumulative} {w}"
        probe = tokenizer.encode(f"{_Q_PREFIX}{cumulative}", add_special_tokens=False)
        if not _is_prefix(probe, full_query_probe):
            return QueryWordResult(
                status="failed",
                reason=f"Word {i} ({w!r}) cumulative probe is not a token-level prefix of the query.",
            )
        word_end = len(probe)
        if word_end <= prev_end:
            return QueryWordResult(
                status="failed",
                reason=f"Word {i} ({w!r}) produced a degenerate/empty token span.",
            )
        spans.append(QueryWordSpan(
            index=i, text=w, token_start=prev_end, token_end=word_end,
            num_subtokens=word_end - prev_end,
        ))
        prev_end = word_end

    if prev_end != n_q_end:
        return QueryWordResult(
            status="failed",
            reason=f"Union of word spans ends at {prev_end}, expected query_span end {n_q_end}.",
        )

    return QueryWordResult(status="ok", words=spans)


def classify_query_words(words: List[QueryWordSpan], clean_document: str) -> None:
    """
    Assign content_or_stopword and matched_or_unmatched in place.

    `clean_document` must be the ORIGINAL CLEAN passage (pre-attack), never
    the attacked_passage — matched/unmatched is defined against the clean
    document specifically.
    """
    doc_tokens = {normalize_word(t) for t in clean_document.split()}
    for w in words:
        norm = normalize_word(w.text)
        w.normalized = norm
        w.content_or_stopword = "stopword" if is_stopword(norm) else "content"
        w.matched_or_unmatched = "matched" if norm in doc_tokens else "unmatched"
