"""
exp10lib/tagging.py
====================
Per-position tagging: {injected_attack_token, other_document, query,
connective_template} for every token position in the shared Type B/C
(padded-control / attacked) sequence layout.

Reuses two already-verified, alignment-checked utilities rather than
re-deriving span boundaries:
  - Experiment 6's ``find_query_and_doc_spans`` (exp6lib/spans.py) for the
    query-span and document-span boundaries (probe-substring, token-level
    prefix check against "Query: {query} Document: {passage} Relevant:").
  - Experiment 1's ``align_attack`` (src/alignment.py) for the exact
    injected-attack-token sub-span within the document span.

Experiment 6 computes both pieces internally (it calls align_attack via
Experiment 1's padded-control builder) but only ever consumes the SCALAR
count ``n_inserted`` to widen the document span — it never surfaces
``inserted_spans`` (the precise sub-range) to callers. That is the one gap
this module fills; everything else is a direct reuse (see DECISIONS.md).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from transformers import T5Tokenizer

from src.alignment import AlignmentResult
from exp6lib.spans import QueryDocSpans, find_query_and_doc_spans

TAG_INJECTED = "injected_attack_token"
TAG_OTHER_DOC = "other_document"
TAG_QUERY = "query"
TAG_CONNECTIVE = "connective_template"

ALL_TAGS = (TAG_INJECTED, TAG_OTHER_DOC, TAG_QUERY, TAG_CONNECTIVE)


def _tag_positions(
    seq_len: int,
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
    inserted_spans: List[Tuple[int, int]],
) -> List[str]:
    """
    Precedence: connective_template (default) < query / other_document <
    injected_attack_token. Query and document spans should never overlap
    (find_query_and_doc_spans already asserts strict ordering); injected
    spans are applied last so they take priority over the document-span
    fill within their own sub-range.
    """
    tags = [TAG_CONNECTIVE] * seq_len

    q_start, q_end = query_span
    for i in range(max(q_start, 0), min(q_end, seq_len)):
        tags[i] = TAG_QUERY

    d_start, d_end = doc_span
    for i in range(max(d_start, 0), min(d_end, seq_len)):
        tags[i] = TAG_OTHER_DOC

    for s, e in inserted_spans:
        for i in range(max(s, 0), min(e, seq_len)):
            tags[i] = TAG_INJECTED

    return tags


def tag_example_positions(
    tokenizer: T5Tokenizer,
    query: str,
    passage: str,
    align_result: AlignmentResult,
    seq_len: int,
) -> Tuple[Optional[List[str]], QueryDocSpans]:
    """
    Compute the query/document spans (Experiment 6's alignment-checked
    probes) and tag every position 0..seq_len-1.

    Returns (tags, spans). ``tags`` is None if span detection failed
    (``spans.status == "failed"``); callers should skip the example and
    count it, matching Experiment 1/6's alignment-failure convention.
    """
    spans = find_query_and_doc_spans(
        tokenizer=tokenizer, query=query, passage=passage,
        n_attack_tokens=align_result.n_inserted,
    )
    if spans.status != "ok":
        return None, spans

    tags = _tag_positions(
        seq_len=seq_len,
        query_span=spans.query_span,
        doc_span=spans.doc_span_control_attack,
        inserted_spans=align_result.inserted_spans,
    )
    return tags, spans
