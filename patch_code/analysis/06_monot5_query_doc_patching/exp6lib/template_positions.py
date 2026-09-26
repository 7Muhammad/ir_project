"""
exp6lib/template_positions.py
==============================
Experiment 6 extension — template-token sink/signal analysis, Step 0.

Locates the 7 named template-token positions that Part 2's query/document
positional patching (exp6lib/engine.py, exp6lib/spans.py) always freezes at
the control run's own value: the literal ``"Query"``, ``":"``, ``"Document"``,
``":"``, ``"Relevant"``, ``":"`` markers and the final ``</s>``. This module
answers the question the main report's Limitations section left open
(15-22% of the contribution-norm difference lives here, never causally
tested) by giving each of those 7 positions a token-index list that
`template_engine.py`'s single-position patch can target directly.

Why this is derived from query_span/doc_span rather than re-probed
--------------------------------------------------------------------
`exp6lib.spans.find_query_and_doc_spans` already computes, per example,
exactly the three "gap" regions that are excluded from every §4.4 patching
condition:

    leading region = [0, query_span[0])                       -- before the query
    mid region      = [query_span[1], doc_span[0])              -- between query and document
    tail region     = [doc_span[1], seq_len)                    -- after the document (Relevant: + EOS)

These are precisely the union of the 7 template positions (see docstring of
`get_template_positions` for the empirical token-count breakdown). No new
probe strings are needed -- re-using the query/doc spans keeps this in sync
with Experiment 6's existing alignment logic by construction instead of by
convention.

Why positions are grouped by DECODED CONTENT, not by hardcoded offsets
--------------------------------------------------------------------------
T5's SentencePiece tokenizer splits the literal template text differently
depending on whether it sits at the very start of the string or mid-string:
``"Query: "`` (string-initial) tokenizes as 3 pieces (``"▁"``, ``"Query"``,
``":"``) while ``" Document: "`` and ``" Relevant:"`` (mid-string) tokenize
as 2 and 3 pieces respectively, with `"Relevant:"` itself splitting into
``"▁Relevan"`` + ``"t"`` + ``":"``. Verified directly against
castorini/monot5-base-msmarco's tokenizer on 3 examples of different length
(see scripts/05_template_tokenizer_ground_truth.py): the leading region is
always 3 raw tokens, the mid region always 2, and the tail region always 4
(9 total -- matches the main report's "9 connective positions" and "Relevant,
t, :" token list exactly). Rather than hardcode those counts, this module
walks each region and classifies tokens by their DECODED text (the last
token of the leading/mid regions is always ":"; the last token of the tail
region is always the tokenizer's EOS id), so a future tokenizer/checkpoint
change would fail loudly (AssertionError) instead of silently mis-grouping.

The one deliberate merge: the leading region's string-initial "▁" token has
no meaning independent of the word "Query" that follows it (it is a
SentencePiece artefact of being at position 0, not a fourth region), so it
is folded into the "Query" unit. This keeps the position count at exactly
7, matching the experiment prompt's named list, while the raw excluded
token count is still 9 (2 for "Query" + 1 ":" + 1 "Document" + 1 ":" + 2 for
"Relevant" + 1 ":" + 1 "</s>").
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from transformers import T5Tokenizer

TEMPLATE_POSITION_NAMES: List[str] = [
    "Query",
    ":_after_query",
    "Document",
    ":_after_document",
    "Relevant",
    ":_after_relevant",
    "</s>",
]


def _decode_one(tokenizer: T5Tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id]).strip()


def _split_leading_or_mid(
    tokenizer: T5Tokenizer, input_ids: List[int], region: List[int], word_name: str, colon_name: str
) -> Dict[str, List[int]]:
    if len(region) < 2:
        raise AssertionError(
            f"Template region for {word_name!r} has only {len(region)} token(s) "
            f"({region}) -- expected at least 2 (word + colon). Tokenizer output "
            f"may have changed; re-run scripts/05_template_tokenizer_ground_truth.py."
        )
    colon_idx = region[-1]
    colon_tok = input_ids[colon_idx]
    decoded = _decode_one(tokenizer, colon_tok)
    if decoded != ":":
        raise AssertionError(
            f"Expected the last token of the {word_name!r} region to decode to ':', "
            f"got {decoded!r} (id={colon_tok}, index={colon_idx}). Tokenizer output may have changed."
        )
    return {word_name: list(region[:-1]), colon_name: [colon_idx]}


def _split_tail(tokenizer: T5Tokenizer, input_ids: List[int], region: List[int]) -> Dict[str, List[int]]:
    if len(region) < 3:
        raise AssertionError(
            f"Tail (Relevant/:/</s>) region has only {len(region)} token(s) "
            f"({region}) -- expected at least 3 (Relevant + ':' + EOS)."
        )
    eos_idx = region[-1]
    eos_tok = input_ids[eos_idx]
    if eos_tok != tokenizer.eos_token_id:
        raise AssertionError(
            f"Expected the last token of the tail region to be EOS "
            f"(id={tokenizer.eos_token_id}), got id={eos_tok} (index={eos_idx}) "
            f"({_decode_one(tokenizer, eos_tok)!r})."
        )
    colon_idx = region[-2]
    colon_tok = input_ids[colon_idx]
    decoded = _decode_one(tokenizer, colon_tok)
    if decoded != ":":
        raise AssertionError(
            f"Expected the second-to-last token of the tail region to decode to ':', "
            f"got {decoded!r} (id={colon_tok}, index={colon_idx})."
        )
    return {
        "Relevant": list(region[:-2]),
        ":_after_relevant": [colon_idx],
        "</s>": [eos_idx],
    }


def get_template_positions(
    tokenizer: T5Tokenizer,
    input_ids: List[int],
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
) -> Dict[str, List[int]]:
    """
    Split the 3 gap regions (before query, between query and document, after
    document) surrounding `query_span`/`doc_span` (both exclusive-end token
    ranges, shared control/attack Type-B/C indexing -- the same spans
    `exp6lib.engine.build_position_mask` already uses) into the 7 named
    template-token position groups.

    Parameters
    ----------
    input_ids : the actual token-id sequence (e.g. attack_enc["input_ids"][0]
        as a plain list) this query_span/doc_span were computed against --
        needed to decode and sanity-check each boundary token's identity.

    Returns
    -------
    dict[str, list[int]], keys == TEMPLATE_POSITION_NAMES, values sorted
    ascending token-INDEX lists (never empty; "Query"/"Relevant" may have
    more than one index, all others have exactly one).
    """
    seq_len = len(input_ids)
    leading_region = list(range(0, query_span[0]))
    mid_region = list(range(query_span[1], doc_span[0]))
    tail_region = list(range(doc_span[1], seq_len))

    positions: Dict[str, List[int]] = {}
    positions.update(_split_leading_or_mid(tokenizer, input_ids, leading_region, "Query", ":_after_query"))
    positions.update(_split_leading_or_mid(tokenizer, input_ids, mid_region, "Document", ":_after_document"))
    positions.update(_split_tail(tokenizer, input_ids, tail_region))

    assert set(positions.keys()) == set(TEMPLATE_POSITION_NAMES)
    return {name: positions[name] for name in TEMPLATE_POSITION_NAMES}
