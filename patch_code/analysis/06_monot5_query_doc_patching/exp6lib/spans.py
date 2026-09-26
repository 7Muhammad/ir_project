"""
exp6lib/spans.py
=================
Locate query-span and document-span token-index ranges in the monoT5
encoder input, alignment-checked the same way Experiment 1 checks attack
insertion positions (src/alignment.py): build a probe substring, verify it
is an EXACT token-level prefix of the full prompt, and treat any mismatch as
an alignment failure to be skipped and counted, never silently guessed.

Why the query/document boundary needs its own probes (can't reuse
Experiment 1's n_prefix directly)
-----------------------------------------------------------------
Experiment 1's alignment (src/alignment.py) finds where the ATTACK TOKENS
were inserted relative to the original prompt. For "start" attacks that
insertion point happens to coincide with the start of the document field,
but for "end" and "random" attacks the attack tokens are inserted later
inside the passage — so Experiment 1's insertion boundary is NOT the same
as the "Document: " literal boundary in general. The query/document
boundary is independent of where within the document field the attack
lands, so it is computed directly from the CLEAN prompt structure via
three probes:

    "Query: "                              -> n_q_start
    "Query: {query}"                       -> n_q_end   (query span end)
    "Query: {query} Document: "            -> doc_start (document span start)
    "Query: {query} Document: {passage}"   -> doc_end   (document span end,
                                               in the ORIGINAL/Type-A prompt)

Each probe is tokenised with add_special_tokens=False (no EOS) and checked
to be an exact prefix of the full original prompt's token ids
(add_special_tokens=True only appends EOS at the very end, so it does not
affect prefix positions).

Document span in the control/attack (Type B/C) encodings
----------------------------------------------------------
The attack is always injected strictly INSIDE the document field (between
doc_start and doc_end of the original prompt), regardless of position
(start/end/random), so:

    query_span  (same in Type A, B, C) = (n_q_start, n_q_end)
    doc_span    (Type A / original)    = (doc_start, doc_end)
    doc_span    (Type B / Type C)      = (doc_start, doc_end + n_attack_tokens)

This is why "document-only" patching (which uses the Type B/C span) is
expected to trivially reproduce the whole attack in Experiment 6 Part 2:
the injected keyword-stuffing tokens are, by the template's own structure,
part of the document field, not a fourth region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import T5Tokenizer

from src.model_utils import build_monot5_input


@dataclass
class QueryDocSpans:
    status: str                      # "ok" | "failed"
    reason: str = ""
    query_span: Tuple[int, int] = (0, 0)          # same in Type A / B / C
    doc_span_original: Tuple[int, int] = (0, 0)    # Type A (clean) indexing
    doc_span_control_attack: Tuple[int, int] = (0, 0)  # Type B / C indexing


_Q_START_TEXT = "Query: "


def _is_prefix(probe_ids: list, full_ids: list) -> bool:
    return len(probe_ids) <= len(full_ids) and full_ids[: len(probe_ids)] == probe_ids


def find_query_and_doc_spans(
    tokenizer: T5Tokenizer,
    query: str,
    passage: str,
    n_attack_tokens: int,
) -> QueryDocSpans:
    """
    Find query-span and document-span token ranges, alignment-checked.

    Parameters
    ----------
    tokenizer : T5Tokenizer
    query, passage : str — the ORIGINAL clean query/passage text (not the
        attacked passage; the attacked passage is not needed here because
        the query/document boundary never depends on the attack).
    n_attack_tokens : int — N, the number of tokens the attack inserts
        (from the existing alignment result for this example), used only
        to compute doc_span_control_attack.

    Returns
    -------
    QueryDocSpans. On failure (status="failed"), all spans are (0, 0) and
    `reason` explains which probe did not match; callers should skip the
    example, matching Experiment 1's alignment-failure convention.
    """
    original_text = build_monot5_input(query, passage)
    original_ids = tokenizer.encode(original_text, add_special_tokens=True)

    probe_q_start = tokenizer.encode(_Q_START_TEXT, add_special_tokens=False)
    if not _is_prefix(probe_q_start, original_ids):
        return QueryDocSpans(status="failed", reason="'Query: ' is not a token-level prefix of the prompt.")
    n_q_start = len(probe_q_start)

    probe_q_end = tokenizer.encode(f"Query: {query}", add_special_tokens=False)
    if not _is_prefix(probe_q_end, original_ids):
        return QueryDocSpans(status="failed", reason="'Query: {query}' is not a token-level prefix of the prompt.")
    n_q_end = len(probe_q_end)

    probe_doc_start = tokenizer.encode(f"Query: {query} Document: ", add_special_tokens=False)
    if not _is_prefix(probe_doc_start, original_ids):
        return QueryDocSpans(status="failed", reason="'Query: {query} Document: ' is not a token-level prefix of the prompt.")
    doc_start = len(probe_doc_start)

    probe_doc_end = tokenizer.encode(f"Query: {query} Document: {passage}", add_special_tokens=False)
    if not _is_prefix(probe_doc_end, original_ids):
        return QueryDocSpans(status="failed", reason="'Query: {query} Document: {passage}' is not a token-level prefix of the prompt.")
    doc_end = len(probe_doc_end)

    if not (n_q_start < n_q_end <= doc_start < doc_end):
        return QueryDocSpans(
            status="failed",
            reason=f"Degenerate span ordering: n_q_start={n_q_start}, n_q_end={n_q_end}, "
                   f"doc_start={doc_start}, doc_end={doc_end}.",
        )

    return QueryDocSpans(
        status="ok",
        query_span=(n_q_start, n_q_end),
        doc_span_original=(doc_start, doc_end),
        doc_span_control_attack=(doc_start, doc_end + n_attack_tokens),
    )
