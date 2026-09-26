"""
exp12lib/structural_masks.py
==============================
Structural input-position groups for Experiment 12's causal controls and
Part A's attention analysis, built ENTIRELY from Experiment 6's existing,
alignment-checked span-finding (never re-probed / hardcoded here):

  exp6lib.spans.find_query_and_doc_spans     -> query_span, doc_span (Type B/C)
  exp6lib.template_positions.get_template_positions
        -> "Query", ":_after_query", "Document", ":_after_document",
           "Relevant", ":_after_relevant", "</s>"

The experiment prompt asks for 9 named structural-control groups:
    Query, Query-colon, query content, Document, Document-colon,
    attack tokens (jointly), original document tokens (jointly),
    Relevant, Relevant-colon
plus (for the 10-position input-structure listing) the final </s>, which is
retained in the template-position dict but not one of the 9 causal controls
(patching past EOS has no defined semantics here).

"attack tokens" and "original document tokens" split doc_span (Type B/C)
using `ExampleInputs.attack_span_indices` (exact inserted-token indices from
the existing alignment machinery, `src.alignment.align_attack` via
`build_padded_control_and_attack_encodings_general`) -- never re-derived.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

STRUCTURAL_GROUP_NAMES: List[str] = [
    "Query",
    "Query_colon",
    "query_content",
    "Document",
    "Document_colon",
    "attack_tokens",
    "original_document",
    "Relevant",
    "Relevant_colon",
]


def build_structural_units(
    template_positions: Dict[str, List[int]],
    query_span: Tuple[int, int],
    doc_span: Tuple[int, int],
    attack_span_indices: List[int],
) -> Dict[str, List[int]]:
    """
    Build the 9 structural-control index groups (see STRUCTURAL_GROUP_NAMES),
    all in the shared Type-B/C (padded-control / attacked) indexing.

    Parameters
    ----------
    template_positions : output of exp6lib.template_positions.get_template_positions
        (already computed against the SAME encoding these indices index into).
    query_span : (start, end) exclusive -- exp6lib.spans query_span.
    doc_span : (start, end) exclusive -- exp6lib.spans doc_span_control_attack.
    attack_span_indices : exact inserted attack-token indices (may be
        non-contiguous for "random" position attacks) -- ExampleInputs.attack_span_indices.
    """
    attack_set = set(attack_span_indices)
    original_document = [i for i in range(doc_span[0], doc_span[1]) if i not in attack_set]

    units = {
        "Query": list(template_positions["Query"]),
        "Query_colon": list(template_positions[":_after_query"]),
        "query_content": list(range(query_span[0], query_span[1])),
        "Document": list(template_positions["Document"]),
        "Document_colon": list(template_positions[":_after_document"]),
        "attack_tokens": sorted(attack_span_indices),
        "original_document": original_document,
        "Relevant": list(template_positions["Relevant"]),
        "Relevant_colon": list(template_positions[":_after_relevant"]),
    }
    assert set(units.keys()) == set(STRUCTURAL_GROUP_NAMES)
    return units
