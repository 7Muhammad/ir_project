"""
src/alignment.py
================
Token-level alignment between original and attacked prompts.

Used by the *general* padded-control construction so that start, end, and
random injection attacks all map onto the same activation-patching geometry:
control and attacked sequences have the same length, with pad slots
(attention_mask=0) at exactly the positions where the attack tokens were
inserted.

We diff the two token-id sequences with difflib.SequenceMatcher.  For a
valid injection attack the diff must consist of "equal" and "insert" opcodes
only — any "delete" or "replace" means the original passage tokens were
re-tokenised differently in the presence of the attack (a SentencePiece
boundary effect) or that SequenceMatcher anchored on the wrong occurrence
of a token shared between attack and passage.  Either way it is unsafe to
patch and we mark the example as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Tuple


@dataclass
class AlignmentResult:
    status: str                               # "ok" | "failed"
    reason: str = ""                          # human-readable explanation if failed
    inserted_positions: List[int] = field(default_factory=list)
                                              # indices in attacked_ids that are insertions
    inserted_spans: List[Tuple[int, int]] = field(default_factory=list)
                                              # contiguous (start, end) ranges in attacked_ids
    n_inserted: int = 0


def align_attack(original_ids: List[int], attacked_ids: List[int]) -> AlignmentResult:
    """
    Find which positions in attacked_ids correspond to injected attack tokens.

    Strategy
    --------
    ECIR-24 injection attacks always insert tokens as a **single contiguous
    block** at one point in the sequence (start, end, or a mid-passage
    position).  We exploit this by first trying a fast prefix-walk + suffix
    check:

    1. Walk both sequences from the left until they first diverge — this is
       the start of the insertion block (``n_prefix``).
    2. The insertion block has length N = len(attacked) - len(original).
    3. Verify that the tokens *after* the insertion block in the attacked
       sequence match the tokens after the divergence point in the original
       sequence.  If they match, the attack inserted a contiguous block of N
       tokens at position n_prefix — we are done.

    This approach is deterministic and immune to the "shared token" ambiguity
    that can mislead SequenceMatcher (e.g. ":" appearing in both the attack
    token "relevant:" and in "Document:" or "Relevant:").

    If the suffix check fails (rare: SentencePiece re-tokenised the passage
    boundary differently due to the presence of the attack), we fall back to
    SequenceMatcher.  In the fallback, the diff must consist only of "equal"
    and "insert" opcodes — any "delete" or "replace" is a failure.

    Parameters
    ----------
    original_ids : token-id list for the clean prompt (build_monot5_input + encode)
    attacked_ids : token-id list for the attacked prompt

    Returns
    -------
    AlignmentResult
        On success:
          - ``inserted_positions``  — every index in attacked_ids that is an
            insertion relative to original_ids.
          - ``inserted_spans``      — contiguous runs of those indices as
            (start, end) pairs (end is exclusive, Python-slice style).
          - ``n_inserted``          — total count of inserted tokens.
        On failure:
          - ``status = "failed"`` with a human-readable ``reason``.

    Failure modes
    -------------
    * attacked length <= original length          — nothing was inserted.
    * Suffix check and SequenceMatcher both fail  — passage tokens shifted.
    """
    n_delta = len(attacked_ids) - len(original_ids)
    if n_delta <= 0:
        return AlignmentResult(
            status="failed",
            reason=(
                f"attacked length ({len(attacked_ids)}) <= original length "
                f"({len(original_ids)}); no tokens were inserted."
            ),
        )

    # --- Strategy 1: prefix-walk + suffix check (preferred) ----------------
    # Walk both sequences until the first divergence.
    n_prefix = 0
    for o_tok, a_tok in zip(original_ids, attacked_ids):
        if o_tok == a_tok:
            n_prefix += 1
        else:
            break

    # Check that the suffix after the insertion block aligns.
    original_suffix = original_ids[n_prefix:]
    attacked_suffix = attacked_ids[n_prefix + n_delta:]
    if original_suffix == attacked_suffix:
        # Contiguous insertion at n_prefix.
        inserted = list(range(n_prefix, n_prefix + n_delta))
        spans = [(n_prefix, n_prefix + n_delta)]
        return AlignmentResult(
            status="ok",
            inserted_positions=inserted,
            inserted_spans=spans,
            n_inserted=n_delta,
        )

    # --- Strategy 2: SequenceMatcher fallback --------------------------------
    # Used when a SentencePiece boundary shift means the simple suffix check
    # fails (the attacked passage re-tokenises the boundary slightly differently).
    matcher = SequenceMatcher(a=original_ids, b=attacked_ids, autojunk=False)
    inserted_fb: List[int] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            inserted_fb.extend(range(j1, j2))
            continue
        # "delete" or "replace" — cannot align safely.
        return AlignmentResult(
            status="failed",
            reason=(
                f"Prefix-walk suffix check failed (SentencePiece boundary shift), "
                f"and SequenceMatcher fallback produced a '{tag}' opcode at "
                f"original[{i1}:{i2}] vs attacked[{j1}:{j2}]. "
                f"The passage tokens are not preserved at the same positions. "
                f"This example cannot be safely patched."
            ),
        )

    if len(inserted_fb) != n_delta:
        return AlignmentResult(
            status="failed",
            reason=(
                f"SequenceMatcher fallback: inserted count {len(inserted_fb)} "
                f"!= length delta {n_delta}."
            ),
        )

    # Collapse contiguous runs into spans.
    spans_fb: List[Tuple[int, int]] = []
    if inserted_fb:
        start = prev = inserted_fb[0]
        for idx in inserted_fb[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                spans_fb.append((start, prev + 1))
                start = prev = idx
        spans_fb.append((start, prev + 1))

    return AlignmentResult(
        status="ok",
        inserted_positions=inserted_fb,
        inserted_spans=spans_fb,
        n_inserted=len(inserted_fb),
    )
