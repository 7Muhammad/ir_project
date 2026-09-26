"""
src/padded_control.py
=====================
Token-level construction of the three input variants used by the DecoderLens
rank experiment, plus the position-agnostic alignment that makes the padded
control and the attacked input the same length with aligned passage tokens.

Three input variants
--------------------
A. original:
       "Query: q Document: original_passage Relevant:"
   All real tokens, attention_mask = 1 everywhere.  Reference only.

B. padded_control:
       "Query: q Document: [PAD slots replacing injected tokens] passage Relevant:"
   The injected-token positions are replaced with tokenizer.pad_token_id and
   masked with attention_mask = 0.  Same length as the attacked input, with
   the real passage tokens at the SAME absolute positions.

C. attack:
       "Query: q Document: attacked_passage Relevant:"
   All real tokens, attention_mask = 1 everywhere.

Alignment
---------
We diff the original and attacked prompt token-id sequences to find which
positions in the attacked sequence are injected tokens.  Using
difflib.SequenceMatcher (with a fast prefix-walk + suffix-check shortcut) makes
this work for start, end, and random attacks, for repetition counts 1..5, and
for single-token, multi-token, and compound injection tokens.

(Algorithm adapted from 01_monot5_layer_patching/src/{alignment,model_utils}.py —
kept self-contained so this experiment is logically independent.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_monot5_input(query: str, passage: str) -> str:
    """Construct the monoT5 prompt: 'Query: {q} Document: {p} Relevant:'."""
    return f"Query: {query} Document: {passage} Relevant:"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    status: str                                   # "ok" | "failed"
    reason: str = ""
    inserted_positions: List[int] = field(default_factory=list)
    inserted_spans: List[Tuple[int, int]] = field(default_factory=list)
    n_inserted: int = 0


def align_attack(original_ids: List[int], attacked_ids: List[int]) -> AlignmentResult:
    """
    Find which positions in attacked_ids correspond to injected attack tokens.

    Strategy 1 (preferred): prefix-walk + suffix-check, which handles the common
    contiguous-block insertion deterministically and is immune to the shared
    token ambiguity that can mislead SequenceMatcher.

    Strategy 2 (fallback): difflib.SequenceMatcher.  The diff must consist of
    "equal" and "insert" opcodes only — any "delete"/"replace" means the passage
    tokens shifted (a SentencePiece boundary effect) and the example is failed.
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

    # --- Strategy 1: prefix-walk + suffix check --------------------------------
    n_prefix = 0
    for o_tok, a_tok in zip(original_ids, attacked_ids):
        if o_tok == a_tok:
            n_prefix += 1
        else:
            break

    original_suffix = original_ids[n_prefix:]
    attacked_suffix = attacked_ids[n_prefix + n_delta:]
    if original_suffix == attacked_suffix:
        inserted = list(range(n_prefix, n_prefix + n_delta))
        return AlignmentResult(
            status="ok",
            inserted_positions=inserted,
            inserted_spans=[(n_prefix, n_prefix + n_delta)],
            n_inserted=n_delta,
        )

    # --- Strategy 2: SequenceMatcher fallback ----------------------------------
    matcher = SequenceMatcher(a=original_ids, b=attacked_ids, autojunk=False)
    inserted_fb: List[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            inserted_fb.extend(range(j1, j2))
            continue
        return AlignmentResult(
            status="failed",
            reason=(
                f"Prefix-walk suffix check failed (SentencePiece boundary shift), "
                f"and SequenceMatcher fallback produced a '{tag}' opcode at "
                f"original[{i1}:{i2}] vs attacked[{j1}:{j2}]."
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


# ---------------------------------------------------------------------------
# Variant encodings
# ---------------------------------------------------------------------------

def build_variant_encodings(
    tokenizer,
    query: str,
    passage: str,
    attacked_passage: str,
    max_length: int,
    device: torch.device,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], AlignmentResult]:
    """
    Build the three variant encoder inputs at the token level.

    Returns
    -------
    original_enc, control_enc, attack_enc : dicts with "input_ids" (1, L) and
        "attention_mask" (1, L), or (None, None, None) on alignment failure.
    result : AlignmentResult — always returned; check result.status == "ok".

    On alignment failure, callers should record the failure (do NOT silently
    skip) and continue.
    """
    pad_id = tokenizer.pad_token_id

    original_text = build_monot5_input(query, passage)
    attacked_text = build_monot5_input(query, attacked_passage)

    original_ids: List[int] = tokenizer.encode(original_text, add_special_tokens=True)
    attacked_ids: List[int] = tokenizer.encode(attacked_text, add_special_tokens=True)

    result = align_attack(original_ids, attacked_ids)
    if result.status != "ok":
        return None, None, None, result

    inserted_set = set(result.inserted_positions)
    control_ids: List[int] = [
        pad_id if i in inserted_set else tok for i, tok in enumerate(attacked_ids)
    ]
    control_mask: List[int] = [
        0 if i in inserted_set else 1 for i in range(len(attacked_ids))
    ]
    attack_mask: List[int] = [1] * len(attacked_ids)
    original_mask: List[int] = [1] * len(original_ids)

    # Truncate to max_length.  Control and attack share length so the same slice
    # preserves alignment.
    control_ids = control_ids[:max_length]
    control_mask = control_mask[:max_length]
    attacked_ids_t = attacked_ids[:max_length]
    attack_mask = attack_mask[:max_length]
    original_ids_t = original_ids[:max_length]
    original_mask = original_mask[:max_length]

    def _enc(ids: List[int], mask: List[int]) -> Dict:
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
        }

    original_enc = _enc(original_ids_t, original_mask)
    control_enc = _enc(control_ids, control_mask)
    attack_enc = _enc(attacked_ids_t, attack_mask)
    return original_enc, control_enc, attack_enc, result


def encode_clean(
    tokenizer,
    query: str,
    passage: str,
    max_length: int,
    device: torch.device,
) -> Dict:
    """Tokenise a clean 'Query/Document/Relevant' prompt (all real tokens)."""
    text = build_monot5_input(query, passage)
    ids: List[int] = tokenizer.encode(text, add_special_tokens=True)[:max_length]
    mask = [1] * len(ids)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
    }
