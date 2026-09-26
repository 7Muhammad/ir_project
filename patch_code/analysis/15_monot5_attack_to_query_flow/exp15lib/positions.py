"""
exp15lib/positions.py
======================
Per-example SOURCE and TARGET token positions, resolved only with existing
project machinery (no string search over the rendered prompt):

  A  (attack source set)  = AlignmentResult.inserted_positions from Exp 01's
                            build_padded_control_and_attack_encodings_general
                            (token-level diff of clean vs attacked prompt).
                            ALL injected positions of the example, i.e. every
                            repetition and every scattered span of a random
                            attack, form ONE source set.
  Q  (query targets)      = range(*query_span) from Exp 06's
                            find_query_and_doc_spans: exactly the model-token
                            positions of the query TEXT. "▁ Query :" before,
                            "▁Document :" after, "▁Relevan t :" and </s> at
                            the end are template positions and are excluded
                            by construction.

Positions are raw T5/SentencePiece token positions — no word reconstruction
or subword merging (DECISIONS.md item 11). Token strings are stored only for
interpretability.

Every invariant the intervention relies on is checked here and violations
raise (fail loudly) rather than being skipped silently.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

import exp15lib  # noqa: F401
from exp6lib.spans import find_query_and_doc_spans
from src.model_utils import (
    build_monot5_input,
    build_padded_control_and_attack_encodings_general,
)


class PositionError(RuntimeError):
    pass


def build_encodings_and_positions(
    tokenizer,
    query: str,
    passage: str,
    attacked_passage: str,
    max_length: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict, Dict, Dict]:
    """
    Returns (control_enc, attack_enc, info) where info holds:
      attack_source_positions, attack_spans, query_target_positions,
      query_token_ids, query_token_strings, seq_len, doc_span.
    Raises PositionError on any alignment/span inconsistency.
    """
    control_enc, attack_enc, align = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=query, passage=passage,
        attacked_passage=attacked_passage, max_length=max_length, device=device,
    )
    if align.status != "ok":
        raise PositionError(f"attack alignment failed: {align.reason}")

    full_len = len(tokenizer.encode(build_monot5_input(query, attacked_passage), add_special_tokens=True))
    if full_len > max_length:
        raise PositionError(
            f"attacked prompt has {full_len} tokens > max_length={max_length}; truncation would "
            "silently drop positions (never observed in the Exp 01 pools)."
        )

    spans = find_query_and_doc_spans(tokenizer, query, passage, align.n_inserted)
    if spans.status != "ok":
        raise PositionError(f"query/doc span detection failed: {spans.reason}")

    a_ids = attack_enc["input_ids"][0].tolist()
    c_ids = control_enc["input_ids"][0].tolist()
    a_mask = attack_enc["attention_mask"][0].tolist()
    c_mask = control_enc["attention_mask"][0].tolist()
    seq_len = len(a_ids)
    A = sorted(int(p) for p in align.inserted_positions)
    q0, q1 = spans.query_span
    Q = list(range(q0, q1))
    d0, d1 = spans.doc_span_control_attack

    # --- invariants ---------------------------------------------------------
    if len(c_ids) != seq_len:
        raise PositionError("control/attack length mismatch")
    if not A or len(A) != align.n_inserted or len(set(A)) != len(A):
        raise PositionError(f"bad attack source set {A}")
    if not Q:
        raise PositionError("empty query span")
    if set(A) & set(Q):
        raise PositionError("attack positions overlap query positions")
    original_ids = tokenizer.encode(build_monot5_input(query, passage), add_special_tokens=True)
    Aset = set(A)
    if [t for i, t in enumerate(a_ids) if i not in Aset] != original_ids:
        raise PositionError("removing A from the attacked prompt does not give the clean prompt")
    if not (min(A) >= q1 and max(A) < d1):
        raise PositionError(f"attack positions {A} outside (query_end={q1}, doc_end={d1})")
    # Known, benign alignment non-uniqueness (DECISIONS.md item 33): when an
    # attack block "▁relevant :" lands right after "▁Document :", the token
    # diff may label the template ':' as inserted instead of the injected ':'
    # (identical token ids -> both labelings are valid insertion sets). We keep
    # Exp 01's labeling because the padded control masks exactly these
    # positions; we only flag it.
    boundary_shift = min(A) < d0
    if d1 != seq_len - 4:
        # tail region = "▁Relevan", "t", ":", "</s>" (Exp 06 template_positions ground truth)
        raise PositionError(f"unexpected tail length: doc_end={d1}, seq_len={seq_len}")
    for i in range(seq_len):
        if i in Aset:
            if not (c_ids[i] == tokenizer.pad_token_id and c_mask[i] == 0 and a_mask[i] == 1):
                raise PositionError(f"padded control not masked at attack position {i}")
        elif c_ids[i] != a_ids[i] or c_mask[i] != 1 or a_mask[i] != 1:
            raise PositionError(f"control/attack differ at non-attack position {i}")

    spans_list = []
    for p in A:
        if spans_list and spans_list[-1][1] == p:
            spans_list[-1][1] = p + 1
        else:
            spans_list.append([p, p + 1])

    q_ids = [a_ids[q] for q in Q]
    info = {
        "attack_source_positions": A,
        "attack_spans": spans_list,
        "query_target_positions": Q,
        "query_token_ids": q_ids,
        "query_token_strings": tokenizer.convert_ids_to_tokens(q_ids),
        "seq_len": seq_len,
        "doc_span": [d0, d1],
        "alignment_boundary_shift": bool(boundary_shift),
    }
    return control_enc, attack_enc, info
